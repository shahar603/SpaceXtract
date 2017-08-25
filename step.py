from math import fabs, sqrt, sin, cos, asin, pi, degrees, e, radians
from geographiclib import geodesic
import simplejson as json
from collections import OrderedDict
import datetime
import sys
import websocket
import trendline
from formula import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from sys import argv


##### Constants #####
STAGE_SEPARATION_ACCELERATION = 1  # [m/s^2]



class RocketEngine(object):

    def __init__(self, isp_sl, isp_vac, mass_flow_rate, nozzle_pressure, area):
        self.isp_sl = isp_sl
        self.isp_vac = isp_vac
        self.mfr = mass_flow_rate
        self.nozzle_pressure = nozzle_pressure
        self.area = area

    def isp(self, data):
        p = get_atmos_data(data['altitude'])[2]
        return self.isp_sl + ((self.isp_vac - self.isp_sl) * (self.nozzle_pressure - p) * 1e-5)

    def momentum_thrust(self, data):
        return self.mfr * self.isp(data) * 9.8

    def pressure_thrust(self, data):
        h, T, p, rho = get_atmos_data(data['altitude'])
        return (self.nozzle_pressure - p) * self.area

    def max_thrust(self, data):
        return self.momentum_thrust(data) + calc_pressure_thrust(data)


class RocketStage(object):

    def __init__(self, velocity, dry_mass, prop_mass, engines, area, coefficient_of_drag, id, next_stage):
        self.dry_mass = dry_mass
        self.prop_mass = prop_mass
        self.engines = engines
        self.cd = coefficient_of_drag
        self.id = id
        self.next_stage = next_stage
        self.area = area

        self.velocity = velocity
        self.altitude = 0
        self.acceleration = 0
        self.thrust = 0
        self.velocity_angle = pi / 2
        self.downrange_distance = 0
        self.velocity_x = 0
        self.velocity_r = 0

    def front_stage(self):
        stage = self

        while stage.next_stage is not None:
            stage = stage.next_stage

        return stage

    def pressure_thrust(self, data):
        sum = 0

        for engine in self.engines:
            sum += engine.pressure_thrust(data)

        return sum

    def coefficient_of_drag(self, data):
        return self.front_stage().cd * drag_divergent(data['velocity'])

    def mass(self):
        stage = self
        total_mass = 0

        while stage is not None:
            total_mass += stage.dry_mass + stage.prop_mass
            stage = stage.next_stage

        return total_mass

    def calc_thrust(self, data):
        weight = self.mass() * earth_gravity(self.altitude)

        if self.velocity_r == 0:
            drag = calc_drag(self)
        else:
            drag = calc_drag(self) * fabs(self.velocity_r)/self.velocity_r

        self.thrust = max(0, calc_thrust(drag, self.acceleration, self.velocity_angle, self.mass()))

    def calc_mass(self, prev_data, data):
        dt = data['time'] - prev_data['time']
        isp = self.engines[0].isp(data)

        self.prop_mass -= max(0, dt * (self.thrust - self.pressure_thrust(data)) / (isp * 9.8))

    def max_thrust(self, data):
        total_thrust = 0

        for engine in self.engines:
            total_thrust += 9.8 * engine.isp(data) * engine.mfr

        return total_thrust
        
    def update(self):
        temp_stage = self.next_stage
        
        while temp_stage is not None:
            temp_stage.velocity = self.velocity
            temp_stage.altitude = self.altitude
            temp_stage.acceleration = self.acceleration
            temp_stage.thrust = 0
            temp_stage.velocity_angle = self.velocity_angle
            temp_stage.downrange_distance = self.downrange_distance
            temp_stage.velocity_x = self.velocity_x
            temp_stage.velocity_r = self.velocity_r
            
            temp_stage = temp_stage.next_stage
            





def create_falcon9(payload_mass, dragon = True):
    merlin_1d_sl = RocketEngine(288.5, 312, 298, 101325, 0.5)
    merlin_1d_vac = RocketEngine(73, 348, 298, 101325, 1.4)

    if dragon:
        payload = RocketStage(0, 4200, payload_mass, [], 10.52, 0.2, 3, None)
    else:
        payload = RocketStage(0, 0, payload_mass, [], 21.23, 0.2, 3, None)

    stage2 = RocketStage(0, 4490, 107500, [merlin_1d_vac], 10.52, 0, 2, payload)
    stage1 = RocketStage(0, 26430, 411000, 9*[merlin_1d_sl], 10.52, 0.8, 1, stage2)

    return stage1




def rtnd(number, n):
    return int(number * 10 ** n) / 10 ** n




def remove_land_data(data):
    flag = False
    i = 0
    j = 0

    for v in data['velocity']:
        if v == 0 and not flag:
            j += 1

        if v == 0 and flag:
            break

        if v != 0:
            flag = True

        i += 1

    data['time'] = data['time'][j:i+1]
    data['velocity'] = data['velocity'][j:i+1]
    data['altitude'] = data['altitude'][j:i+1]

def add_list_to_data(data_list, key, lst):
    for i in range(len(data_list)):
        data_list[i][key] = lst[i]

def read_list(file):
    data = json.loads(file.readline())

    for x in data:
        data[x] = [data[x]]

    for line in file:
        dict = json.loads(line)

        for x in dict:
            data[x].append(dict[x])

    return data


def list_to_dict(lst):
    dict = {}

    for i in range(len(lst['time'])):
        dict[lst['time'][i]] = {}
        for key in lst:
            dict[lst['time'][i]][key] = lst[key][i]

    return dict

def list_to_data(lst):
    data_list = []
    for i in range(len(lst['time'])):
        data_list.append({
            'time': lst['time'][i],
            'velocity': lst['velocity'][i],
            'altitude': lst['altitude'][i]
        })
    return data_list


def check_stage_separation(acceleration):
    return acceleration < STAGE_SEPARATION_ACCELERATION



def fill_data(data, start, end):
    if end - start <= 0:
        return

    delta_alt = (data['altitude'][end] - data['altitude'][start])/(end - start)

    for i in range(start+1, end):
        data['altitude'][i] = data['altitude'][i-1] + delta_alt


def interpulate_data(data):
    prev_index = 0
    prev_altitude = data['altitude'][0]

    max_alt = max(data['altitude'])

    for i in range(1, len(data['time'])):
        if prev_altitude == max_alt and data['altitude'][i] != prev_altitude:
            break

        if data['altitude'][i] != prev_altitude:
            fill_data(data, prev_index, i)
            prev_index = i
            prev_altitude = data['altitude'][i]

    prev_altitude = data['altitude'][i]

    prev_index = i-1
    for i in range(i, len(data['time'])):
        if data['altitude'][i] != prev_altitude:
            fill_data(data, prev_index, i)
            prev_index = i
            prev_altitude = data['altitude'][i]



def refine_altitude(time, altitude):
    new_time = [time[0]]
    new_altitude = [altitude[0]]

    for i in range(1, len(time)):
        #if altitude[i]/1000 > 0 and altitude[i]/1000 != int(altitude[i]/1000):
        #    continue

        if new_altitude[-1] < altitude[i]:
            new_altitude.append(altitude[i])
            new_time.append(time[i])

        elif new_altitude[-1] > altitude[i] and altitude[i] != altitude[i-1]:
            new_altitude.append(altitude[i-1])
            new_time.append(time[i-1])


    for i in range(len(new_altitude)):
        if new_altitude[i] >= 100000:
            new_altitude[i] -= 500
        elif new_altitude[i] > 50:
            new_altitude[i] -= 50

    return new_time, new_altitude


def mirror_graph(x_list, y_list, delta_time):
    extended_time = []
    extended_altitude = []

    for i in range(len(x_list)):
        if x_list[i] < delta_time:
            extended_time.append(-x_list[i])
            extended_altitude.append(y_list[i])

    extended_time = extended_time[::-1]
    extended_altitude = extended_altitude[::-1]

    extended_time += x_list
    extended_altitude += y_list

    lst = []
    lst2 = []

    for i in range(len(x_list)):
        if x_list[i] > x_list[-1] - delta_time:
            lst.append((2 * x_list[-1] - x_list[i]))
            lst2.append(y_list[i])

    extended_time += lst[::-1]
    extended_altitude += lst2[::-1]

    return extended_time, extended_altitude


def abs(lst):
    return [fabs(x) for x in lst]



def fill(data, start, end):
    if end - start <= 0:
        return

    delta_alt = (data[end] - data[start])/(end - start)

    for i in range(start+1, end):
        data[i] = data[i-1] + delta_alt




def fill_xy_data(full_xlist, xlist, ylist):
    j = 0
    full_y_list = [0]

    for i in range(len(full_xlist)):
        if j < len(xlist) and full_xlist[i] == xlist[j]:
            full_y_list.append(ylist[j])
            j += 1
        else:
            full_y_list.append(full_y_list[-1])

    full_y_list = full_y_list[1:]

    xlist = [0] + xlist + [full_xlist[-1]]

    return full_y_list



def calc_approx_func(time, altitude, div = 5, der = 0 ,deg = 100, lim = 1):
    dict = {}

    for i in range(div):
        f = trendline.approx_func(
            time[max(int((i - lim) * len(time) / div), 0):min(len(time), int((i + lim) * len(time) / div))],
            altitude[max(int((i - lim) * len(time) / div), 0):min(len(time), int((i + lim) * len(time) / div))], deg)

        for j in range(der):
            f = np.polyder(f)

        dict[min(time[-1], time[int((i + 1) * len(time) / div) - 1])] = f

    return dict



def calc_approx(time, altitude, div = 5, dt = 1.0, deg = 100, lim = 1):
    time_list, altitude_list, velocity_y = [],[],[]

    for i in range(div):
        f = trendline.approx_func(
            time[max(int((i - lim) * len(time) / div), 0):min(len(time), int((i + lim) * len(time) / div))],
            altitude[max(int((i - lim) * len(time) / div), 0):min(len(time), int((i + lim) * len(time) / div))], deg)
        der = np.polyder(f)

        #print(time[int(i * len(time) / div)], min(time[-1], time[int((i + 1) * len(time) / div) - 1]))

        for t in np.arange(time[int(i * len(time) / div)], min(time[-1], time[int((i + 1) * len(time) / div) - 1]), dt):
        #for t in time:
        #    if t >= time[int(i * len(time) / div)] and t < min(time[-1], time[int((i + 1) * len(time) / div) - 1]):
            time_list.append(t)
            altitude_list.append(f(t))
            velocity_y.append(der(t))

    return time_list, velocity_y, altitude_list


def cut_graph(time_range, time, list):
    lst = [l for t, l in zip(time, list) if t >= time_range[0] and t <= time_range[-1]]
    tm = [t for t in time if t >= time_range[0] and t <= time_range[-1]]
    return tm, lst




def fill_gaps(xlist, original_x, original_y):
    i = 0
    ylist = []

    for x in original_x:

        if xlist[i] > x:
            pass

        if xlist[i] == x:
            ylist.append(original_y[i])

        if xlist[i] > x and xlist[i+1] < x:
            ylist[i] = (original_y[i] + original_y[i+1])/2

        i+=1

    return ylist

def apply(func, x):

    keys = sorted(list(func.keys()))

    if x < keys[0]:
        return func[keys[0]](x)

    for i in range(len(keys)-1):
        if keys[i] <= x <= keys[i+1]:
            return func[keys[i+1]](x)

    return func[keys[-1]](x)


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return (a / np.expand_dims(l2, axis))[0]




# Read data
file = open(argv[1], 'r')
out = open(argv[2], 'w')
data = read_list(file)

# Remove useless data
remove_land_data(data)

# Create a dictionary of the data
#dict = list_to_dict(data)

# Set time, velocity and altitude variables
time = data['time']
velocity = data['velocity']

# Convert altitude to meters
altitude = [1000 * alt for alt in data['altitude']]

# Remove noise in altitude data
new_time, new_altitude = refine_altitude(time, altitude)


# Estimate altitude data using 'f' and y velocity using 'der'
extended_time, extended_altitude = mirror_graph(new_time, new_altitude, 60)
#f = trendline.approx_func(extended_time, extended_altitude, 100)
#der = np.polyder(f)


altitude_func = calc_approx_func(extended_time, extended_altitude, deg = 100, div = 15, der = 0, lim = 2)
velocity_y_func = calc_approx_func(extended_time, extended_altitude, deg = 100, div = 15, der = 1, lim = 2)


# Removing useless data
'''
time_y, velocity_y = cut_graph(data['time'], time_list, velocity_y)
_, altitude = cut_graph(data['time'], time_list, altitude_list)

my_velocity = fill_gaps(time_y, time, velocity)

velocity_y = [ v for t, v in zip(time_y, velocity_y) if t in data['time']]
altitude = [dict[t]['altitude'] for t, a in zip(time_y, altitude) if t in data['time']]
'''

velocity_y = [apply(velocity_y_func, t) for t in time]
altitude_t = [apply(altitude_func, t) for t in time]


# Fix velocity_y and altitude_t
"""
for i in range(len(velocity_y)):
    if time[i] > 455:
        velocity_y[i] = 0


for i in range(len(altitude_t)):
    if time[i] < 2.5:
        altitude_t[i] = 0

    if time[i] > 455:
        altitude_t[i] = altitude_t[i-1]
"""


ratio = [earth_gravity(At)*max(min(Vy/Vt, 1), -1) if Vt != 0 else 0 for Vy, Vt, At in zip(velocity_y, velocity, altitude_t)]


# Fix ratio
"""
for i in range(len(ratio)):
    if time[i] > 445:
        ratio[i] = -g0
"""

dit = 10
acceleration = []

# Calculate acceleration
for i in range(len(time)-dit):
    if i % dit != 0 or time[i+dit] == time[i]:
        acceleration.append(acceleration[-1])
    else:
        acceleration.append((velocity[i+dit]- velocity[i]) / (time[i+dit] - time[i]))

acceleration += dit * [0]


# Smooth acceleration
temp_time, temp_acceleration = refine_altitude(time, acceleration)

for i in range(5):
    temp_acceleration = savgol_filter(temp_acceleration, 5, 1)

acceleration = fill_xy_data(time, temp_time, temp_acceleration)
acceleration = savgol_filter(acceleration, 31, 1)



acceleration = [a + r for a, r in zip(acceleration, ratio)]


'''
for t in data['time']:
    velocity_y.append(der(t))
    my_altitude.append(f(t))

    #if False and t < 72:
    #    velocity_y.append(dict[t]['velocity'])
    #    my_altitude.append(f(t))
    #else:
    #    velocity_y.append(min(dict[t]['velocity'], der(t)))
    #    my_altitude.append(f(t))

'''


velocity_x = []

for i in range(len(time)):
    if (velocity[i]**2 - velocity_y[i]**2) < 0:
        velocity_x.append(0)
    else:
        velocity_x.append((velocity[i]**2 - velocity_y[i]**2)**0.5)

    #if time[i] > 198:
    #    velocity_x[-1] *= -1


down = [0]
angle = []


for i in range(len(time)-1):
    down.append(down[i] + (time[i+1] - time[i])*velocity_x[i])


for t, v, y in zip(time, velocity, velocity_y):
    if v == 0 or fabs(y/v) > 1:
        angle.append(pi/2)
    else:
        angle.append(asin(y/v))


#time, angle = trendline.approx(time, angle)

'''
for i in range(len(time_y)):
    if time_y[i] > 350:
        angle[i] = angle[i-1]-(pi/180)*0.01
'''


'''
for i in range(len(time_y)-1):
    down.append(down[i] + velocity_x[i]*(time_y[i+1]-time_y[i]))


#for i in range(len(data['time'])):
#    altitude_list.append(f(data['time'][i]))



data_list = list_to_data(data)

data_list = [i for i in data_list if i['time'] in time]
velocity_t = [i for t, i in zip(data['time'], data['velocity']) if t in time]
velocity_x = [i for t, i in zip(data['time'], velocity_x) if t in time]
velocity_y = [i for t, i in zip(data['time'], velocity_y) if t in time]


acceleration_y_list = [0]
acceleration_x_list = [0]
acceleration_t = [0]

for i in range(len(velocity_y)-1):
    acceleration_y_list.append((velocity_y[i+1]-velocity_y[i])/(time[i+1]-time[i]))
    acceleration_x_list.append((velocity_x[i + 1] - velocity_x[i]) / (time[i + 1] - time[i]) + 9.8)
    acceleration_t.append((acceleration_x_list[-1]**2+acceleration_y_list[-1]**2)**0.5)

thrust_list = [0]

acceleration_list = [0]
'''

current_stage = create_falcon9(475, False)
s1 = current_stage
s2 = current_stage.next_stage
final_stage = 2


data = {
    'time': time,
    'velocity': velocity,
    'altitude': altitude_t
}

data_list = list_to_data(data)
acceleration = abs(acceleration)

delta_v_drag = [0]
gravity_loss = [0]
delta_v = [0]


for i in range(len(time)-1):
    current_stage.velocity = velocity[i]
    current_stage.altitude = altitude_t[i]
    current_stage.velocity_x = velocity_x[i]
    current_stage.velocity_r = velocity_y[i]
    current_stage.velocity_angle = angle[i]
    current_stage.acceleration = acceleration[i]

    if check_stage_separation(current_stage.acceleration) and current_stage.id < 2:
        current_stage.update()
        current_stage = current_stage.next_stage
        #current_stage.cd = 1
        #current_stage.engines = current_stage.engines[:3]

    #if time[i] > 420:
    #    current_stage.engines = current_stage.engines[:1]

    current_stage.calc_thrust(data_list[i+1])
    current_stage.calc_mass(data_list[i], data_list[i+1])

    pe = earth_potential_energy(current_stage.altitude) - earth_potential_energy(0)
    ke = current_stage.velocity**2/2



    out_dict = OrderedDict([
    ('time', data['time'][i]),
    ('velocity', data['velocity'][i]),
    ('altitude', altitude_t[i]/1000),
    ('velocity_x', velocity_x[i]),
    ('velocity_y', velocity_y[i]),
    ('acceleration', acceleration[i]),
    ('downrange_distance', down[i]/1000),
    ('angle', degrees(angle[i])),
    ('thrust', current_stage.thrust),
    ('throttle', 100 * current_stage.thrust/current_stage.max_thrust(data_list[i+1])),
    ('s1_mass', s1.prop_mass),
    ('s2_mass', s2.prop_mass),
    ('cd', current_stage.coefficient_of_drag(data_list[i])),
    ('drag', calc_drag(current_stage)/current_stage.mass()),
    ('gravity', earth_gravity(current_stage.altitude)),
    ('delta_v_drag', delta_v_drag[-1]),
    ('gravity_loss', gravity_loss[-1]),
    ('delta_v', delta_v[-1]),
    ('ke', ke),#(current_stage.prop_mass + current_stage.dry_mass) * ke),
    ('pe', pe),#(current_stage.prop_mass + current_stage.dry_mass) * pe),
    ('energy', ke+pe),#(current_stage.prop_mass + current_stage.dry_mass) * (ke+pe)),
    ('thrust', current_stage.thrust)
    ])

    delta_v_drag.append(delta_v_drag[i] + (data['time'][i+1] - data['time'][i]) * calc_drag(current_stage)/current_stage.mass())
    gravity_loss.append(gravity_loss[i] + (data['time'][i+1] - data['time'][i])*earth_gravity(current_stage.altitude))
    delta_v.append(delta_v[i] + (data['time'][i+1] - data['time'][i]) * current_stage.acceleration)

    out.write(json.dumps(out_dict) + '\n')
