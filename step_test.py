import numpy as np
import json

from pandas.util.testing import all_timeseries_index_generator
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import scipy
import matplotlib.pyplot as plt
from math import fabs, pi, asin, sin, log, degrees, acos
import trendline
from scipy.interpolate import interp1d
import sys
import Telemetry.Orbit as orbit
from collections import OrderedDict
from skaero.atmosphere import coesa
import math
import scipy.signal as ss



def read_list(file):
    data = json.loads(file.readline())

    for x in data:
        data[x] = [data[x]]

    for line in file:
        dict = json.loads(line)

        for x in dict:
            data[x].append(dict[x])

    return data


def gravity(altitude):
    return 4*10**14/(6.375*10**6 + altitude)**2



def refine_graph(x, y):
    new_x = [x[0]]
    new_y = [y[0]]


    for i in range(1, len(x)):
        if y[i] != new_y[-1]:
            new_x.append(x[i])
            new_y.append(y[i])


    return new_x, new_y



def refine_altitude(time, altitude, blur=True):
    new_time = [time[0]]
    new_altitude = [altitude[0]]

    for i in range(1, len(time)):
        if blur and altitude[i]/1000 > 0 and altitude[i]/1000 != int(altitude[i]/1000):
            continue

        if new_altitude[-1] < altitude[i]:
            new_altitude.append(altitude[i])
            new_time.append(time[i])

        elif new_altitude[-1] > altitude[i] and altitude[i] != altitude[i-1]:
            new_altitude.append(altitude[i-1])
            new_time.append(time[i-1])

    if new_time[-1] != time[-1]:
        new_time.append(time[-1])
        new_altitude.append(altitude[-1])


    for i in range(len(new_altitude)):
        if new_altitude[i] >= 100000:
            new_altitude[i] -= 500
        elif new_altitude[i] > 50:
            new_altitude[i] -= 50


    return new_time, new_altitude


def derivative(x_axis, y_axis, dx):
    """
    Calculate the derivative of f(x)
    :param x_axis: list of the x axis data
    :param y_axis: list of the y axis data
    :param dx: difference of x axis
    :return: f'(x)
    """
    der = (dx//2+dx%2)*[0]

    for i in range(len(x_axis) - dx):
        der.append((y_axis[i + dx] - y_axis[i]) / (x_axis[i + dx] - x_axis[i]))

    der += dx // 2 * [der[-1]]

    for i in range(dx//2+dx%2):
        der[i] = der[dx//2+dx%2+1]

    return der


def flip_direction(x_axis, y_axis, flip):
    for i in range(1, len(x_axis)):
        if x_axis[i] > flip:
            y_axis[i] = -y_axis[i]


def find_MECO(acceleration):
    return np.where(acceleration < 5)[0][0]


def find_altitude_graph(time, altitude, blur=False, interp=False):
    altitude = np.multiply(1000, altitude)
    temp_time, temp_altitude = refine_altitude(time, altitude, blur=blur)

    f = interp1d(temp_time, temp_altitude, kind=3)

    global altitude_time, ALTITUDE_INTERVAL

    t = np.arange(temp_time[0], temp_time[-1], ALTITUDE_INTERVAL)

    if interp:
        return np.interp(altitude_time, t, f(t))

    return np.interp(altitude_time, temp_time, temp_altitude)


def pythagoras(hypotenuse, leg):
    return [max(0, h**2-l**2)**0.5 for h, l in zip(hypotenuse, leg)]


def final_altitude(velocity, altitude):
    u = 4*10**14
    return -u/(velocity**2/2 - u/(altitude+6.375*10**6)) - 6.375*10**6




def find_angle_graph(velocity, vertical_velocity, interp=False):
    angle = []

    for i in range(len(velocity)):
        if velocity[i] == 0:
            angle.append(angle[-1])
        else:
            ratio = max(-1, min(vertical_velocity[i] / velocity[i], 1))
            angle.append(asin(ratio))

    angle = savgol_filter(angle, 5, 1)

    if interp:
        angle = savgol_filter(angle, 11, 1)
        return ss.medfilt(angle, kernel_size=7)

    return angle






def find_downrange_graph(time, horizontal_velocity, d0=0, dx=1):
    downrange_distance = [d0]

    for i in range(dx, len(time)):
        downrange_distance.append(downrange_distance[-1] +
                                  (time[i] - time[i - dx]) * (
                                  horizontal_velocity[i] + horizontal_velocity[i - dx]) / 2
                                  )

    return downrange_distance


def find_flip_point(y_axis, y_der, thresh = 1, dx = 10, start_index = 0):
    small = [i for i, y in enumerate(y_axis) if fabs(y) < thresh]

    if len(small) == 0:
        return None

    for i in small:
        if i >= dx and i < len(y_axis)-dx:
            if y_der[i-dx] * y_der[i+dx] < 0:
                return i

    return None




def smooth_altitude_with_velocity(altitude):
    new_altitude = []

    velocity_altitude = find_downrange_graph(altitude_time, velocity, d0=altitude[0])
    velocity_altitude_rev = find_downrange_graph(altitude_time, velocity[::-1], d0=altitude[-1])

    for i in range(len(altitude_time)):
        if velocity_altitude[i] - altitude[i] < 0 or velocity_altitude[i] < 1000:
            new_altitude.append(velocity_altitude[i])
            vertical_velocity[i] = velocity[i]

        elif (velocity_altitude_rev[-(i + 1)] - altitude[i] < 0 and fabs(altitude[i]) < 10000) or fabs(altitude[i]) < 1000:
            new_altitude.append(altitude[i])
            # new_altitude.append(velocity_altitude[-1] - velocity_altitude[i])
            vertical_velocity[i] = np.sign(vertical_velocity[i - 1]) * velocity[i]
        else:
            new_altitude.append(altitude[i])

    return new_altitude





def find_gap(data):
    delta_time = [0] + [data['time'][i] - data['time'][i - 1] for i in range(1, len(data['time']))]
    max_index = np.argmax(delta_time)

    start = data['time'][max_index - 1]
    end = data['time'][max_index]

    if max_index == 0 or end - start < MIN_COAST:
        return -1

    return max_index-1


def find_flip_point2(time, velocity, vertical_velocity, acceleration):
    dv = np.subtract(velocity, vertical_velocity)
    meco_time = time[find_MECO(acceleration)]

    flip_time = [t for t, v in zip(time, dv) if v <= 0 and meco_time < t < meco_time + 120]

    if len(flip_time) > 0:
        return flip_time[0]
    return None


def acceleration_func(x, Isp, m_dot, m0):
    return 9.8*Isp*9*m_dot/(m0-9*m_dot*x)

def velocity_func(x, Ve, m_dot, m0, g0):
    return Ve*np.log(m0/(m0-9*m_dot*x))  - g0


def get_atmos_data(altitude):
    """
    This function calculates data about the atmosphere
    :param altitude: Distance from the surface of the Earth [m]
    :return:
    """
    h, T, P, rho = coesa.table(altitude)

    if np.isnan(P) or type(P) == complex:
        P = 0
        rho = 0

    return h, T, P, rho


def get_q(velocity, altitude):
    return 0.5*get_atmos_data(1000*altitude)[-1]*velocity**2






MAX_STRING_LENGTH = 1024

VELOCITY_INTERVAL = 0.1
ALTITUDE_INTERVAL = 1
MIN_COAST = 600


# Get the data
file = open(sys.argv[1], 'r')
data = read_list(file)

# Set the end points of the data
start = data['time'][0]
end = data['time'][-1]


end_index = find_gap(data)


if end_index is not None and end_index != -1 and False:
    start = data['time'][0]
    end = data['time'][end_index]

    altitude_time = np.arange(start, end, ALTITUDE_INTERVAL)
    altitude = find_altitude_graph(data['time'][:end_index], data['altitude'][:end_index])

    der = derivative(altitude_time, altitude, dx = 5)
    crit = []

    last_der = 0
    i = 0
    for t,a in zip(altitude_time, der):
        if fabs(a) < 3 and i - last_der > 60:
            crit.append(i)
            last_der = i
        i+=1


    coast_start = crit[len(crit)//2]


    # Set time interval for the data
    velocity_time = np.arange(start, end, VELOCITY_INTERVAL)
    altitude_time = np.arange(start, end, ALTITUDE_INTERVAL)

    # Smooth altitude and velocity data
    altitude = find_altitude_graph(data['time'], data['altitude'])
    velocity = np.interp(altitude_time, data['time'], data['velocity'])

    coast_start_time = altitude_time[coast_start]
    coast_end_time = data['time'][end_index+1]


    dict = {
        'time': [],
        'velocity': [],
        'altitude': []
    }


    for i, t in enumerate(data['time']):
        if data['time'][0] <= t <= coast_start_time:
            dict['time'].append(t)
            dict['velocity'].append(data['velocity'][i])
            dict['altitude'].append(data['altitude'][i])
        else:
            break


    new_data = orbit.calculate(
        data['altitude'][i-1]*1000,
        data['altitude'][end_index+1]*1000,
        velocity[coast_start-1],
        coast_start_time,
        coast_end_time)


    dict['time'] += new_data['time']
    dict['velocity'] += new_data['velocity']
    dict['altitude'] += new_data['altitude']

    for i, t in enumerate(data['time']):
        if coast_end_time < t <= data['time'][-1]:
            dict['time'].append(t)
            dict['velocity'].append(data['velocity'][i])
            dict['altitude'].append(data['altitude'][i])

    data = dict


start = data['time'][0]
end = data['time'][end_index]

if end_index is not None and end_index != -1:
    data['time'] = data['time'][:end_index+1]
    data['velocity'] = data['velocity'][:end_index+1]
    data['altitude'] = data['altitude'][:end_index+1]

flip = len(sys.argv) >= 4

if data['altitude'][-1] - data['altitude'][0] < 50:
    stage = 1
else:
    stage = 2

# Set time interval for the data
velocity_time = np.arange(start, end, VELOCITY_INTERVAL)
altitude_time = np.arange(start, end, ALTITUDE_INTERVAL)

# Smooth altitude and velocity data
altitude = find_altitude_graph(data['time'], data['altitude'], interp=False)
altitude = np.maximum(0, altitude)


velocity = np.interp(altitude_time, *refine_graph(data['time'], data['velocity']))


# Find vertical velocity
vertical_velocity = derivative(altitude_time, altitude, 3)
vertical_velocity = [np.sign(vv)*min(fabs(vv), vt) for vv, vt in zip(vertical_velocity, velocity)]


altitude = smooth_altitude_with_velocity(altitude)

angle = find_angle_graph(velocity, vertical_velocity, interp=(stage == 2))


med_vertical_velocity = ss.medfilt(vertical_velocity, kernel_size=7)
wie_vertical_velocity = ss.wiener(np.array(med_vertical_velocity))
sav = savgol_filter(wie_vertical_velocity, 5, 1)
vertical_velocity = sav



acceleration = derivative(altitude_time, velocity, 1)

med_acceleration = ss.medfilt(acceleration, kernel_size=7)
wie_acceleration = ss.wiener(np.array(med_acceleration))
acceleration = np.add(wie_acceleration, np.multiply(list(map(gravity, altitude)), np.sin(angle)))
acceleration = savgol_filter(acceleration, 3, 1)


FLIP_TIME = find_flip_point2(altitude_time, velocity, vertical_velocity, acceleration)

horizontal_velocity = pythagoras(velocity, vertical_velocity)


horizontal_velocity_org = horizontal_velocity


if FLIP_TIME is not None:
    flip_direction(altitude_time, horizontal_velocity, FLIP_TIME)


med_horizontal_velocity = ss.medfilt(horizontal_velocity, kernel_size=15)
wie_horizontal_velocity = ss.wiener(np.array(med_horizontal_velocity))
sav = savgol_filter(wie_horizontal_velocity, 15, 3)
horizontal_velocity = np.minimum(velocity, sav)


downrange_distance = find_downrange_graph(altitude_time, horizontal_velocity)

index = find_MECO(acceleration)
v0 = vertical_velocity[index]
a0 = altitude[index]

#index = np.argmax(altitude, axis=0)
#apogee = velocity_time[index]

#index -= 100
#p0=[288, 300, 5.5*10**5]
#param_bounds=([200, 250, 4.5*10**5], [400, 400, 6.5*10**5])
#popt, pcov = curve_fit(acceleration_func, velocity_time[:index], acceleration[:index], p0=p0, bounds=param_bounds)
#print('Isp = {}, m_dot = {}, mass = {:.2f}'.format(*popt))

downrange_distance = np.multiply(0.001, downrange_distance)
altitude = np.multiply(0.001, altitude)
altitude = np.maximum(0, altitude)


out_file = open(sys.argv[2], 'w')
out_string = ''

velocity_time = np.subtract(velocity_time, data['time'][0])

index = np.where(np.abs(acceleration) > 1)

for i in range(min(len(altitude_time)-10, index[0][-1]+10)):
    data_dict = OrderedDict([
        ('time', float('{:.3f}'.format(altitude_time[i]))),
        ('velocity', float('{:.3f}'.format(velocity[i]))),
        ('altitude', float('{:.3f}'.format(altitude[i]))),
        ('velocity_y', float('{:.3f}'.format(vertical_velocity[i]))),
        ('velocity_x', float('{:.3f}'.format(horizontal_velocity[i]))),
        ('acceleration', float('{:.3f}'.format(acceleration[i]))),
        ('downrange_distance', float('{:.3f}'.format(downrange_distance[i]))),
        ('angle', float('{:.3f}'.format(degrees(angle[i])))),
        ('q', get_q(velocity[i], altitude[i]))
    ])

    out_string += str(json.dumps(data_dict)) + '\n'

    if len(out_string) >= MAX_STRING_LENGTH:
        out_file.write(out_string)
        out_string = ''

out_file.write(out_string)



