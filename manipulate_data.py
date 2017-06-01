from math import fabs, sqrt, sin, asin, pi, degrees, e, radians
from geographiclib import geodesic
import simplejson as json
from skaero.atmosphere import coesa
from collections import OrderedDict
import datetime
import sys
import websocket

G = 6.67408 * 10 ** -11  # m^3 kg^-1 s^-2
EARTH_RADIUS = 6.375 * 10 ** 6  # m
EARTH_MASS = 5.972 * 10 ** 24  # kg
g0 = 9.80665 # m/s^2

# Constants
ACCELERATION_INTERVAL = 1  # s
VELOCITY_INTERVAL = 5 # s
ANGLE_INTERVAL = 5 # s

FAIRING_AREA = 25.66  # m^2
Cd = 0.2
MAX_THRUST = 7.6*10**6  # ne

S1_dry_mass = 36000
S1_prop_mass = 404600
S2_dry_mass = 5000
S2_prop_mass = 103000
PAYLOAD_MASS = 6000


PE_0 = - G * EARTH_MASS / EARTH_RADIUS

dv = 408


#ws = websocket.create_connection("ws://192.168.0.101:13900")


def check_angle(prev_data, data, prev_phi, phi):
    return True


def rtnd(number, n):
    return int(number * 10 ** n) / 10 ** n


def delta_mass(prev_data, data, thrust, isp):
    dt = data['time'] - prev_data['time']
    return dt * thrust/(isp*g0)


def mach_number(data):
    h, T, p, rho = coesa.table(1000 * data['altitude'])
    return data['velocity']/sqrt(1.66*p/rho)


def calc_thrust(Fg, Fd, current_acceleration, theta, total_mass):
    total_F = total_mass * current_acceleration

    a = 1
    if theta != 0:
        a = sin(theta)/fabs(sin(theta))

    return total_F + a*Fd + Fg*fabs(sin(theta))


def calc_g(current_data):
    return G * EARTH_MASS / (EARTH_RADIUS + 1000*current_data['altitude'])**2


def calc_PE(data):
    return -G * EARTH_MASS / (EARTH_RADIUS + 1000*data['altitude'])


def calc_drag(current_data):
    h, T, p, rho = coesa.table(1000 * current_data['altitude'])

    if type(rho) == complex:
        return 0

    return 0.5 * (Cd + mach_number(current_data))* rho * FAIRING_AREA * current_data['velocity'] ** 2


def calc_x_distance(previous_data, current_data, current_horizontal_velocity):
    R = EARTH_RADIUS + (current_data['altitude'] + previous_data['altitude']) / 2
    delta_t = current_data['time'] - previous_data['time']
    delta_x = delta_t * current_horizontal_velocity/1000
    theta = delta_x / R
    return theta * EARTH_RADIUS


def calc_acceleration(previous_data, current_data):
    delta_t = current_data['time'] - previous_data['time']
    delta_v = current_data['velocity'] - previous_data['velocity']

    return delta_v / delta_t


def calc_centripetal_acceleration(data, Vx):
    return (Vx + dv) ** 2 / (EARTH_RADIUS + 1000 * data['altitude'])


def calc_vertical_velocity(previous_data, current_data):
    delta_t = current_data['time'] - previous_data['time']
    delta_h = 1000 * (current_data['altitude'] - previous_data['altitude'])

    if delta_t == 0:
        return None

    return delta_h / delta_t


def calc_velocity_components(previous_data, current_data):
    current_vertical_velocity = calc_vertical_velocity(previous_data, current_data)

    if fabs(current_data['velocity']) < fabs(current_vertical_velocity):
        return None, None

    return sqrt(current_data['velocity']**2 - current_vertical_velocity**2), current_vertical_velocity



def analyze_capture(file, dfile):
    line = file.readline()
    
    prev_data = json.JSONDecoder(object_pairs_hook=OrderedDict).decode(line)#['data']

    prev_acceleration_data = prev_data
    prev_altitude_data = prev_data
    prev_phi_data = prev_data

    downrange_distance = 0
    phi = pi / 2
    acceleration = 0
    velocity_r = 0
    velocity_x = 0
    F = 0

    stage_sep = False

    global S1_dry_mass
    global S1_prop_mass
    global S2_dry_mass
    global S2_prop_mass
    global MAX_THRUST

    mass = S1_dry_mass + S1_prop_mass + S2_dry_mass + S2_prop_mass

    uid = 0

    while True:
        line = file.readline()#data_fron_websock()

        if len(line) == 0:
            break

        data = json.JSONDecoder(object_pairs_hook=OrderedDict).decode(line)#['data']
        if data['time'] - prev_acceleration_data['time'] >= ACCELERATION_INTERVAL:
            acceleration = calc_acceleration(prev_acceleration_data, data)

            if acceleration < -2:
                stage_sep = True

            F = calc_thrust(Fg, Fd, acceleration, phi, mass)

            if not stage_sep:
                S1_prop_mass -= max(0, delta_mass(prev_acceleration_data, data, F, 311))
                mass = S1_dry_mass + S1_prop_mass + S2_dry_mass + S2_prop_mass + PAYLOAD_MASS
                MAX_THRUST = 7.607*10**6
            else:
                S2_prop_mass -= max(0, delta_mass(prev_acceleration_data, data, F, 348))
                mass = S2_dry_mass + S2_prop_mass + PAYLOAD_MASS
                MAX_THRUST = 8.45*10**5

            prev_acceleration_data = data




        if data['time'] - prev_altitude_data['time'] >= VELOCITY_INTERVAL:
            Vx, Vr = calc_velocity_components(prev_altitude_data, data)
            
            if Vx is not None:
                velocity_x = Vx
                velocity_r = Vr


            if data['time'] >= 40 and data['time'] - prev_phi_data['time'] >= ANGLE_INTERVAL and data['velocity'] != 0 and \
            fabs(velocity_r / data['velocity']) <= 1:
                phi = asin(velocity_r / data['velocity'])
                prev_phi_data = data
            
            downrange_distance += calc_x_distance(prev_altitude_data, data, velocity_x)
            prev_altitude_data = data


        PE = calc_PE(data) - PE_0
        KE_x = velocity_x ** 2 / 2
        KE_r = velocity_r ** 2 / 2

        Fd = calc_drag(data)
        Fg = mass * calc_g(data)

        centripetal_acceleration = calc_centripetal_acceleration(data, velocity_x)
        g_force = acceleration + (Fg/mass)*sin(phi)


        dict_data = OrderedDict([
            ('time', rtnd(data['time'], 2)),
            ('velocity', rtnd(data['velocity'], 2)),
            ('altitude', rtnd(data['altitude'], 2)),
            ('acceleration', rtnd(acceleration, 2)),
            ('force', 100 * rtnd(F, 2) / MAX_THRUST),
            ('s1_mass', rtnd(S1_prop_mass, 2)),
            ('s2_mass', rtnd(S2_prop_mass, 2)),
            ('vertical_velocity', rtnd(velocity_r, 2)),
            ('horizontal_velocity', rtnd(velocity_x, 2)),
            ('angle', rtnd(degrees(phi), 2)),
            ('downrange_distance', downrange_distance),
            ('gravity', rtnd(Fg, 2) / mass),
            ('g_force', rtnd(g_force, 2)),
            ('drag', rtnd(Fd, 2) / mass),
            ('pe', rtnd(PE, 2)),
            ('horizontal_ke', rtnd(KE_x, 2)),
            ('vertical_ke', rtnd(KE_r, 2)),
            ('ce', rtnd(centripetal_acceleration, 2)),
            ('thrust', rtnd(F, 2))
        ])

        print(json.dumps(OrderedDict(
            [('uid', uid), ('timestamp', datetime.datetime.utcnow().isoformat() + 'Z'), ('data', dict_data)])))

        dfile.write(json.dumps(dict_data) + '\n')

        prev_data = data
        uid += 2


def main():
    file = open(sys.argv[1], 'r')
    dfile = open('a.txt', 'w')
    analyze_capture(file, dfile)


# Entry point to the program.
if __name__ == '__main__':
    main()
