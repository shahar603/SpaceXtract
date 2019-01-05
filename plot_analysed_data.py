import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import style
from matplotlib.lines import Line2D
from sys import argv
import os
from os.path import splitext, basename
import json
from skaero.atmosphere import coesa
from collections import OrderedDict
import math



unit_dict = {
    3: 'k',
    6: 'M',
    9: 'G'
}

events = None


events_to_text = {
    'maxq': 'Max-Q',
    'throttle_down_start': 'Throttle Down',
    'throttle_down_end': 'Throttle Up',
    'meco': 'MECO',
    'boostback_start': 'Boostback Burn Ignition',
    'boostback_end': 'Boostback Burn Shutdown',
    'apogee': 'Apogee',
    'entry_start': 'Entry Burn Ignition',
    'entry_end': 'Entry Burn Shutdown',
    'landing_start': 'Landing Burn Ignition',
    'landing_end': 'Landing Burn Shutdown',
    'ses1': 'SES',
    'seco1': 'SECO',
    'ses2': None,
    'seco2': None
}



font = {'family'  : 'normal',
        'size'    : 14}


def Test2(rootDir):
    paths = []
    for lists in os.listdir(rootDir):
        path = os.path.join(rootDir, lists)
        if path.endswith('json'):
            paths.append(path)
    return paths



def read_list(file):
    data = json.loads(file.readline())

    for x in data:
        data[x] = [data[x]]

    for line in file:
        dict = json.loads(line)

        for x in dict:
            data[x].append(dict[x])

    return data

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


def gravitational_acceleration(altitude):
    return 4*10**14/np.power(6.375*10**6 + np.multiply(1000, altitude), 2)


def potential_energy(altitude):
    return 4*10**14/(6.375*10**6) - 4*10**14/(1000*altitude+6.375*10**6)


def centrifugal_acceleration(velocity, altitude):
    temp = 6.375*10**6*np.ones(np.size(altitude)) + np.multiply(1000,altitude)
    return np.divide(np.power(velocity, 2),temp)


def velocity_to_kinetic_energy(velocity):
    return velocity**2/2


def animate(i):
    graph_data = open(argv[1],'r')
    data = read_list(graph_data)
    #ax1.clear()
    #ax1.plot(data['time'], data['velocity'])

def get_base(*args):
    arr = np.array([])

    for i in range(len(args)):
        arr = np.union1d(arr, args[i])
    return int(3*(np.log10(np.max(arr)) // 3))

dpi = 200
figure_size = (19.2, 10.8)



def add_horizontal_lines(data, events, x_key, y_key, rotation=0):
    max_x = max(data[x_key])
    min_x = min(data[x_key])

    ratio_positive = math.fabs(min_x)/(math.fabs(min_x)+math.fabs(max_x))

    for key in events:
        if events[key] is not None:
            if events[key] >= len(data[x_key]):
                continue

            if np.sign(data[x_key][events[key]]) > 0:
                xmin = ratio_positive
                xmax = 1
                x = 0.65*max_x
            else:
                xmin = 0
                xmax = ratio_positive
                x = min_x

            plt.text(x, data[y_key][int(events[key])], events_to_text[key], fontsize=15, rotation=rotation)
            plt.axhline(y=data[y_key][int(events[key])], xmin=xmin, xmax=xmax, color='black', linestyle='--')


def add_lines(events, y, rotation=-90):
    for key in events:
        if events[key] is not None:
            plt.text(int(events[key]), y, events_to_text[key], fontsize=15, rotation=rotation)
            plt.axvline(x=int(events[key]), color='black', linestyle='--')


def myround(x, base=5):
    return int(base * round(float(x)/base))


def add_points(*args):
    dict = {}

    for event in args:
        if event in events and event is not None and events[event] is not None:
            dict[events[event]] = event

    return dict



def divide_to_lines(x_axis, y_axis, *args):
    dict = []

    dict.append((x_axis[:events[args[0]]], y_axis[:events[args[0]]]))

    for i in range(1,len(args)):
        if events[args[i-1]] is None or events[args[i]] is None:
            continue

        start = events[args[i-1]]-1
        end = events[args[i]]

        dict.append((x_axis[start:end], y_axis[start:end]))

    if events[args[-1]] is not None:
        dict.append((x_axis[events[args[-1]]-1:], y_axis[events[args[-1]]-1:]))

    return dict


def create_images(path, lst, events):
    plt.figure('Velocity', figsize=figure_size, tight_layout=True)
    for name, data in lst:
        plt.plot(data['time'], data['velocity'], label='Velocity [m/s]')
        plt.plot(data['time'], data['velocity_y'], label='Vertical Velocity [m/s]')
        plt.plot(data['time'], data['velocity_x'], label='Horizontal Velocity [m/s]')

        #add_lines(events)

        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')

    plt.legend()
    plt.grid()
    plt.savefig(path + '\\' + 'Velocity.png', format='png', dpi=dpi)


    plt.figure('Velocity Abs', figsize=figure_size, tight_layout=True)
    for name, data in lst:
        plt.plot(data['time'], data['velocity'], label='Velocity [m/s]')
        plt.plot(data['time'], np.abs(data['velocity_y']), label='Vertical Velocity [m/s]')
        plt.plot(data['time'], np.abs(data['velocity_x']), label='Horizontal Velocity [m/s]')

        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')

    plt.legend()
    plt.grid()
    plt.savefig(path + '\\' + 'Velocity Abs.png', format='png', dpi=dpi)



    plt.figure('Altitude', figsize=figure_size, tight_layout=True)
    for name, data in lst:
        plt.plot(data['time'], data['altitude'])

        plt.xlabel('Time [s]')
        plt.ylabel('Altitude [km]')

    plt.grid()
    plt.savefig(path + '\\' + 'Altitude.png', format='png', dpi=dpi)



    plt.figure('Acceleration', figsize=figure_size, tight_layout=True)
    for name, data in lst:
        plt.plot(data['time'], data['acceleration'])

        #add_lines(events)

        plt.xlabel('Time [s]')
        plt.ylabel('Acceleration [m/s^2]')

    plt.grid()
    plt.savefig(path + '\\' + 'Acceleration.png', format='png', dpi=dpi)


    plt.figure('Gravity', figsize=figure_size, tight_layout=True)
    for name, data in lst:
        ce = centrifugal_acceleration(data['velocity_x'], data['altitude'])
        ge = gravitational_acceleration(data['altitude'])
        plt.plot(data['time'], ce)
        plt.plot(data['time'], ge)
        plt.plot(data['time'], np.subtract(ge, ce))

        plt.xlabel('Time [s]')
        plt.ylabel('Acceleration [m/s^2]')

    plt.grid()
    plt.savefig(path + '\\' + 'Gravity.png', format='png', dpi=dpi)


    plt.figure('Acceleration vs Altitude', figsize=figure_size, tight_layout=True)
    for name, data in lst:
        max_index = np.argmax(data['altitude'])

        plt.plot(data['acceleration'][:max_index], data['altitude'][:max_index], label='Ascent', color='black')
        plt.plot(data['acceleration'][max_index:], data['altitude'][max_index:], label='Decent', color='red')

        plt.xlabel('Acceleration [m/s^2]')
        plt.ylabel('Altitude [km]')

    plt.grid()
    plt.savefig(path + '\\' + 'Acceleration vs Altitude.png', format='png', dpi=dpi)


    plt.figure('Downrange Distance', figsize=figure_size, tight_layout=True)
    for name, data in lst:
        plt.plot(data['time'], data['downrange_distance'])

        plt.xlabel('Time [s]')
        plt.ylabel('Downrange Distance [km]')

    plt.grid()
    plt.savefig(path + '\\' + 'Downrange Distance.png', format='png', dpi=dpi)


    plt.figure('Flight Profile', figsize=(2*19.2, 2*10.8), tight_layout=True)
    plt.gca().set_aspect('equal', adjustable='box')


    for name, data in lst:
        major_ticks_x = np.arange(myround(min(data['downrange_distance'])),
                                myround(max(data['downrange_distance'])),
                                10)

        major_ticks_y = np.arange(myround(min(data['altitude'])),
                                myround(max(data['altitude'])),
                                10)
        """
        plt.plot(data['downrange_distance'][:events['meco']], data['altitude'][:events['meco']], 'r')
        plt.plot(data['downrange_distance'][events['meco']-1:events['boostback_start']], data['altitude'][events['meco']-1:events['boostback_start']], 'b')
        plt.plot(data['downrange_distance'][events['boostback_start']-1:events['boostback_end']], data['altitude'][events['boostback_start']-1:events['boostback_end']], 'r')
        plt.plot(data['downrange_distance'][events['boostback_end']-1:events['entry_start']], data['altitude'][events['boostback_end']-1:events['entry_start']], 'b')
        plt.plot(data['downrange_distance'][events['entry_start']-1:events['entry_end']], data['altitude'][events['entry_start']-1:events['entry_end']], 'r')
        plt.plot(data['downrange_distance'][events['entry_end']-1:events['landing_start']], data['altitude'][events['entry_end']-1:events['landing_start']], 'b')
        plt.plot(data['downrange_distance'][events['landing_start']-1:events['landing_end']], data['altitude'][events['landing_start']-1:events['landing_end']], 'r')
        """

        plt.plot(data['downrange_distance'], data['altitude'])

        plt.xlabel('Downrange Distance [km]')
        plt.ylabel('Altitude [km]')


        plt.xticks(major_ticks_x)
        plt.yticks(major_ticks_y)

        plt.grid()
        plt.savefig(path + '\\' + 'Flight Profile.png', format='png', dpi=dpi)


    plt.figure('Velocity Angle(Time)', figsize=figure_size, tight_layout=True)

    for name, data in lst:
        plt.plot(data['time'], data['angle'])

    plt.xlabel('Time [s]')
    plt.ylabel('Angle [degrees]')

    plt.grid()
    plt.savefig(path + '\\' + 'Velocity Angle.png', format='png', dpi=dpi)


    plt.figure('Altitude Vs Velocity', figsize=figure_size, tight_layout=True)
    for name, data in lst:
        max_index = np.argmax(data['altitude'])

        plt.plot(data['velocity'][:max_index], data['altitude'][:max_index], label='Ascent', color='black')
        plt.plot(data['velocity'][max_index:], data['altitude'][max_index:], label='Decent', color='red')

        plt.xlabel('Velocity [m/s]')
        plt.ylabel('Altitude [km]')
    plt.legend()
    plt.grid()
    plt.savefig(path + '\\' + 'Altitude Vs Velocity.png', format='png', dpi=dpi)


    plt.figure('Specific Mechanical Energy', figsize=figure_size, tight_layout=True)
    for name, data in lst:
        pe = np.array(list(map(potential_energy, data['altitude'])))
        ke = np.array(list(map(velocity_to_kinetic_energy, data['velocity'])))

        base = get_base(np.add(pe, ke))
        unit = 10 ** -base

        plt.plot(data['time'], unit * pe, label='Potential Energy')
        plt.plot(data['time'], unit * ke, label='Kinetic Energy')
        plt.plot(data['time'], unit * np.add(ke, pe), label='Total Energy')

        plt.xlabel('Time [s]')
        plt.ylabel('Specific Energy [{}J/kg]'.format(unit_dict[base]))
    plt.legend()
    plt.grid()
    plt.savefig(path + '\\' + 'Specific Mechanical Energy.png', format='png', dpi=dpi)

    plt.figure('Kinetic Energy', figsize=figure_size, tight_layout=True)
    for name, data in lst:
        tke = np.array(list(map(velocity_to_kinetic_energy, data['velocity'])))
        xke = np.array(list(map(velocity_to_kinetic_energy, data['velocity_x'])))
        yke = np.array(list(map(velocity_to_kinetic_energy, data['velocity_y'])))

        base = get_base(tke, xke, yke)
        unit = 10 ** -base

        plt.plot(data['time'], unit * tke, label='Total Kinetic Energy')
        plt.plot(data['time'], unit * xke, label='Horizontal Kinetic Energy')
        plt.plot(data['time'], unit * yke, label='Vertical Kinetic Energy')

        plt.xlabel('Time [s]')
        plt.ylabel('Specific Energy [{}J/kg]'.format(unit_dict[base]))
    plt.legend()
    plt.grid()
    plt.savefig(path + '\\' + 'Kinetic Energy.png', format='png', dpi=dpi)

    plt.figure('Aerodynamic pressure', figsize=figure_size, tight_layout=True)
    for name, data in lst:
        plt.plot(data['time'], data['q'])

    plt.savefig(path + '\\' + 'Aerodynamic pressure.png', format='png', dpi=dpi)



def create_plot(acc, color_list=['r', 'b']):
    for i, a in enumerate(acc):
        plt.plot(a[0], a[1], color_list[i % len(color_list)], linewidth=2)



def ceil_to(num, interval):
    return interval * math.ceil(num / interval)

def floor_to(num, interval):
    return interval * math.floor(num / interval)

def get_ticks_major(axis, interval):
    return np.append(
        np.arange(ceil_to(min(axis), interval), 1, interval),
        np.arange(0, floor_to(max(axis), interval) + 1, interval))

def get_ticks_minor(axis, interval):
    return np.append(
        np.arange(floor_to(min(axis), interval), 1, interval),
        np.arange(0, ceil_to(max(axis), interval) + 1, interval))


def get_ticks_by_density(ticks, frq):
    return np.linspace(ticks[0], ticks[-1], frq)


def create_grid(ax, x, y, major_x_interval, major_y_interval, minor_x_interval=0, minor_y_interval=0):
    major_x = get_ticks_minor(x, major_x_interval)
    major_y = get_ticks_minor(y, major_y_interval)

    ax.set_xticks(major_x)
    ax.set_yticks(major_y)

    if minor_x_interval != 0:
        minor_x = get_ticks_minor(x, minor_x_interval)
        ax.set_xticks(minor_x, minor=True)

    if minor_y_interval != 0:
        minor_y = get_ticks_minor(y, minor_y_interval)
        ax.set_yticks(minor_y, minor=True)

    ax.grid(which='major')
    ax.grid(which='minor', alpha=0.5)



def real_create_grid(ax, x, y,
                     major_x_interval=[0.1, 0.2, 0.25, 0.5, 1, 2.5, 5, 10, 25, 50, 100, 200, 250, 500, 1000],
                     major_y_interval=[0.1, 0.2, 0.25, 0.5, 1, 2.5, 5, 10, 25, 50, 100, 200, 250, 500, 1000],
                     num_x=20,
                     num_y=20):

    major_x = ticks_spacing_and_number(x, major_x_interval, num_x)
    major_y = ticks_spacing_and_number(y, major_y_interval, num_y)

    ax.set_xticks(major_x)
    ax.set_yticks(major_y)
    plt.grid()


def lines_with_color(ax, color_events, t, x, y):
    color_dict = {}
    keys = list(color_events.keys())

    for key in keys:
        color_dict[key] = {'x': [], 'y': []}

    count = 0
    prev_key = keys[0]
    for i in range(len(t)):
        for key in keys:
            if key > t[i]:
                if key != prev_key and count > 0:
                    color_dict[key]['x'].append(x[i - 1])
                    color_dict[key]['y'].append(y[i - 1])

                color_dict[key]['x'].append(x[i])
                color_dict[key]['y'].append(y[i])
                count += 1
                prev_key = key
                break

    width = 2

    for key in keys:
        ax.plot(color_dict[key]['x'], color_dict[key]['y'], color=color_events[key], linewidth=width)




def events_to_colordict(events):
    color_events = OrderedDict([])

    for key in engine_event_dict:
        if events[key] is not None:
            color_events[events[key]] = engine_event_dict[key]

    color_events[10 ** 5] = color_events[list(color_events.keys())[-2]]

    return color_events




def ticks_spacing_and_number(value, spacing, num):
    tick_list = [get_ticks_minor(value, space) for space in spacing]
    cost = list(map(lambda x: math.fabs(num-len(x)), tick_list))
    return tick_list[cost.index(min(cost))]



def merge(*args):
    new_list = []

    for lst in args:
        new_list += list(lst)

    return new_list


time_major_interval = 25
time_minor_interval = 10

def graph(lst, save=False, anotate=False):
    mpl.rcParams['font.size'] = 13
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['axes.titlesize'] = 20
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['legend.loc'] = 'best'



    if anotate:
        suffix = ' annotated'
    else:
        suffix = ''


    ##################### Velocity ####################

    fig = plt.figure('Velocity', figsize=figure_size, tight_layout=True)
    ax = fig.add_subplot(1,1,1)

    x = []
    y = []

    for name, data, events in lst:
        ax.plot(data['time'], data['velocity'], label='Velocity [m/s]')
        ax.plot(data['time'], data['velocity_y'], label='Vertical Velocity [m/s]')
        ax.plot(data['time'], data['velocity_x'], label='Horizontal Velocity [m/s]')

        x = merge(x, data['time'])
        y = merge(y, data['velocity'], data['velocity_y'], data['velocity_x'])


    #add_lines(events, max(y)//2)
    real_create_grid(ax, x, y)
    ax.set_xlim(0, max(x)+1)
    ax.set_ylim(1.05*min(y), 1.05*max(y))

    #create_grid(ax, x, y, time_major_interval, 1000, time_minor_interval, 250)
    plt.title('Velocity vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.legend()


    if save:
        plt.savefig(os.path.dirname(name) + os.path.sep + 'Velocity.png', format='png', dpi=dpi)

    ##################### Absolute Velocity ####################



    fig = plt.figure('Velocity Abs', figsize=figure_size, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    x = []
    y = []

    for name, data, events in lst:
        ax.plot(data['time'], data['velocity'], label='|Velocity| [m/s]')
        ax.plot(data['time'], np.abs(data['velocity_y']), label='|Vertical Velocity| [m/s]')
        ax.plot(data['time'], np.abs(data['velocity_x']), label='|Horizontal Velocity| [m/s]')

        x = merge(x, data['time'])
        y = merge(y, data['velocity'], np.abs(data['velocity_y']), np.abs(data['velocity_x']))

    if anotate:
        add_lines(events, max(y)//2)

    real_create_grid(ax, x, y)
    ax.set_xlim(0, max(x)+1)
    ax.set_ylim(1.05*min(y), 1.05*max(y))

    #create_grid(ax, x, y, time_major_interval, 1000, time_minor_interval, 250)
    plt.title('|Velocity| vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.legend()
    if save:
        plt.savefig(os.path.dirname(name) + os.path.sep + 'Velocity Abs' + suffix + '.png', format='png', dpi=dpi)






    ##################### Altitude vs Time ####################

    fig = plt.figure('Altitude', figsize=figure_size, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    x = []
    y = []


    for name, data, events in lst:
        ax.plot(data['time'], data['altitude'])
        #lines_with_color(ax, data['time'], data['time'], data['altitude'])
        x = merge(x, data['time'])
        y = merge(y, data['altitude'])

    if anotate:
        add_lines(events, max(y)//2)
    real_create_grid(ax, x, y)
    ax.set_xlim(0, max(x)+1)
    ax.set_ylim(1.05*min(y), 1.05*max(y))

    #create_grid(ax, x, y, time_major_interval, 25, time_minor_interval, 5)
    plt.xlabel('Time [s]')
    plt.ylabel('Altitude [km]')
    plt.title('Altitude vs Time')
    if save:
        plt.savefig(os.path.dirname(name) + os.path.sep + 'Altitude' + suffix + '.png', format='png', dpi=dpi)





    ##################### Acceleration vs Time ####################


    fig = plt.figure('Acceleration', figsize=figure_size, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    x = []
    y = []

    """
    for name, data in lst:
        lines_with_color(ax, data['time'], data['time'], data['acceleration'])
        x = merge(x, data['time'])
        y = merge(y, data['acceleration'])
    """

    for name, data, events in lst:
        ax.plot(data['time'], data['acceleration'])
        x = merge(x, data['time'])
        y = merge(y, data['acceleration'])

    if anotate:
        add_lines(events, max(y)//2)
    #create_grid(ax, x, y, time_major_interval, 5, time_minor_interval, 1)
    real_create_grid(ax, x, y)
    ax.set_xlim(0, max(x)+1)
    ax.set_ylim(1.05*min(y), 1.05*max(y))
    plt.title('Acceleration vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s^2]')

    if save:
        plt.savefig(os.path.dirname(name) + os.path.sep + 'Acceleration' + suffix + '.png', format='png', dpi=dpi)






    ##################### Forces on the rocket ####################

    fig = plt.figure('External Forces vs Time', figsize=figure_size, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    x = []
    y = []

    for name, data, events in lst:
        ce = centrifugal_acceleration(data['velocity_x'], data['altitude'])
        ge = gravitational_acceleration(data['altitude'])

        ax.plot(data['time'], ce, label='Centrifugal acceleration')
        ax.plot(data['time'], ge, label='Gravitional acceleration')
        ax.plot(data['time'], np.subtract(ge, ce), label='Difference')

        x = merge(x, data['time'])
        y = merge(y, ce, ge)

    #create_grid(ax, x, y, time_major_interval, 1, time_minor_interval)
    real_create_grid(ax, x, y)
    ax.set_xlim(0, max(x)+1)
    ax.set_ylim(1.05*min(y), 1.05*max(y))
    plt.legend()
    plt.title('Forces vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s^2]')

    if save:
        plt.savefig(os.path.dirname(name) + os.path.sep + 'External Forces vs Time.png', format='png', dpi=dpi)




    ##################### Altitude vs Acceleration ####################


    fig = plt.figure('Altitude vs Acceleration', figsize=figure_size, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    x = []
    y = []


    for name, data, events in lst:
        max_index = np.argmax(data['altitude'])

        ax.plot(data['acceleration'][:max_index], data['altitude'][:max_index], label='Ascent', color='black')
        ax.plot(data['acceleration'][max_index:], data['altitude'][max_index:], label='Decent', color='red')

        x = merge(x, data['acceleration'])
        y = merge(y, data['altitude'])


    #create_grid(ax, x, y, 5, 25, 1, 10)

    if anotate:
        add_horizontal_lines(data, events, 'acceleration', 'altitude', rotation=0)
    real_create_grid(ax, x, y)
    ax.set_xlim(min(x)-1, max(x)+1)
    ax.set_ylim(1.05*min(y), 1.05*max(y))
    plt.legend()
    plt.title('Altitude vs Acceleration')
    plt.xlabel('Acceleration [m/s^2]')
    plt.ylabel('Altitude [km]')

    if save:
        plt.savefig(os.path.dirname(name) + os.path.sep + 'Altitude vs Acceleration' + suffix + '.png', format='png', dpi=dpi)




    ##################### Downrange Distance vs Time ####################


    fig = plt.figure('Downrange Distance', figsize=figure_size, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    x = []
    y = []

    count = 0
    for name, data, events in lst:
        ax.plot(data['time'], data['downrange_distance'])
        #lines_with_color(ax, data['time'], data['time'], data['altitude'])
        x = merge(x, data['time'])
        y = merge(y, data['downrange_distance'])
        if data['downrange_distance'][-1] < 0:
            print(count, data['downrange_distance'][-1])
        count+=1

    #create_grid(ax, x, y, time_major_interval, 100, time_minor_interval, 50)
    if anotate:
        add_lines(events, max(y)//2)

    real_create_grid(ax, x, y)
    ax.set_xlim(0, max(x)+1)
    ax.set_ylim(min(y)-1, 1.05*max(y))
    plt.title('Downrange Distance vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Downrange Distance [km]')

    if save:
        plt.savefig(os.path.dirname(name) + os.path.sep + 'Downrange Distance' + suffix + '.png', format='png', dpi=dpi)







    ##################### Flight Trajectory ####################

    x = []
    y = []

    for name, data, events in lst:
        x = merge(x, data['downrange_distance'])
        y = merge(y, data['altitude'])


    fig = plt.figure('Flight Trajectory', figsize=figure_size)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal', adjustable='box')


    for name, data, events in lst:
        lines_with_color(ax, events_to_colordict(events), data['time'], data['downrange_distance'], data['altitude'])

    custom_lines = [Line2D([0], [0], color='b', lw=2),
                    Line2D([0], [0], color='r', lw=2),]

    plt.legend(custom_lines, ['Engines Off', 'Engines On'])

    if max(y) < 200:
        interval_y = 5
    elif 200 <= max(y) < 400:
        interval_y = 10
    else:
        interval_y = 25

    if max(x) < 200:
        interval_x = 5
    elif 200 <= max(x) < 400:
        interval_x = 10
    else:
        interval_x = 25
        interval_y = 25

    #real_create_grid(ax, x, y, num_y=10)

    x_limit = min(1.005*max(x), 700)+1
    y_limit = 1.01*max(y)


    if x_limit/y_limit < 2:
        x_limit = max(max(x), x_limit)
        real_create_grid(ax, x, y)
    else:
        create_grid(ax, x, y, interval_x, interval_y)

    ax.set_xlim(min(x)-1, x_limit)
    ax.set_ylim(0, y_limit)

    plt.title('Flight Profile')
    plt.xlabel('Downrange Distance [km]')
    plt.ylabel('Altitude [km]')

    if save:
        plt.savefig(os.path.dirname(name) + os.path.sep + 'Flight Trajectory.png', format='png', dpi=dpi, bbox_inches='tight')






    ##################### Velocity Angle vs Time ####################

    fig = plt.figure('Velocity Angle vs Time', figsize=figure_size, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    x = []
    y = []

    for name, data, events in lst:
        ax.plot(data['time'], data['angle'])

        x = merge(x, data['time'])
        y = merge(y, data['angle'])

    #add_lines(events, max(y) // 2)
    ax.set_xlim(0, max(x)+1)
    ax.set_ylim(min(y)-1, max(y)+1)
    real_create_grid(ax, x, y, major_y_interval=[1, 5, 10, 15, 30, 45, 90])
    plt.title('Velocity Angle vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Angle [degrees]')

    if save:
        plt.savefig(os.path.dirname(name) + os.path.sep + 'Velocity Angle vs Time.png', format='png', dpi=dpi)








    ##################### Altitude Vs Velocity ####################


    fig = plt.figure('Altitude Vs Velocity', figsize=figure_size, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    x = []
    y = []

    for name, data, events in lst:
        max_index = np.argmax(data['altitude'])

        ax.plot(data['velocity'][:max_index], data['altitude'][:max_index], label='Ascent', color='black')
        ax.plot(data['velocity'][max_index:], data['altitude'][max_index:], label='Decent', color='red')

        x = merge(x, data['velocity'])
        y = merge(y, data['altitude'])

    real_create_grid(ax, x, y)
    ax.set_xlim(min(x)-1, max(x)+1)
    ax.set_ylim(0, 1.05*max(y))
    #create_grid(ax, x, y, 500, 25, 100, 5)
    plt.title('Altitude Vs Velocity')
    plt.xlabel('Velocity [m/s]')
    plt.ylabel('Altitude [km]')
    plt.legend()

    if save:
        plt.savefig(os.path.dirname(name) + os.path.sep + 'Altitude Vs Velocity.png', format='png', dpi=dpi)






    ##################### Altitude Vs Aerodynamic Pressure ####################


    fig = plt.figure('Altitude Vs Aerodynamic Pressure', figsize=figure_size, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    x = []
    y = []

    for name, data, events in lst:
        max_index = np.argmax(data['altitude'])

        base = get_base(data['q'])
        unit = 10**-base

        ax.plot(unit*np.array(data['q'][:max_index]), data['altitude'][:max_index], label='Ascent', color='black')
        ax.plot(unit*np.array(data['q'][max_index:]), data['altitude'][max_index:], label='Decent', color='red')

        x = merge(x, unit*np.array(data['q']))
        y = merge(y, np.array(data['altitude']))


    #real_create_grid(ax, x, range(46), num_y=15, num_x=15)
    ax.set_xlim(0, max(x)+1)
    ax.set_ylim(0, min(max(y), 45))
    create_grid(ax, x, range(46), 2.5, 2.5)
    plt.title('Altitude Vs Aerodynamic Pressure')
    plt.xlabel('Aerodynamic Pressure [{}N/m^2]'.format(unit_dict[base]))
    plt.ylabel('Altitude [km]')
    plt.legend()

    if save:
        plt.savefig(os.path.dirname(name) + os.path.sep + 'Altitude Vs Velocity.png', format='png', dpi=dpi)




    ##################### Specific Mechanical Energy ####################


    fig = plt.figure('Specific Mechanical Energy', figsize=figure_size, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    x = []
    y = []

    for name, data, events in lst:
        pe = np.array(list(map(potential_energy, data['altitude'])))
        ke = np.array(list(map(velocity_to_kinetic_energy, data['velocity'])))

        base = get_base(np.add(pe, ke))
        unit = 10**-base

        ax.plot(data['time'], unit*pe, label='Potential Energy')
        ax.plot(data['time'], unit*ke, label='Kinetic Energy')
        ax.plot(data['time'], unit*np.add(ke, pe), label='Total Energy')

        x = merge(x, data['time'])
        y = merge(y, unit*pe, unit*ke, unit*np.add(ke, pe))


    #create_grid(ax, x, y, time_major_interval, 0.5, time_minor_interval, 0.1)
    #add_lines(events, max(y)//2)
    real_create_grid(ax, x, y)
    ax.set_xlim(0, max(x)+1)
    ax.set_ylim(0, 1.05*max(y))
    plt.title('Specific Mechanical Energy')
    plt.xlabel('Time [s]')
    plt.ylabel('Specific Energy [{}J/kg]'.format(unit_dict[base]))
    plt.legend(loc='upper left')

    if save:
        plt.savefig(os.path.dirname(name) + os.path.sep + 'Specific Mechanical Energy.png', format='png', dpi=dpi)






    ##################### Kinetic Energy ####################


    fig = plt.figure('Specific Kinetic Energy', figsize=figure_size, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    x = []
    y = []

    for name, data, events in lst:
        tke = np.array(list(map(velocity_to_kinetic_energy, data['velocity'])))
        xke = np.array(list(map(velocity_to_kinetic_energy, data['velocity_x'])))
        yke = np.array(list(map(velocity_to_kinetic_energy, data['velocity_y'])))

        base = get_base(tke, xke, yke)
        unit = 10**-base

        ax.plot(data['time'], unit*tke, label='Total Kinetic Energy')
        ax.plot(data['time'], unit*xke, label='Horizontal Kinetic Energy')
        ax.plot(data['time'], unit*yke, label='Vertical Kinetic Energy')

        x = merge(x, data['time'])
        y = merge(y, unit*tke, unit*xke, unit*yke)



    #create_grid(ax, x, y, time_major_interval, 0.5, time_minor_interval, 0.1)
    #add_lines(events, max(y)//2)

    real_create_grid(ax, x, y)
    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    plt.title('Specific Kinetic Energy')
    plt.xlabel('Time [s]')
    plt.ylabel('Specific Energy [{}J/kg]'.format(unit_dict[base]))
    plt.legend(loc='best')

    if save:
        plt.savefig(os.path.dirname(name) + os.path.sep + 'Specific Kinetic Energy.png', format='png', dpi=dpi)







    ##################### Aerodynamic Pressure ####################


    fig = plt.figure('Aerodynamic Pressure', figsize=figure_size, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    x = []
    y = []


    for name, data, events in lst:
        ax.plot(data['time'], 10**-3*np.array(data['q']))
        x = merge(x, data['time'])
        y = merge(y, 10**-3*np.array(data['q']))


    #create_grid(ax, x, y, time_major_interval, 5000, time_minor_interval, 1000)
    #add_lines(events, max(y)//2)

    real_create_grid(ax, x, y)
    ax.set_xlim(0, max(x)+1)
    ax.set_ylim(0, 1.05*max(y))
    plt.title('Aerodynamic pressure')
    plt.xlabel('Time [s]')
    plt.ylabel('Dynamic Pressure [kN/m^2]')

    if save:
        plt.savefig(os.path.dirname(name) + os.path.sep + 'Aerodynamic Pressure.png', format='png', dpi=dpi)


    if not save:
        plt.show()


lst = []



engine_event_dict = OrderedDict([
    ('meco', '#ff0000'),
    ('ses1', 'b'),
    ('seco1', '#ff0000'),
    ('boostback_start', 'b'),
    ('boostback_end', '#ee0000'),
    ('entry_start', 'b'),
    ('entry_end', '#ee0000'),
    ('landing_start', 'b'),
    ('landing_end', '#cc0000'),
])



for i in range(1, len(argv), 2):
    data_file = open(argv[i], 'r')
    event_file = open(argv[i+1], 'r')
    events = json.load(event_file)

    lst.append((argv[i], read_list(data_file), events))

graph(lst, save=False, anotate=True)


#create_images(os.path.dirname(file_name), lst, json.load(event_file))
#fig = plt.figure()
#ax1 = fig.add_subplot(1,1,1)
#ani = animation.FuncAnimation(fig, animate, interval=200)
#plt.show()
