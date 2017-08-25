import matplotlib.pyplot as plt
import json
from sys import argv
from os.path import splitext, basename
import numpy as np
import trendline
from math import fabs, sin, radians
from scipy.signal import savgol_filter


from mpl_toolkits.axes_grid1 import ImageGrid




im = np.arange(100)
im.shape = 10, 10



def refine_data(time ,data):
    tl = []
    pl = []
    for t, p in zip(time, data):
        if len(tl) == 0 or (pl[-1] != p and tl[-1] != t):
            tl.append(t)
            pl.append(p)
    
    return tl, pl


def convert_unit(lst, unit):
    return [x*unit for x in lst]


def approx(x1, y1):
    # calculate polynomial
    z = np.polyfit(x1, y1, 1)
    print(z)
    f = np.poly1d(z)

    # calculate new x's and y's
    x_new = np.linspace(x1[0], x1[-1], len(x1))
    y_new = f(x_new)

    with open('a.txt', 'a') as f:
        for i in range(len(x_new)):
            f.write(json.dumps(
                {'time': x_new[i],
                 'acceleration': y_new[i]
                })+'\n')
    return x_new, y_new


def abs(lst):
    return [fabs(x) for x in lst]
    
def approx_time(x, y, t1, t2):
    return approx(x[30*t1:30*t2], y[30*t1:30*t2])


def add_grid(x_data, y_data, x_gap, y_gap):
    max_x = max(x_data)
    max_y = max(y_data)

    min_x = min(x_data)
    min_y = min(y_data)

    plt.xticks(np.arange(0, max_x + x_gap - max_x % x_gap + 1, x_gap))
    plt.yticks(np.arange(0, max_y + y_gap - max_y % y_gap + 1, y_gap))




def plot_graph(xdata, ydata, name, xlabel, ylabel, title, color='black', symbol='-', ):
    if name is None:
        plt.plot(xdata, ydata, symbol, color=color)
    else:
        plt.plot(xdata, ydata, symbol, label=name, color=color)

    plt.grid()
    plt.legend()
    plt.title(title, fontsize=18, y=1.04)


events = {
    1: 'Launch',
    148.5: 'Main Engine Cut Off',
    158: 'Second Engine Startup',
    44.5: 'Throttle down',
    60.5: 'Throttle up',
    552: 'Second Engine Cut Off',
    492: 'Second Engine throttle down'
}


def add_labels(y_data):
    for key in events:
        plt.axvline(x=key, color='black', linewidth=2, alpha=0.5, linestyle='--')
        plt.text(key, 0.65*max(y_data), events[key], rotation=-90)


def graph(lst):
    plt.figure('Velocity(time)')
    for name, data in lst:
        plot_graph(data['time'], data['velocity'], 'Velocity [m/s]', 'Time [s]', 'Velocity [m/s]', 'Velocity(time)',color='red')
        plot_graph(data['time'], data['velocity_x'], 'Tangential Velocity [m/s]', 'Time [s]', 'Velocity [m/s]', 'Velocity(time)', color='blue')
        plot_graph(data['time'], data['velocity_y'], 'Radial Velocity [m/s]', 'Time [s]', 'Velocity [m/s]', 'Velocity(time)', color='green')

        add_grid(data['time'], data['velocity'], 50, 500)

        add_labels(data['velocity'])

    """
    plt.figure('|Velocity|(time)')
    for name, data in lst:
        plot_graph(data['time'], data['velocity'], 'Velocity [m/s]', 'Time [s]', 'Velocity [m/s]', 'Velocity(time)')
        plot_graph(data['time'], abs(data['velocity_x']), '|Tangential Velocity| [m/s]', 'Time [s]', 'Velocity [m/s]', 'Velocity(time)', color='blue')
        plot_graph(data['time'], abs(data['velocity_y']), '|Radial Velocity| [m/s]', 'Time [s]', 'Velocity [m/s]', 'Velocity(time)', color='green')
    """



    plt.figure('Altitude(time)')
    for name, data in lst:
        plot_graph(*refine_data(data['time'], data['altitude']), None, 'Time [s]', 'Altitude [km]', 'Altitude(time)')
        add_grid(data['time'], data['altitude'], 50, 50)

        add_labels(data['altitude'])




    plt.figure('Downrange Distance(time)')
    for name, data in lst:
        plot_graph(*trendline.approx(*refine_data(data['time'], data['downrange_distance'])), None, 'Time [s]', 'Downrange distance [km]', 'Downrange Distance(time)')
        add_grid(data['time'], data['downrange_distance'], 50, 100)

        add_labels(data['downrange_distance'])



    plt.figure('Acceleration(time)')
    for name, data in lst:
        plot_graph(*refine_data(data['time'], data['acceleration']) , None, 'Time [s]', 'Acceleration [m*s^-2]', 'Acceleration(time)')
        add_grid(data['time'], data['acceleration'], 50, 5)

        add_labels(data['acceleration'])

    #plt.figure('Acceleration(Altitude)')
    #for name, data in lst:
    #    plot_graph(*refine_data(data['acceleration'], data['altitude']), name, 'Acceleration [m*s^-2]', 'Altitude [km]', 'Acceleration(Altitude)')
    
    plt.figure('Flight profile')
    for name, data in lst:
        plot_graph(*refine_data(data['downrange_distance'], data['altitude']), None, 'Downrange distance [km]', 'Altitude [km]', 'Flight profile', color='blue')

        add_grid(data['downrange_distance'], data['altitude'], 100, 100)
        plt.axis('equal')

        #for key in events:
        #    plt.axvline(x=key, color='black', linewidth=2, alpha=0.5, linestyle='--')
        #    plt.text(key, max(y_data) / 2, data['downrange_distance'], rotation=-90)

    #plt.figure('Velocity(Altitude)')
    #for name, data in lst:
    #    plot_graph(*refine_data(data['velocity'], data['altitude']), name, 'Velocity [m/s]', 'Altitude [km]', 'Velocity(Altitude)')
    #    plot_graph(data['vertical_velocity'], data['altitude'], name, 'Velocity [m/s]', 'Altitude [km]', 'Velocity(Altitude)')
    #    plot_graph(data['horizontal_velocity'], data['altitude'], name, 'Velocity [m/s]', 'Altitude [km]', 'Velocity(Altitude)')


    plt.figure('Acceleration effect external forces')
    for name, data in lst:
        plt.yticks(np.arange(min(data['gravity']+data['drag']), max(data['gravity']+data['drag']) + 1, 2.0))
        plt.xticks(np.arange(min(data['time']), max(data['time']) + 1, 10))

        plt.fill_between(data['time'], data['gravity'], 0, color='#9ceebf', label='Gravity')
        plt.fill_between(data['time'], data['drag'], 0, color='#4289f1', label='Drag')
        plt.legend()

        plt.xlabel('Time [s]', fontsize=14)
        plt.ylabel('Acceleration [m/s^2]', fontsize=14)
        plt.title('Acceleration effect external forces', fontsize=18, y=1.04)

        add_grid(data['time'], data['gravity'], 25, 1)
        plt.grid()

    plt.figure('Velocity angle(time)')
    for name, data in lst:
        plot_graph(data['time'], data['angle'], None, 'Time [s]', 'Angle [degrees]',
                'Velocity angle(time)')

        add_grid(data['time'], data['angle'], 25, 5)



    plt.figure('Delta V(time)')
    for name, data in lst:
        plot_graph(*refine_data(data['time'], data['delta_v']), 'Delta V', 'Time [s]', 'Delta V [m/s]',
                'Delta V(time)')
        plot_graph(*refine_data(data['time'], data['delta_v_drag']), 'Delta V - Drag', 'Time [s]', 'Delta V [m/s]',
               'Delta V(time)', color='green')
        plot_graph(*refine_data(data['time'], data['gravity_loss']), 'Gravity losses', 'Time [s]', 'Delta V [m/s]',
               'Delta V(time)', color='blue')



    #plt.figure('Drag(Altitude)')
    #for name, data in lst:
    #    plot_graph(data['drag'], data['altitude'], name, 'Drag [m*s^-2]', 'Altitude [km]',
    #            'Drag(Altitude)')


    plt.figure('Throttle(time)')
    for name, data in lst:
        plot_graph(*refine_data(data['time'], data['throttle']), None, 'Time [s]', 'Throttle [%]',
                        'Throttle(time)')

        add_labels(data['throttle'])
        #plot_graph(*trendline.avg_data(*refine_data(data['time'], data['force']), deg = 5), name, 'Time [s]', 'Throttle [%]',
        #        'Throttle avg(time)')


    plt.figure('Propellant Mass(time)')
    for name, data in lst:
        plot_graph(data['time'], data['s1_mass'], 'Stage 1 propellant mass', 'Time [s]', 'Propellant Mass [kg]',
                   'Propellant Mass(time)')
        plot_graph(data['time'], data['s2_mass'], 'Stage 2 propellant mass', 'Time [s]', 'Propellant Mass [kg]',
                   'Propellant Mass(time)')

        add_labels(data['s1_mass'])



    plt.figure('Velocity angle(Altitude)')
    for name, data in lst:
        plot_graph(data['angle'], data['altitude'], None, 'Velocity angle [degrees]', 'Altitude [km]',
                'Velocity angle(Altitude)')

        add_grid(data['angle'], data['altitude'], 5, 50)


    plt.figure('Cd (time)')
    for name, data in lst:
        plot_graph(data['time'], data['cd'], None, 'Time [s]','Cd',
                'Cd (Velocity)')


    plt.figure('Thrust (time)')
    for name, data in lst:
        plot_graph(data['time'], convert_unit(data['thrust'], 10**-3), None, 'Time [s]','Thrust [KN]',
                'Thrust (time)')

        add_labels(convert_unit(data['thrust'], 10**-3))




    plt.figure('Energy (time)')
    for name, data in lst:
        #plot_graph(data['time'], convert_unit(data['ke'], 10**-6), 'Kinetic Energy', 'Time [s]','Kinetic Energy',
        #        'Energy vs Time', color='blue')
        #plot_graph(data['time'], convert_unit(data['pe'], 10**-6), 'Potential Energy', 'Time [s]', 'Potential Energy',
        #        'Energy vs Time', color='green')
        #plot_graph(data['time'], convert_unit(data['energy'], 10**-6), 'Mechanical Energy', 'Time [s]', 'Energy [MJ]',
        #       'Energy vs Time', color='red')

        plt.fill_between(data['time'], convert_unit(data['energy'], 10 ** -6), 0, label='Mechanical Energy', color='blue')
        plt.fill_between(data['time'], convert_unit(data['ke'], 10**-6), 0, label='Potential Energy', color='green')
        plt.fill_between(data['time'], convert_unit(data['pe'], 10 ** -6), 0, label='Kinetic Energy', color='red')

        add_grid(data['time'], convert_unit(data['energy'], 10**-6), 25, 5)
        plt.grid()
        plt.xlabel('Time [sec]', size=16)
        plt.ylabel('Energy [MJ]', size=16)
        plt.legend()



    plt.figure('Pie (time)')
    for name, data in lst:
        sum_drag = sum(data['drag'])
        sum_gravity = sum( [g*sin(radians(a)) for g, a in zip(data['gravity'], data['angle'])] )
        _, _, autotexts = plt.pie((sum_gravity, sum_drag), labels = ('Gravity', 'Drag'),autopct='%1.1f%%',pctdistance=0.95,startangle=90,colors=('#9ceebf','#4289f1'))
        plt.axis('equal')
        for autotext in autotexts:
            autotext.set_color('red')

    plt.show()
    
    
    


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return (a / np.expand_dims(l2, axis))[0]



def graph_awesome(lst):
    plt.figure('Velocity(time)')
    for name, data in lst:
        plt.grid()
        plt.plot(data['time'], data['velocity'], label=name)
        plt.legend()
        plt.xticks(np.arange(0, 500, 20))

        plt.grid()

    plt.figure('acc')
    for name, data in lst:
        plt.grid()
        plt.plot(data['time'], data['acceleration'], label=name)
        plt.legend()
        plt.xticks(np.arange(0, 500, 20))

        plt.grid()

    plt.show()


def read_list(file):
    data = json.loads(file.readline())
    
    for x in data:
        data[x] = [data[x]]

    for line in file:
        dict = json.loads(line)

        for x in dict:
            data[x].append(dict[x])

    return data

lst = []


for file_name in argv[1:]:
    file = open(file_name, 'r')
    lst.append((splitext(basename(file_name))[0], read_list(file)))

graph(lst)
#plot(lst)
#graph_awesome(lst)
