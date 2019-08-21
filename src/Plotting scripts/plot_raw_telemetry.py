import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sys import argv
import os
from os.path import splitext, basename
import json
from math import e


def Test2(rootDir):
    paths = []
    for lists in os.listdir(rootDir):
        path = os.path.join(rootDir, lists)
        if os.path.isdir(path):
            paths += Test2(path)
        else:
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



def animate(i):
    graph_data = open(argv[1],'r')
    data = read_list(graph_data)
    ax1.clear()
    
    ax1.plot(data['time'], data['velocity'])
    ax1.plot(data['time'], data['altitude'])


def graph(lst):

    plt.figure('Velocity(time)')
    for name, data in lst:
        plt.plot(data['time'], data['velocity'], label=name)

    plt.legend()
    plt.grid()

    plt.figure('Altitude(time)')
    for name, data in lst:
        plt.plot(data['time'], data['altitude'], label=name)

    plt.legend()
    plt.grid()


    plt.figure('Altitude(Velocity)')
    for name, data in lst:
        plt.plot(data['velocity'], data['altitude'], label=name)

    plt.legend()
    plt.grid()

    plt.show()
    
    
def animate_graph():
    fig = plt.figure()
    global ax1
    ax1 = fig.add_subplot(1,1,1)
    ani = animation.FuncAnimation(fig, animate, interval=100)
    plt.show()


lst = []

if len(argv) == 1:
    arguments = Test2('.')
else:
    arguments = argv[1:]


for file_name in arguments:
    file = open(file_name, 'r')
    lst.append((splitext(basename(file_name))[0], read_list(file)))


graph(lst)
#animate_graph()