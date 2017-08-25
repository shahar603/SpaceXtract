import numpy as np

from collections import OrderedDict
import json
import matplotlib.pyplot as plt

def approx(x_data, y_data, deg = 50):
    # calculate polynomial
    z = np.polyfit(x_data, y_data, deg)
    f = np.poly1d(z)

    return x_data, f(x_data)

def refine_data(time, data):
    tl = []
    pl = []
    for t, p in zip(time, data):
        if len(tl) == 0 or (pl[-1] != p and tl[-1] != t):
            tl.append(t)
            pl.append(p)

    return tl, pl

def approx_func(x_data, y_data, deg = 50):
    # calculate polynomial
    z = np.polyfit(x_data, y_data, deg)
    f = np.poly1d(z)

    return f

def get_data(file, x_name, y_name):
    x = []
    y = []

    for line in file:
        data = json.JSONDecoder(object_pairs_hook=OrderedDict).decode(line)
        x.append(data[x_name])
        y.append(data[y_name])

    return approx(x,y)


def get_func(file, x_name, y_name):
    x = []
    y = []

    for line in file:
        data = json.JSONDecoder(object_pairs_hook=OrderedDict).decode(line)
        x.append(data[x_name])
        y.append(data[y_name])

    z = np.polyfit(x, y, 50)
    return np.poly1d(z)



def avg_data(x, y, deg = 3):
    # calculate polynomial

    for j in range(deg):
        y = [(y[i]+y[i+1])/2 for i in range(len(y)-1)]
    return [i+deg/2 for i in x[:-deg]], y


