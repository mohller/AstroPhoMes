"""Functions for the mass scaling exponent based on data
"""

import numpy as np
from scipy.interpolate import interp1d
from os.path import join
import sys
sys.path.append('../')
from config import *

nsc = np.load(join(global_path, 'data/scaling_lines.npy'))

e = np.linspace(.1, 800, 300)

def sigm(x, shift=0., gap=1, speed=1, base=0.):
    return gap*( 1. - 1. /(1 + np.exp(- speed * (x - shift)))) + base

def low(x):
    y = []
    
    for xi in x:
        if xi < 7:
            y.append(sigm(xi, 1., .114, .75, .927))
        elif xi < 100:
            y.append(np.interp(xi, nsc[2], nsc[3]))
        else:
            y.append(sigm(xi, 160, .231, .05, .66))
            
    return np.array(y)

def med(x):
    emid = np.linspace(600, 700, 21)
    emid = np.array(range(600, 630, 10) + range(680, 720, 10))

    ymid = np.where(emid < 650, np.interp(emid, nsc[0], nsc[1]), .66)

    s = interp1d(emid, ymid, 'slinear')

    y = []
    for xi in x:
        if xi < 7:
            y.append(sigm(xi, .7, .099, .7, .942))
        elif xi < 300:
            y.append(np.interp(xi, nsc[0], nsc[1]))
        else:
            y.append(sigm(xi, 430, .2, .01, .66))

    return np.array(y)

def hie(x):
    y = []
    
    for xi in x:
        if xi < 7:
            y.append(sigm(xi, .7, .074, .6, .956))
        elif xi < 140:
            y.append(np.interp(xi, nsc[4], nsc[5]))
        else:
            y.append(sigm(xi, 430, .4555, .001, .66))

    return np.array(y)

def hieer(x):
    y = []
    
    for xi in x:
        if xi < 7:
            y.append(sigm(xi, .7, .074, .6, .956))
        elif xi < 140:
            y.append(np.interp(xi, nsc[4], nsc[5]))
        else:
            y.append(sigm(xi, -80, .1, .01, .91))

    return np.array(y)


A_eff_low = lambda A, x: float(A)**np.array(low(x))
A_eff_med = lambda A, x: float(A)**np.array(med(x))
A_eff_hig = lambda A, x: float(A)**np.array(hie(x))
A_eff_higher = lambda A, x: float(A)**np.array(hieer(x))


def main():
    import matplotlib.pyplot as plt

    ee = np.logspace(-1, 4, 1000)
    plt.plot(ee, hieer(ee), label='hieer')
    plt.plot(ee, hie(ee), label='hie')
    plt.plot(ee, low(ee), label='low')
    plt.plot(ee, med(ee), label='med')
    plt.legend()

    plt.semilogx()

    plt.show()


if __name__ == '__main__':
    main()