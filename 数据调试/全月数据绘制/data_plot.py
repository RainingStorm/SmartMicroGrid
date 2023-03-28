import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

sns.set()

font = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }

for i in range(1, 29):
    data_path = './data/LMP/lmp_' + str(i) + '.csv'
    lmp = np.loadtxt(data_path, dtype=np.float64, delimiter=',', unpack=True)
    plt.xlabel("Time", font)
    plt.ylabel("LMP", font)
    plt.plot(lmp)
plt.show()

for i in range(1, 29):
    data_path = './data/Load/load_' + str(i) + '.csv'
    load = np.loadtxt(data_path, dtype=np.float64, delimiter=',', unpack=True)
    plt.xlabel("Time", font)
    plt.ylabel("Load", font)
    plt.plot(load)
plt.show()

for i in range(1, 29):
    data_path = './data/Solar/solar_' + str(i) + '.csv'
    solar = np.loadtxt(data_path, dtype=np.float64, delimiter=',', unpack=True)
    plt.xlabel("Time", font)
    plt.ylabel("Solar", font)
    plt.plot(solar)
plt.show()

for i in range(1, 29):
    data_path = './data/Wind/wind_' + str(i) + '.csv'
    wind = np.loadtxt(data_path, dtype=np.float64, delimiter=',', unpack=True)
    plt.xlabel("Time", font)
    plt.ylabel("Wind", font)
    plt.plot(wind)
plt.show()