import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

sns.set()

font = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
LMP = []
for i in range(1, 29):
    data_path = './data/LMP/lmp_' + str(i) + '.csv'
    lmp = list(np.loadtxt(data_path, dtype=np.float64, delimiter=',', unpack=True))
    LMP += lmp
plt.xlabel("Time", font)
plt.ylabel("LMP", font)
plt.plot(LMP, label="LMP")
plt.legend()
plt.show()

LOAD = []
for i in range(1, 29):
    data_path = './data/Load/load_' + str(i) + '.csv'
    load = list(np.loadtxt(data_path, dtype=np.float64, delimiter=',', unpack=True))
    LOAD += load
plt.xlabel("Time", font)
plt.ylabel("Power", font)
plt.plot(LOAD, label="load")
plt.legend()
plt.show()


SOLAR = []
for i in range(1, 29):
    data_path = './data/Solar/solar_' + str(i) + '.csv'
    solar = list(np.loadtxt(data_path, dtype=np.float64, delimiter=',', unpack=True))
    SOLAR += solar
plt.xlabel("Time", font)
plt.ylabel("Power", font)
plt.plot(SOLAR, label="solar")
plt.legend()
plt.show()

WIND = []
for i in range(1, 29):
    data_path = './data/Wind/wind_' + str(i) + '.csv'
    wind = list(np.loadtxt(data_path, dtype=np.float64, delimiter=',', unpack=True))
    WIND += wind
plt.xlabel("Time", font)
plt.ylabel("Power", font)
plt.plot(WIND, label="wind")
plt.legend()
plt.show()