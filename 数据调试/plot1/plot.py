import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

sns.set()

load = np.loadtxt('./data/load.csv',dtype = np.float64, delimiter = ',', unpack = True)/1000
solar = np.loadtxt('./data/solar.csv',dtype = np.float64, delimiter = ',', unpack = True)/1000
wind = np.loadtxt('./data/wind.csv',dtype = np.float64, delimiter = ',', unpack = True)/1000
lmp = np.loadtxt('./data/lmp_1.csv',dtype = np.float64, delimiter = ',', unpack = True)/10

print(load)
print(solar)
print(wind)
print(lmp)
i = 1
print('./data/lmp_'+str(i)+'.csv')

P_net = load-solar-wind
LMP = lmp

x = np.linspace(0, 23, 24)
y1 = P_net
y2 = LMP

fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(x, y1, label='Pnet')

ax2 = ax.twinx()
ax2.plot(x, y2, '-y*', label='LMP')
fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)

ax.set_xlabel("Time", size = 23)
ax.set_ylabel(r"Pnet/kw", size = 23)
ax2.set_ylabel(r"LMP", size = 23)
ax2.grid(False)

plt.show()