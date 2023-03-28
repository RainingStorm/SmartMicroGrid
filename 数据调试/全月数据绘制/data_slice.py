import numpy as np

new_load = []
new_lmp = []
new_solar = []
new_wind = []

for i in range(22, 29):
    data_path = './data/Load/load_' + str(i) + '.csv'
    load = list(np.loadtxt(data_path, dtype=np.float64, delimiter=',', unpack=True))
    new_load += load
np.savetxt('./new_data/load_4.csv', np.array(new_load), fmt='%.3f', delimiter=',')

for i in range(22, 29):
    data_path = './data/LMP/lmp_' + str(i) + '.csv'
    lmp = list(np.loadtxt(data_path, dtype=np.float64, delimiter=',', unpack=True))
    new_lmp += lmp
np.savetxt('./new_data/lmp_4.csv', np.array(new_lmp), fmt='%.3f', delimiter=',')

for i in range(22, 29):
    data_path = './data/Solar/solar_' + str(i) + '.csv'
    solar = list(np.loadtxt(data_path, dtype=np.float64, delimiter=',', unpack=True))
    new_solar += solar
np.savetxt('./new_data/solar_4.csv', np.array(new_solar), fmt='%.3f', delimiter=',')

for i in range(22, 29):
    data_path = './data/Wind/wind_' + str(i) + '.csv'
    wind = list(np.loadtxt(data_path, dtype=np.float64, delimiter=',', unpack=True))
    new_wind += wind
np.savetxt('./new_data/wind_4.csv', np.array(new_wind), fmt='%.3f', delimiter=',')