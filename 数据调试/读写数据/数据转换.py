import pandas as pd
import numpy as np

#df = pd.read_csv("da_hrl_lmps.csv")['total_lmp_da'][0:24]
#print(df)
#df.to_csv('solar_1.csv')
#print(np.array(df))

#np.savetxt('lmp_1.csv', np.array(df), fmt='%.3f', delimiter=',')


for i in range(1, 29):
    read_path = './Data_/LMP/da_hrl_lmps_' + str(i) + '.csv'
    write_path = './data/LMP/lmp_' + str(i) + '.csv'
    #print(read_path)
    df = pd.read_csv(read_path)['total_lmp_da'][0:24]
    np.savetxt(write_path, np.array(df), fmt='%.3f', delimiter=',')