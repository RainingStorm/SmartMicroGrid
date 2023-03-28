import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

sns.set()
#rewards1 = np.loadtxt('rewards1.csv',dtype = np.float, delimiter = ',', unpack = True)
#rewards2 = np.loadtxt('rewards2.csv',dtype = np.float, delimiter = ',', unpack = True)
ma_rewards1 = np.loadtxt('ma_rewards1.csv',dtype = np.float, delimiter = ',', unpack = True)
ma_rewards2 = np.loadtxt('ma_rewards2.csv',dtype = np.float, delimiter = ',', unpack = True)
ma_rewards3 = np.loadtxt('ma_rewards3.csv',dtype = np.float, delimiter = ',', unpack = True)
#rewards=np.vstack((rewards1,rewards2)) # 合并数组
ma_rewards=np.vstack((ma_rewards1,ma_rewards2))
ma_rewards=np.vstack((ma_rewards,ma_rewards3))

#df1 = pd.DataFrame(rewards).melt(var_name='episode',value_name='reward')
df2 = pd.DataFrame(ma_rewards).melt(var_name='episode',value_name='ma_reward')

#sns.lineplot(x="episode", y="reward", data=df1)
#plt.show()

sns.lineplot(x="episode", y="ma_reward", data=df2)
plt.show()