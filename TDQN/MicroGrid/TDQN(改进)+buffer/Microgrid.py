import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random


class MicroGrid(gym.Env):
	"""docstring for MicroGrid"""
	def __init__(self):
		super(MicroGrid, self).__init__()
		self.viewer = None
		self.max_Soc = 200                 # 储能装置最大容量
		self.min_Soc = 0.1 * self.max_Soc  # 储能装置最小容量
		self.ini_Soc = 0.1 * self.max_Soc  # 储能装置初始容量
		self.eta_ESS = 0.9
		self.start_time = 0                # 开始时间
		self.end_time = 23                 # 结束时间
		self.reg_anums = 11                # 生产设备的动作数量
		self.reg1_max = 10                 # 生产设备1最大工作功率
		self.reg2_max = 20                 # 生产设备2最大工作功率


		self.reset()
		self.seed()


	def seed(self, seed=None):
		# 产生一个随机化时需要的种子，同时返回一个np_random对象，支持后续的随机化生成操作
		self.np_random, seed = seeding.np_random(seed)  
		return [seed]

	def _clip(self, x, min, max):
		if x < min:
			return min
		elif x > max:
			return max
		return x

	def step(self, action):
		L23, L22, L21, L20, L19, L18, L17, L16, L15, L14, L13, L12, L11, L10, L9, L8, L7, L6, L5, L4, L3, L2, L1, L0, pnet, soc, t = self.state

		# 计算下一时刻状态
		if t == self.end_time:
			done = True
		else:
			done = False
		
		if not done:
			t_ = t+1
			pnet_ = self.P_net[int(t_)]
			
		else:
			t_ = 0
			pnet_ = self.P_net[0]
			

		L_0 = self.LMP[int(t_)]
		L_1 = self.LMP[int(t_-1)]
		L_2 = self.LMP[int(t_-2)]
		L_3 = self.LMP[int(t_-3)]
		L_4 = self.LMP[int(t_-4)]
		L_5 = self.LMP[int(t_-5)]
		L_6 = self.LMP[int(t_-6)]
		L_7 = self.LMP[int(t_-7)]
		L_8 = self.LMP[int(t_-8)]
		L_9 = self.LMP[int(t_-9)]
		L_10 = self.LMP[int(t_-10)]
		L_11 = self.LMP[int(t_-11)]
		L_12 = self.LMP[int(t_-12)]
		L_13 = self.LMP[int(t_-13)]
		L_14 = self.LMP[int(t_-14)]
		L_15 = self.LMP[int(t_-15)]
		L_16 = self.LMP[int(t_-16)]
		L_17 = self.LMP[int(t_-17)]
		L_18 = self.LMP[int(t_-18)]
		L_19 = self.LMP[int(t_-19)]
		L_20 = self.LMP[int(t_-20)]
		L_21 = self.LMP[int(t_-21)]
		L_22 = self.LMP[int(t_-22)]
		L_23 = self.LMP[int(t_-23)]

		# 根据决策的储能装置动作计算其等效功率
		pb_t = self._clip(action[1][2], -50, 50)
		if pb_t >= 0:
			Pb = min(pb_t*self.eta_ESS, self.max_Soc-soc)
		else:
			Pb = max(pb_t/self.eta_ESS, self.min_Soc-soc)
		soc_ = soc + Pb
        
        #发电机动作裁剪
		Pdg1 = self._clip(action[1][0], 0, 30)
		Pdg2 = self._clip(action[1][1], 0, 20)

		# Preg = (self.reg_max/(self.reg_anums-1)) * action[0]
		Preg1 = int(action[0] / self.reg_anums) * (self.reg1_max/(self.reg_anums-1))
		Preg2 = (action[0] % self.reg_anums) * (self.reg2_max / (self.reg_anums - 1))
		Pg = pnet + Pb + Preg1 + Preg2 - Pdg1 - Pdg2

		#主电网电费计算
		if Pg >= 0:
			rg = -1*Pg*L0
		else:
			rg = -1*0.5*Pg*L0

		# 发电机电费计算
		rdg1 = -1*(0.01*Pdg1**2 + 6.04*Pdg1 + 13.187)
		rdg2 = -1*(0.01*Pdg2**2 + 7.15*Pdg2 + 6.615)

		# 生产设备收益
		rreg1 = 45.82 * (Preg1 ** 0.5)
		rreg2 = 57.51 * (Preg2 ** 0.5)
		#rreg = self.reg_p*Preg

        # 奖励计算
		self.reward = (rg + rdg1 + rdg2 + rreg1 + rreg2) * 0.001
		self.state = (L_23, L_22, L_21, L_20, L_19, L_18, L_17, L_16, L_15, L_14, L_13, L_12, L_11, L_10, L_9, L_8, L_7, L_6, L_5, L_4, L_3, L_2, L_1, L_0, pnet_, soc_, t_)

		return np.array(self.state), self.reward, done, {}





	def reset(self):
		# 外部数据加载
		self.load = np.loadtxt('./data/load.csv', dtype=np.float64, delimiter=',', unpack=True) / 1000
		self.solar = np.loadtxt('./data/solar.csv', dtype=np.float64, delimiter=',', unpack=True) / 1000
		self.wind = np.loadtxt('./data/wind.csv', dtype=np.float64, delimiter=',', unpack=True) / 1000
		self.lmp = np.loadtxt('./data/lmp.csv', dtype=np.float64, delimiter=',', unpack=True) / 10

		self.P_net = self.load - self.solar - self.wind  # 内部功率变化
		self.LMP = self.lmp  # 电价变化


		# 加入随机噪声

		for i in range(24):
			self.P_net[i] += random.uniform(-15, 15)
			self.LMP[i] += random.uniform(-1.5, 1.5)

		self.reg_p = sorted(self.LMP)[int(len(self.LMP) / 2)]

		# 状态空间设置
		self.low_state = np.array([min(self.LMP),
								   min(self.LMP),
								   min(self.LMP),
								   min(self.LMP),
								   min(self.LMP),
								   min(self.LMP),
								   min(self.LMP),
								   min(self.LMP),
								   min(self.LMP),
								   min(self.LMP),
								   min(self.LMP),
								   min(self.LMP),
								   min(self.LMP),
								   min(self.LMP),
								   min(self.LMP),
								   min(self.LMP),
								   min(self.LMP),
								   min(self.LMP),
								   min(self.LMP),
								   min(self.LMP),
								   min(self.LMP),
								   min(self.LMP),
								   min(self.LMP),
								   min(self.LMP),
								   min(self.P_net),
								   self.min_Soc,
								   self.start_time
								   ])
		self.high_state = np.array([max(self.LMP),
									max(self.LMP),
									max(self.LMP),
									max(self.LMP),
									max(self.LMP),
									max(self.LMP),
									max(self.LMP),
									max(self.LMP),
									max(self.LMP),
									max(self.LMP),
									max(self.LMP),
									max(self.LMP),
									max(self.LMP),
									max(self.LMP),
									max(self.LMP),
									max(self.LMP),
									max(self.LMP),
									max(self.LMP),
									max(self.LMP),
									max(self.LMP),
									max(self.LMP),
									max(self.LMP),
									max(self.LMP),
									max(self.LMP),
									max(self.P_net),
									self.max_Soc,
									self.end_time
									])
		self.observation_space = spaces.Box(self.low_state, self.high_state, dtype=np.float32)

		# 动作空间设置[发电机1，发电机2，储能装置]
		parameters_min = np.array([0, 0, -50])
		parameters_max = np.array([30, 20, 50])
		self.action_space = spaces.Tuple((spaces.Discrete(self.reg_anums**2),
										  spaces.Box(parameters_min, parameters_max)))
		# 状态重置
		self.state = np.array([self.LMP[-23],
							   self.LMP[-22],
							   self.LMP[-21],
							   self.LMP[-20],
							   self.LMP[-19],
							   self.LMP[-18],
							   self.LMP[-17],
							   self.LMP[-16],
							   self.LMP[-15],
							   self.LMP[-14],
							   self.LMP[-13],
							   self.LMP[-12],
							   self.LMP[-11],
							   self.LMP[-10],
							   self.LMP[-9],
							   self.LMP[-8],
							   self.LMP[-7],
							   self.LMP[-6],
							   self.LMP[-5],
							   self.LMP[-4],
							   self.LMP[-3],
							   self.LMP[-2],
							   self.LMP[-1],
							   self.LMP[0],
							   self.P_net[0],
							   self.ini_Soc,
							   0
							   ])
		return self.state, self.LMP

	def close(self):
		if self.viewer: self.viewer.close()

if __name__ =="__main__":
	env = MicroGrid()
	print(env.reg_p)
	s = env.reset()
	state_dim = env.observation_space.shape[0]
	c_dim = env.action_space[1].shape[0]
	d_dim = env.action_space[0].n
	max_action = env.action_space[1].high
	min_action = env.action_space[1].low
	print((state_dim, d_dim, c_dim, min_action, max_action))
	print(s)
	for _ in range(30):
		action = env.action_space.sample()
		#print(action)
		s_prime, r, done, info = env.step(action)
		#print(r)
		#print(s_prime)
		if done:
			break

