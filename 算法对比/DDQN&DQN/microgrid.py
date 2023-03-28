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

		self.max_Soc = 200                                #储能装置最大容量
		self.min_Soc = 0.1*self.max_Soc                  #储能装置最小容量
		self.start_time = 0                               #开始时间
		self.end_time = 23                                #结束时间
		self.all_anums = 48
		self.reg1_max = 10  # 生产设备1最大工作功率
		self.reg2_max = 20  # 生产设备2最大工作功率
		self.actiondict = {
			0: [0, 0, 0, 0, -50],
			1: [0, 0, 0, 0, 0],
			2: [0, 0, 0, 0, 50],
			3: [0, 0, 0, 20, -50],
			4: [0, 0, 0, 20, 0],
			5: [0, 0, 0, 20, 50],
			6: [0, 0, 30, 0, -50],
			7: [0, 0, 30, 0, 0],
			8: [0, 0, 30, 0, 50],
			9: [0, 0, 30, 20, -50],
			10: [0, 0, 30, 20, 0],
			11: [0, 0, 30, 20, 50],
			12: [0, 20, 0, 0, -50],
			13: [0, 20, 0, 0, 0],
			14: [0, 20, 0, 0, 50],
			15: [0, 20, 0, 20, -50],
			16: [0, 20, 0, 20, 0],
			17: [0, 20, 0, 20, 50],
			18: [0, 20, 30, 0, -50],
			19: [0, 20, 30, 0, 0],
			20: [0, 20, 30, 0, 50],
			21: [0, 20, 30, 20, -50],
			22: [0, 20, 30, 20, 0],
			23: [0, 20, 30, 20, 50],
			24: [10, 0, 0, 0, -50],
			25: [10, 0, 0, 0, 0],
			26: [10, 0, 0, 0, 50],
			27: [10, 0, 0, 20, -50],
			28: [10, 0, 0, 20, 0],
			29: [10, 0, 0, 20, 50],
			30: [10, 0, 30, 0, -50],
			31: [10, 0, 30, 0, 0],
			32: [10, 0, 30, 0, 50],
			33: [10, 0, 30, 20, -50],
			34: [10, 0, 30, 20, 0],
			35: [10, 0, 30, 20, 50],
			36: [10, 20, 0, 0, -50],
			37: [10, 20, 0, 0, 0],
			38: [10, 20, 0, 0, 50],
			39: [10, 20, 0, 20, -50],
			40: [10, 20, 0, 20, 0],
			41: [10, 20, 0, 20, 50],
			42: [10, 20, 30, 0, -50],
			43: [10, 20, 30, 0, 0],
			44: [10, 20, 30, 0, 50],
			45: [10, 20, 30, 20, -50],
			46: [10, 20, 30, 20, 0],
			47: [10, 20, 30, 20, 50]
		}

		self.reset()
		self.seed()


	def seed(self, seed=None):
		#产生一个随机化时需要的种子，同时返回一个np_random对象，支持后续的随机化生成操作
		self.np_random, seed = seeding.np_random(seed)  
		return [seed]

	def getaction(self, action, soc):
		act_list = self.actiondict[action]
		Preg1 = act_list[0]
		Preg2 = act_list[1]
		Pdg1 = act_list[2]
		Pdg2 = act_list[3]
		pb_t = act_list[4]
		if pb_t >= 0:
			Pb = min(pb_t, self.max_Soc-soc)
		else:
			Pb = max(pb_t, self.min_Soc-soc)

		return Preg1, Preg2, Pdg1, Pdg2, Pb


	def step(self, action):
		L23, L22, L21, L20, L19, L18, L17, L16, L15, L14, L13, L12, L11, L10, L9, L8, L7, L6, L5, L4, L3, L2, L1, L0, pnet, soc, t = self.state

		if t == self.end_time:
			done = True
		else:
			done = False

		Preg1, Preg2, Pdg1, Pdg2, Pb = self.getaction(action, soc)

		#计算下一时刻状态
		L_0 = L23
		L_1 = L0
		L_2 = L1
		L_3 = L2
		L_4 = L3
		L_5 = L4
		L_6 = L5
		L_7 = L6
		L_8 = L7
		L_9 = L8
		L_10 = L9
		L_11 = L10
		L_12 = L11
		L_13 = L12
		L_14 = L13
		L_15 = L14
		L_16 = L15
		L_17 = L16
		L_18 = L17
		L_19 = L18
		L_20 = L19
		L_21 = L20
		L_22 = L21
		L_23 = L22
		soc_ = soc + Pb
		if not done:
			pnet_ = self.P_net[int(t+1)]
			t_ = t+1
		else:
			pnet_ = self.P_net[0]
			t_ = 0
        

		Pg = pnet + Pb + Preg1 + Preg2 - Pdg1 - Pdg2

		#主电网电费计算
		if Pg >= 0:
			rg = -1*Pg*L0
		else:
			rg = -1*0.5*Pg*L0
		#发电机电费计算
		rdg1 = -1*(0.01*Pdg1**2 + 6.04*Pdg1 + 13.187)
		rdg2 = -1*(0.01*Pdg2**2 + 7.15*Pdg2 + 6.615)

		rreg1 = 45.82 * (Preg1 ** 0.5)
		rreg2 = 57.51 * (Preg2 ** 0.5)
        
        #奖励计算
		self.reward = (rg + rdg1 + rdg2 + rreg1 + rreg2)*0.001
		self.state = (L_23, L_22, L_21, L_20, L_19, L_18, L_17, L_16, L_15, L_14, L_13, L_12, L_11, L_10, L_9, L_8, L_7, L_6, L_5, L_4, L_3, L_2, L_1, L_0, pnet_, soc_, t_)

		return np.array(self.state), self.reward, done, {}





	def reset(self):
		self.load = np.loadtxt('./data/load.csv', dtype=np.float64, delimiter=',', unpack=True) / 1000
		self.solar = np.loadtxt('./data/solar.csv', dtype=np.float64, delimiter=',', unpack=True) / 1000
		self.wind = np.loadtxt('./data/wind.csv', dtype=np.float64, delimiter=',', unpack=True) / 1000
		self.lmp = np.loadtxt('./data/lmp_1.csv', dtype=np.float64, delimiter=',', unpack=True) / 10
		self.P_net = self.load - self.solar - self.wind  # 内部功率变化
		self.LMP = self.lmp  # 电价变化

		'''
		for i in range(24):
			self.P_net[i] +=  random.uniform(-15, 15)
			self.LMP[i] += random.uniform(-1.5, 1.5)
		'''

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
		self.action_space = spaces.Discrete(self.all_anums)
		self.observation_space = spaces.Box(self.low_state, self.high_state, dtype=np.float32)
		self.state = np.array([self.LMP[1],
							   self.LMP[2],
							   self.LMP[3],
							   self.LMP[4],
							   self.LMP[5],
							   self.LMP[6],
							   self.LMP[7],
							   self.LMP[8],
							   self.LMP[9],
							   self.LMP[10],
							   self.LMP[11],
							   self.LMP[12],
							   self.LMP[13],
							   self.LMP[14],
							   self.LMP[15],
							   self.LMP[16],
							   self.LMP[17],
							   self.LMP[18],
							   self.LMP[19],
							   self.LMP[20],
							   self.LMP[21],
							   self.LMP[22],
							   self.LMP[23],
							   self.LMP[0],
							   self.P_net[0],
							   0.10*self.max_Soc,
							   0
							   ])
		return self.state 

	def close(self):
		if self.viewer: self.viewer.close()

if __name__ =="__main__":
	env = MicroGrid()
	# print(env.reg_p)
	s = env.reset()
	state_dim = env.observation_space.shape[0]
	d_dim = env.action_space.n
	print((state_dim, d_dim))
	# print(s)
	for _ in range(30):
		action = env.action_space.sample()
		#print(action)
		s_prime, r, done, info = env.step(action)
		#print(r)
		#print(s_prime)
		if done:
			break

