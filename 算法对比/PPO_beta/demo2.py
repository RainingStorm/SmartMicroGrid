from model import Actor
import numpy as np
import torch
import gym
from microgrid import MicroGrid
from matplotlib import pyplot as plt
import seaborn as sns # 导入模块
sns.set() # 设置美化参数，一般默认就好

gym.logger.set_level(40)

def Action_adapter(d, c):
	c[0] = c[0]*30
	c[1] = c[1]*20
	c[2] = (c[2] - 0.5) * 100

	return  (d, c)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./model/actor_model.ckpt"

max_Soc = 200
min_Soc = 0.1*max_Soc
reg_anums1 = reg_anums2 = 11
actor = Actor(27, reg_anums1*reg_anums2, 3, 128).to(device)
actor.load_state_dict(torch.load(model_path))

env = MicroGrid()
state, LMP = env.reset()
done = False
ep_r = 0
soc = []
PB = []
DG1 = []
DG2 = []
PREG1 = []
PREG2 = []
PG = []

while not done:

	with torch.no_grad():
		soc_now = state[-2]
		Pnet_now = state[-3]
		soc.append(soc_now/max_Soc)
		state = torch.FloatTensor(state).reshape(1, -1).to(device)
		
		d_probs, alpha, beta = actor(state)
		d = torch.argmax(d_probs).item()
		mode = (alpha) / (alpha + beta)
		c = mode.cpu().numpy().flatten()
	act = Action_adapter(d, c)
	pb_t = c[2]
	if pb_t >= 0:
		Pb = min(pb_t, (max_Soc - soc_now))
	else:
		Pb = max(pb_t, (min_Soc - soc_now))
	PB.append(Pb)
	pdg1 = c[0]
	pdg2 = c[1]
	DG1.append(pdg1)
	DG2.append(pdg2)
	preg1 = int(act[0] / reg_anums1) * 1
	preg2 = (act[0] % reg_anums2) * 2
	PREG1.append(preg1)
	PREG2.append(preg2)
	PG.append(Pnet_now+Pb+preg1+preg2-act[1][0]-act[1][1])
	s_prime, r, done, info = env.step(act)
	ep_r += r
	state = s_prime
	
#print(soc)
#print(PB)
print(ep_r)


x = np.linspace(0, 23, 24)
total_width, n = 0.85, 2   # 柱状图总宽度，有几组数据
width = total_width / n   # 单个柱状图的宽度
x1 = x - width / 2   # 第一组数据柱状图横坐标起始位置
x2 = x1 + width   # 第二组数据柱状图横坐标起始位置


# 储能设备
fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(x, PB, label='Pb', color='royalblue')

ax.plot(np.linspace(-1, 22, 24), np.zeros(24), '-r')

ax2 = ax.twinx()
ax2.plot(x, soc, '-g*', label='SOC')
fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)

ax.set_xlabel("Time", size = 15)
ax.set_ylabel(r"Pb/kw", size = 15)
ax2.set_ylabel(r"SOC", size = 15)
ax2.grid(False)

plt.show()

# 主电网功率
fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(x, PG, label = 'PG', color='orangered')
fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)

ax.plot(np.linspace(-1, 22, 24), np.zeros(24), '-r')

ax.set_xlabel("Time")
ax.set_ylabel(r"Power/kw")
plt.show()

# 发电机
fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(x1, DG1, width=width, label = 'DG1', color='peru')
ax.bar(x2, DG2, width=width, label = 'DG2', color='mediumpurple')
fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)

ax.set_xlabel("Time")
ax.set_ylabel(r"Power/kw")

plt.show()

# 生产设备
fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(x1, PREG1, width=width, label = 'REG1', color='orange')
ax.bar(x2, PREG2, width=width, label = 'REG2', color='mediumseagreen')

ax.plot(np.linspace(-1, 22, 24), np.zeros(24), '-r')

ax2 = ax.twinx()
reg_p = sorted(LMP)[int(len(LMP) / 2)]
ax2.plot(x, LMP, '-b', label = 'LMP')
# ax2.plot(x, np.full(24, reg_p), '--r', label = 'reg_p')
fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)

ax.set_xlabel("Time")
ax.set_ylabel(r"Preg/kw")
ax2.set_ylabel(r"LMP")
ax2.grid(False)

plt.show()

'''
x = np.linspace(0, 23, 24)
y1 = PB
y2 = soc

fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(x, y1, label = 'Pb')

ax.plot(np.linspace(-1, 22, 24), np.zeros(24), '-r')

ax2 = ax.twinx()
ax2.plot(x, y2, '-g*', label = 'SOC')
fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)

ax.set_xlabel("Time")
ax.set_ylabel(r"Pb/kw")
ax2.set_ylabel(r"SOC")
ax2.grid(False)

plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(x, PG, label = 'PG')

ax.plot(np.linspace(-1, 22, 24), np.zeros(24), '-r')

#ax2 = ax.twinx()
ax.plot(x, DG1, '-g*', label = 'DG1')
ax.plot(x, DG2, '-y*', label = 'DG2')
fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)

ax.set_xlabel("Time")
ax.set_ylabel(r"Power/kw")
#ax2.set_ylabel(r"DG/KW")
#ax2.grid(False)

plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(x, PREG, label = 'Preg')

ax.plot(np.linspace(-1, 22, 24), np.zeros(24), '-r')

ax2 = ax.twinx()
ax2.plot(x, LMP, '-g*', label = 'LMP')
ax2.plot(x, np.full(24, 5.8), '--r', label = 'reg_p')
fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)

ax.set_xlabel("Time")
ax.set_ylabel(r"Preg/kw")
ax2.set_ylabel(r"LMP")
ax2.grid(False)

plt.show()
'''

