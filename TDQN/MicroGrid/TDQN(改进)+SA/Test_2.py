from Net import Actor, Critic, Lstm
import numpy as np
import torch
import gym
from Microgrid import MicroGrid
from matplotlib import pyplot as plt
import seaborn as sns # 导入模块
sns.set() # 设置美化参数，一般默认就好

gym.logger.set_level(40)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
critic_model_path = './model/critic_model.ckpt'
actor_model_path = './model/actor_model.ckpt'
lstm_model_path = './model/lstm_model.ckpt'
use_lstm = False

def evaluate(state, actor, critic):
    with torch.no_grad():
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if use_lstm:
            state = LSTM(state)
        continuous_action = actor(state).detach().cpu().numpy()
        ca = torch.FloatTensor(continuous_action).unsqueeze(0).to(device)
        q1, q2 = critic(state, ca)
        q_values = (q1 + q2) / 2
        # q_values = torch.min(q1, q2)
        discrete_action = q_values.max(1)[1].item()

    return (discrete_action, continuous_action)


max_Soc = 200
min_Soc = 0.1*max_Soc
reg_anums = 11
eta_ESS = 0.9
test_days = 3
actor = Actor(27, 3, 128).to(device)
actor.load_state_dict(torch.load(actor_model_path))
critic = Critic(30, reg_anums**2, 128).to(device)
critic.load_state_dict(torch.load(critic_model_path))
if use_lstm:
    LSTM = Lstm().to(device)
    LSTM.load_state_dict(torch.load(lstm_model_path))

env = MicroGrid()
LMP = []
soc = []
PB = []
DG1 = []
DG2 = []
PREG1 = []
PREG2 = []
PG = []
Reward = []

for i in range(test_days):
    state, lmp = env.reset()
    LMP += list(lmp)
    done = False
    ep_r = 0
    while not done:
        soc_now = state[-2]
        soc.append(soc_now / max_Soc)
        act = evaluate(state, actor, critic)
        pb_t = act[1][2]
        if pb_t >= 0:
            Pb = min(pb_t * eta_ESS, (max_Soc - soc_now))
        else:
            Pb = max(pb_t / eta_ESS, (min_Soc - soc_now))
        PB.append(Pb)
        pdg1 = act[1][0]
        pdg2 = act[1][1]
        DG1.append(pdg1)
        DG2.append(pdg2)

        preg1 = int(act[0] / reg_anums) * 1
        preg2 = (act[0] % reg_anums) * 2
        #print(act[0])
        PREG1.append(preg1)
        PREG2.append(preg2)

        PG.append(state[-3]+Pb+preg1+preg2-act[1][0]-act[1][1])
        s_prime, r, done, info = env.step(act)
        ep_r += r
        state = s_prime
    Reward.append(ep_r*-1000)
#print(soc)
#print(PB)
#print(PREG)
print(Reward)

x = np.linspace(0, 24*test_days-1, 24*test_days)
y1 = PB
y2 = soc

fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(x, y1, label='Pb')

ax.plot(np.linspace(-1, 24*test_days-2, 24*test_days), np.zeros(24*test_days), '-r')

ax2 = ax.twinx()
ax2.plot(x, y2, '-g*', label='SOC')
fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)

ax.set_xlabel("Time", size = 15)
ax.set_ylabel(r"Pb/kw", size = 15)
ax2.set_ylabel(r"SOC", size = 15)
ax2.grid(False)

plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(x, PG, label = 'PG')

ax.plot(np.linspace(-1, 24*test_days-2, 24*test_days), np.zeros(24*test_days), '-r')

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
ax.bar(x, PREG1, label = 'Preg1')
ax.bar(x, PREG2, label = 'Preg2')

ax.plot(np.linspace(-1, 24*test_days-2, 24*test_days), np.zeros(24*test_days), '-r')

ax2 = ax.twinx()
reg_p = sorted(LMP)[int(len(LMP) / 2)]
ax2.plot(x, LMP, '-g*', label = 'LMP')
ax2.plot(x, np.full(24*test_days, reg_p), '--r', label = 'reg_p')
fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)

ax.set_xlabel("Time")
ax.set_ylabel(r"Preg/kw")
ax2.set_ylabel(r"LMP")
ax2.grid(False)

plt.show()

SUM = 0
for i in range(len(Reward)):
    SUM = SUM + Reward[i]
    Reward[i] = SUM

plt.plot(Reward, label="$Fees$")
plt.xlabel("Days")
plt.ylabel("Accumulated-fees")
plt.legend()
plt.show()