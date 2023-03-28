from model import Actor
import numpy as np
import torch
import gym
from microgrid import MicroGrid
from matplotlib import pyplot as plt
import seaborn as sns  # 导入模块

sns.set()  # 设置美化参数，一般默认就好

gym.logger.set_level(40)


def Action_adapter(d, c):
    c[0] = c[0] * 30
    c[1] = c[1] * 20
    c[2] = (c[2] - 0.5) * 100

    return (d, c)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./model/actor_model.ckpt"

max_Soc = 200
min_Soc = 0.1 * max_Soc
reg_anums1 = reg_anums2 = 11
actor = Actor(27, reg_anums1 * reg_anums2, 3, 128).to(device)
actor.load_state_dict(torch.load(model_path))
test_days = 3

env = MicroGrid()
state, LMP = env.reset()
soc = []
PB = []
DG1 = []
DG2 = []
PREG1 = []
PREG2 = []
PG = []
Reward = []
step_rewards = []

for i in range(test_days):
    state, lmp = env.reset()
    LMP += list(lmp)
    done = False
    ep_r = 0
    while not done:

        with torch.no_grad():
            soc_now = state[-2]
            Pnet_now = state[-3]
            soc.append(soc_now / max_Soc)
            state = torch.FloatTensor(state).reshape(1, -1).to(device)

            d_probs, mu, sigma = actor(state)
            d = torch.argmax(d_probs).item()
            c = mu.cpu().numpy().flatten()
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
        PG.append(Pnet_now + Pb + preg1 + preg2 - act[1][0] - act[1][1])
        s_prime, r, done, info = env.step(act)
        step_rewards.append(r * -1000)
        ep_r += r
        state = s_prime
    Reward.append(ep_r * -1000)

# print(soc)
# print(PB)
print(Reward)
print(step_rewards)

SUM = 0
for i in range(len(Reward)):
    SUM = SUM + Reward[i]
    Reward[i] = SUM

plt.plot(Reward, label="$Fees$")
plt.xlabel("Days")
plt.ylabel("Accumulated-fees")
plt.legend()
plt.show()

SUM_hour = 0
for i in range(len(step_rewards)):
    SUM_hour = SUM_hour + step_rewards[i]
    step_rewards[i] = SUM_hour

plt.plot(step_rewards, label="$Fees$")
plt.xlabel("Hours")
plt.ylabel("Accumulated-fees")
plt.legend()
plt.show()

np.savetxt('./res/Reward.csv', Reward, fmt='%.2f', delimiter=',')
np.savetxt('./res/step_rewards.csv', step_rewards, fmt='%.2f', delimiter=',')