from Net import  Actor
import numpy as np
import torch
import gym
from microgrid import MicroGrid
from matplotlib import pyplot as plt
import seaborn as sns # 导入模块
sns.set() # 设置美化参数，一般默认就好

gym.logger.set_level(40)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
critic_model_path = './ddpg_model.ckpt'

actor = Actor(27, 4, 128).to(device)
actor.load_state_dict(torch.load(critic_model_path))
max_Soc = 200
min_Soc = 0.1*max_Soc
reg_anums = 11
eta_ESS = 0.9
test_days = 3
env = MicroGrid()

Reward = []
step_rewards = []


def choose_action(state):
    # print(state)
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    action = actor(state)
    return action.detach().cpu().numpy()  # [0]

def Action_adapter(action_all):
    d = int(action_all[0])
    c = action_all[1:]
    return (d, c)


for i in range(test_days):
    state, _ = env.reset()
    # print(state)
    done = False
    ep_r = 0
    while not done:
        act = choose_action(state)
        act = Action_adapter(act)
        s_prime, r, done, info = env.step(act)
        step_rewards.append(r*-1000)
        ep_r += r
        state = s_prime
    Reward.append(ep_r*-1000)

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