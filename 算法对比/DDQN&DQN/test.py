from Net import  NetApproximator
import numpy as np
import torch
import gym
from microgrid import MicroGrid
from matplotlib import pyplot as plt
import seaborn as sns # 导入模块
sns.set() # 设置美化参数，一般默认就好

gym.logger.set_level(40)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
critic_model_path = './ddqn_model.ckpt'

policy_net = NetApproximator(27, 48, hidden_dim=128).to(device)
policy_net.load_state_dict(torch.load(critic_model_path))
max_Soc = 200
min_Soc = 0.1*max_Soc
reg_anums = 11
eta_ESS = 0.9
test_days = 3
env = MicroGrid()

Reward = []
step_rewards = []


def predict(state):
    with torch.no_grad():
        state = torch.tensor(
            np.array(state), device=device, dtype=torch.float32)
        q_values = policy_net(state)
        action = q_values.max(1)[1].item()
        # action = int(np.argmax(q_values.data.numpy()))
    return action




for i in range(test_days):
    state = env.reset()
    # print(state)
    done = False
    ep_r = 0
    while not done:
        act = predict(state)
        # print(act)
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