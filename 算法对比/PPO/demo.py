from model import Actor
import torch
import gym
from microgrid import MicroGrid
from matplotlib import pyplot as plt
import seaborn as sns # 导入模块
sns.set() # 设置美化参数，一般默认就好

gym.logger.set_level(40)

def Action_adapter(d, c):
	c[0] = c[0]*40
	c[1] = c[1]*30

	return  (d, c)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./model/actor_model.ckpt"

max_Soc = 200
min_Soc = 0.1*max_Soc
soc_anums = 21
actor = Actor(4, soc_anums, 2, 128).to(device)
actor.load_state_dict(torch.load(model_path))

env = MicroGrid()
state = env.reset()
done = False
ep_r = 0
soc = []
action = []
PG1 = []
PG2 = []

while not done:

	with torch.no_grad():
		soc_now = state[2]
		soc.append(soc_now/200)
		state = torch.FloatTensor(state).reshape(1, -1).to(device)
		
		d_probs, mu, sigma = actor(state)
		d = torch.argmax(d_probs).item()
		c = mu.cpu().numpy().flatten()
	act = Action_adapter(d, c)
	if d >= int(soc_anums / 2):
		Pb = min(((100 / (soc_anums - 1)) * d - 50), (max_Soc - soc_now))
	else:
		Pb = max(((100 / (soc_anums - 1)) * d - 50), (min_Soc - soc_now))
	action.append(Pb)
	#action.append((100/(soc_anums-1))*d-50)
	PG1.append(act[1][0])
	PG2.append(act[1][1])
	s_prime, r, done, info = env.step(act)
	ep_r += r
	state = s_prime
print(soc)
print(action)
print(ep_r)

plt.xlabel("Time")
plt.ylabel("Pb")
plt.plot(action, label="ESS-ACT")
plt.legend()
plt.show()

plt.xlabel("Time")
plt.ylabel("SOC")
plt.plot(soc, label="$SOC$")
plt.legend()
plt.show()

plt.xlabel("Time")
plt.ylabel("PG")
plt.plot(PG1, label="$PG1$")
plt.legend()
#plt.show()
plt.plot(PG2, label="$PG2$")
plt.legend()
plt.show()