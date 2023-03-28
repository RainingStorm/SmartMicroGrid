import gym
from microgrid import MicroGrid
from itertools import count
import torch
from tqdm import tqdm
from agent import DQN_Agent
import numpy as np

gym.logger.set_level(40)

class DQNConfig:
    def __init__(self):
        self.algo = "DDQN"  # name of algo
        self.env = 'CartPole-v1'
        self.model_path = './ddqn_model.ckpt'
        self.train_eps = 1000  # max trainng episodes
        self.gamma = 0.95  #0.95       有修改                                           **1
        self.epsilon_start = 0.90      #0.90  # start epsilon of e-greedy policy    **2
        self.epsilon_end = 0.01
        self.epsilon_decay = 800
        self.lr = 0.0001  # learning rate
        self.memory_capacity = 10000       #10000  # capacity of Replay Memory       **3
        self.batch_size = 128       #128                                              **4
        self.target_update = 4 # update frequency of target net
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # check gpu
        self.hidden_dim = 128  # hidden size of net
        self.display = False
        self.epochs = 1

def env_agent_config(cfg,seed=1):
    #env = gym.make(cfg.env)
    env = MicroGrid()
    env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    # print(action_dim)
    agent = DQN_Agent(state_dim,action_dim,cfg)
    return env,agent

def train(cfg, env, agent):
    print('Start to train !')
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    rewards = []
    ma_rewards = []  # moveing average reward
    best_reward = 0
    for i_ep in range(cfg.train_eps):
    #for i_ep in count(1):
        state = env.reset()
        done = False
        ep_reward = 0
        while True:
            if cfg.display:
                env.render()
            action = agent.choose_action(state)
            #print(action)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            agent.memory.push((state, action, reward, next_state, done))
            state = next_state
            agent.update(env.observation_space.shape)
            if done:
                break
        if (i_ep+1) % cfg.target_update == 0:
            agent.target_net = agent.policy_net.clone()
        if (i_ep+1)%10 == 0:
            print('Episode:{}/{}, Reward:{}, ma_rewards:{}'.format(i_ep+1, cfg.train_eps, ep_reward, ma_rewards[-1]))
        rewards.append(ep_reward)
        # save ma rewards
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
            best_reward = ep_reward

        if ma_rewards[-1] > best_reward:
            best_reward = ma_rewards[-1]
            agent.save(cfg.model_path)
            print('saving model')
        #if ma_rewards[-1] > env.spec.reward_threshold:
        	#break
    print('Complete training！')
    return rewards, ma_rewards


if __name__ == "__main__":
    cfg = DQNConfig()

    # train
    env,agent = env_agent_config(cfg,seed=1)
    rewards, ma_rewards = train(cfg, env, agent)

    np.savetxt('./reward/rewards1.csv', rewards, fmt='%.2f', delimiter=',')
    np.savetxt('./reward/ma_rewards1.csv', ma_rewards, fmt='%.2f', delimiter=',')

    from matplotlib import pyplot as plt 

    plt.plot(rewards, label="$Train-rewards$")
    plt.legend()


    plt.plot(ma_rewards, label="$ma_rewards$")
    plt.legend()
    plt.show()