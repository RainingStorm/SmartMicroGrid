import gym
from microgrid import MicroGrid
import torch
import numpy as np
from Env import OUNoise
from Agent import DDPG

gym.logger.set_level(40)

class DDPGConfig:
    def __init__(self):
        self.algo = 'DDPG'
        self.env = 'Pendulum-v1' # env name
        self.model_path = './ddpg_model.ckpt'  # path to save results
        self.gamma = 0.99
        self.critic_lr = 1e-3
        self.actor_lr = 1e-4
        self.memory_capacity = 20000     #10000
        self.batch_size = 256    #128
        self.train_eps = 1000
        self.target_update = 4
        self.hidden_dim = 128
        self.soft_tau = 1e-2
        self.display = False
        self.epochs = 1
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")


def env_agent_config(cfg, seed=1):
    env = MicroGrid()
    #env = gym.make(cfg.env)
    env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space[1].shape[0]+1
    agent = DDPG(state_dim, action_dim, cfg)
    return env, agent

def Action_adapter(action_all, ou_noise, i_step):
    '''
    d = int((0.5*action_all[0]+0.5)*11)
    action_all[1] = (0.5 * action_all[1] + 0.5) * 60
    action_all[2] = (0.5 * action_all[2] + 0.5) * 40
    action_all[3] = action_all[3] * 50
    '''
    d = int(action_all[0])
    action_all[1] = ou_noise.get_action(action_all[1], i_step, 0)[0]
    action_all[2] = ou_noise.get_action(action_all[2], i_step, 1)[0]
    action_all[3] = ou_noise.get_action(action_all[3], i_step, 2)[0]
    c = action_all[1:]
    return (d, c)


def train(cfg, env, agent):
    print('Start to train ! ')
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    ou_noise = OUNoise(env.action_space)  # action noise
    rewards = []
    ma_rewards = []  # moving average rewards
    best_reward = 0
    for i_ep in range(cfg.train_eps):
        state, _ = env.reset()
        # print(state)
        ou_noise.reset()
        done = False
        ep_reward = 0
        i_step = 0
        while not done:
            i_step += 1
            action_all = agent.choose_action(state)
            #print(action_all)
            action = Action_adapter(action_all, ou_noise, i_step)

            #action = ou_noise.get_action(
                #action, i_step)  # 即paper中的random process
            #print(action)
            next_state, reward, done, _ = env.step(action)
            #reward = (reward+8)/8
            ep_reward += reward
            agent.memory.push((state, action_all, reward, next_state, done))
            agent.update()
            state = next_state

        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
            best_reward = ep_reward

        if ma_rewards[-1] > best_reward:
            best_reward = ma_rewards[-1]
            agent.save(cfg.model_path)
            print('saving model')
        #if (i_ep+1)%10 == 0:
        print('Episode:{}/{}, Reward:{}, ma_rewards:{}'.format(i_ep+1, cfg.train_eps, ep_reward, ma_rewards[-1]))
        #print('Episode:{}/{}, Reward:{}'.format(i_ep+1, cfg.train_eps, ep_reward))
    print('Complete training！')
    return rewards, ma_rewards


if __name__ == "__main__":
    #torch.autograd.set_detect_anomaly(True)
    cfg = DDPGConfig()

    # train
    env,agent = env_agent_config(cfg,seed=1)
    #print(env.action_space.high)
    rewards, ma_rewards = train(cfg, env, agent)

    np.savetxt('./reward/rewards1.csv', rewards, fmt='%.2f', delimiter=',')
    np.savetxt('./reward/ma_rewards1.csv', ma_rewards, fmt='%.2f', delimiter=',')

    from matplotlib import pyplot as plt 

    plt.plot(rewards, label="$Train-rewards$")
    plt.legend()


    plt.plot(ma_rewards, label="$ma_rewards$")
    plt.legend()
    plt.show()