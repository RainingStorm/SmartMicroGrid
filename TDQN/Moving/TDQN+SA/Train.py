import gym
import gym_hybrid
from Agent import TDQN
import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns

gym.logger.set_level(40)

class TDQNConfig:
    def __init__(self):
        self.algo = 'TDQN'
        self.env = 'Moving-v0' # env name
        self.critic_model_path = './model/critic_model.ckpt'      # path to save model
        self.actor_model_path = './model/actor_model.ckpt'  
        self.lstm_model_path = './model/lstm_model.ckpt'    
        self.gamma = 0.99
        self.critic_lr = 1e-3
        self.actor_lr = 1e-4
        self.memory_capacity = 50000
        self.batch_size = 256
        self.train_eps = 10000
        self.hidden_dim = 128
        self.soft_tau = 5e-2
        self.epsilon_start = 0.999  # start epsilon of e-greedy policy
        self.epsilon_end = 0.001
        self.epsilon_decay = 8000
        self.gaussian_exploration_noise = 0.05
        self.seed = 3407
        self.display = False
        self.use_lstm = False
        self.policy_noise = 0.2  # Noise added to target policy during critic update
        self.noise_clip = 0.5  # Range to clip target policy noise
        self.policy_freq = 2  # Frequency of delayed policy updates
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")


def env_agent_config(cfg):
    env = gym.make(cfg.env)
    eval_env = gym.make(cfg.env)
    env.seed(cfg.seed)
    eval_env.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    state_dim = env.observation_space.shape[0]
    actionD_dim = env.action_space[0].n
    actionC_dim = env.action_space[1].shape[0]
    max_action = env.action_space[1].high
    min_action = env.action_space[1].low
    print(f'State: {state_dim} ad_dim: {actionD_dim} ac_dim: {actionC_dim}')
    agent = TDQN(state_dim, actionD_dim, actionC_dim, max_action, min_action, cfg, env)
    return env, eval_env, agent

def evaluate_policy(eval_env, agent):
    scores = 0
    turns = 3
    for j in range(turns):
        s = eval_env.reset()
        done, ep_r, steps = False, 0, 0
        while not done:
            # Take deterministic actions at test time
            act = agent.evaluate(s)
            s_prime, r, done, info = eval_env.step(act)
            ep_r += r
            steps += 1
            s = s_prime
        scores += ep_r
    return scores / turns

def train(cfg, env, agent, eval_env):
    print('Start to train ! ')
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}, Random Seed:{cfg.seed}')
    rewards = []
    ma_rewards = []  # moving average rewards
    best_reward = 0
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        i_step = 0
        while not done:
            i_step += 1
            continuous_action = agent.select_continuous_action(state)
            discrete_action = agent.select_discrete_action(state, continuous_action)
            action = (discrete_action, continuous_action)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            agent.memory.push((state, discrete_action, continuous_action, reward, next_state, done))
            agent.update()
            state = next_state

        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
            best_reward = ep_reward

        print('Episode:{}/{}, Reward:{}, ma_rewards:{}'.format(i_ep + 1, cfg.train_eps, ep_reward, ma_rewards[-1]))

        if ma_rewards[-1] > best_reward:
            best_reward = ma_rewards[-1]
            agent.save(cfg.critic_model_path, cfg.actor_model_path)
            print('saving model')

        if (i_ep+1) % 100 == 0:
            score = evaluate_policy(eval_env, agent)
            print('EnvName:', cfg.env, 'score:', score, 'progress:', (i_ep+1) / cfg.train_eps)


    print('Complete training！')
    return rewards, ma_rewards


if __name__ == "__main__":
    cfg = TDQNConfig()

    # train
    env, eval_env, agent = env_agent_config(cfg)
    rewards, ma_rewards = train(cfg, env, agent, eval_env)

    # save results
    np.savetxt('./reward/rewards1.csv', rewards, fmt='%.2f', delimiter=',')
    np.savetxt('./reward/ma_rewards1.csv', ma_rewards, fmt='%.2f', delimiter=',')

    # plot training curve
    sns.set()

    font = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 13,
             }
    plt.xlabel("Episode", font)
    plt.ylabel("Reward", font)
    plt.plot(rewards, label="$Train-rewards$")
    plt.plot(ma_rewards, label="$ma-rewards$")
    plt.legend()
    plt.show()