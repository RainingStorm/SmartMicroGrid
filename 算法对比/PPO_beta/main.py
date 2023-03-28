import numpy as np
import torch
import gym
from microgrid import MicroGrid
from PPO import PPO
import argparse
from matplotlib import pyplot as plt
import seaborn as sns

gym.logger.set_level(40)

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--EnvIdex', type=int, default=6, help='BWv3, BWHv3, Lch_Cv2, PV0, Humanv2, HCv2, MOV')
parser.add_argument('--seed', type=int, default=3407, help='random seed')
parser.add_argument('--update_fre', type=int, default=2048, help='frequency of agent update')
parser.add_argument('--Max_train_epsiodes', type=int, default=20000, help='Max training steps')
parser.add_argument('--eval_interval', type=int, default=2048, help='Model evaluating interval, in steps.')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
parser.add_argument('--clip_rate', type=float, default=0.20, help='PPO Clip rate')
parser.add_argument('--K_epochs', type=int, default=8, help='PPO update times')
parser.add_argument('--net_width', type=int, default=128, help='Hidden net width')
parser.add_argument('--a_lr', type=float, default=1e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=1e-4, help='Learning rate of critic')
parser.add_argument('--l2_reg', type=float, default=1e-3, help='L2 regulization coefficient for Critic')
parser.add_argument('--batch_size', type=int, default=64, help='lenth of sliced trajectory ')
parser.add_argument('--entropy_coef', type=float, default=1e-3, help='Entropy coefficient of Actor')
parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')
opt = parser.parse_args()
print(opt)


def Action_adapter(d, c):
    c[0] = c[0]*30
    c[1] = c[1]*20
    c[2] = (c[2]-0.5)*100

    return  (d, c)


def Reward_adapter(r, EnvIdex):
    # For BipedalWalker
    if EnvIdex == 0 or EnvIdex == 1:
        if r <= -100: r = -1
    # For Pendulum-v0
    elif EnvIdex == 3:
        r = (r + 8) / 8
    return r


def evaluate_policy(env, model, render, EnvIdex):
    scores = 0
    turns = 3
    for j in range(turns):
        s, _ = env.reset()
        done, ep_r, steps = False, 0, 0
        while not done:
            # Take deterministic actions at test time
            d, c = model.evaluate(s)
            act = Action_adapter(d, c)  # [0,1] to [-max,max]
            s_prime, r, done, info = env.step(act)
            #r = Reward_adapter(r, EnvIdex)

            ep_r += r
            steps += 1
            s = s_prime
            if render:
                env.render()
        scores += ep_r
    return scores / turns


def main():
    render = False

    #EnvName = ['BipedalWalker-v3', 'BipedalWalkerHardcore-v3', 'LunarLanderContinuous-v2', 'Pendulum-v1', 'Humanoid-v2',
               #'HalfCheetah-v2', 'Sliding-v0']
    #BriefEnvName = ['BWv3', 'BWHv3', 'Lch_Cv2', 'PV0', 'Humanv2', 'HCv2']
    Env_With_Dead = [True, True, True, False, True, False, False]
    EnvIdex = opt.EnvIdex
    env_with_Dead = Env_With_Dead[EnvIdex]  # Env like 'LunarLanderContinuous-v2' is with Dead Signal. Important!
    env = MicroGrid()
    eval_env = MicroGrid()
    state_dim = env.observation_space.shape[0]
    c_dim = env.action_space[1].shape[0]  # 指一次要选几个连续动作
    d_dim = env.action_space[0].n
    max_action = env.action_space[1].high
    #max_steps = env._max_episode_steps
    print('Env:','MicroGrid', '  state_dim:', state_dim, 'd_dim:', d_dim, 'c_dim:', c_dim,
          '  max_a:', max_action, '  min_a:', env.action_space[1].low)
    update_fre = opt.update_fre  # lenth of long trajectory

    Max_train_epsiodes = opt.Max_train_epsiodes
    eval_interval = opt.eval_interval  # in steps

    random_seed = opt.seed
    print("Random Seed: {}".format(random_seed))
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    eval_env.seed(random_seed)
    np.random.seed(random_seed)

    kwargs = {
        "state_dim": state_dim,
        "d_dim": d_dim,
        "c_dim": c_dim,
        "env_with_Dead": env_with_Dead,
        "gamma": opt.gamma,
        "lambd": opt.lambd,  # For GAE
        "clip_rate": opt.clip_rate,  # 0.2
        "K_epochs": opt.K_epochs,
        "net_width": opt.net_width,
        "a_lr": opt.a_lr,
        "c_lr": opt.c_lr,
        "l2_reg": opt.l2_reg,  # L2 regulization for Critic
        "batch_size": opt.batch_size,
        "entropy_coef": opt.entropy_coef,
        # Entropy Loss for Actor: Large entropy_coef for large exploration, but is harm for convergence.
        "entropy_coef_decay": opt.entropy_coef_decay
    }

    model = PPO(**kwargs)

    traj_lenth = 0
    total_steps = 0
    Rewards = []
    ma_rewards = []
    best_score = 0.0
    i_ep = 0
    while i_ep < Max_train_epsiodes:
        s, _ = env.reset()
        done, ep_r, steps = False, 0, 0

        '''Interact & trian'''
        while not done:
            traj_lenth += 1
            steps += 1
            total_steps += 1

            if render:
                env.render()

            d, logprob_d, c, logprob_c, c_buffer = model.select_action(s)

            act = Action_adapter(d, c)  # [0,1] to [-max,max]
            # print(act)
            s_prime, r, done, info = env.step(act)
            #r = Reward_adapter(r, EnvIdex)

            '''distinguish done between dead|win(dw) and reach env._max_episode_steps(rmax); done = dead|win|rmax'''
            '''dw for TD_target and Adv; done for GAE'''

            dw = False

            model.memory.push(s, d, c_buffer, r, s_prime, logprob_d, logprob_c, done, dw)
            s = s_prime
            #print(r)
            ep_r += r

            '''update if its time'''
            if not render:
                if traj_lenth % update_fre == 0:
                    model.train()
                    traj_lenth = 0

            '''record & log'''
            if total_steps % eval_interval == 0:
                score = evaluate_policy(eval_env, model, False, EnvIdex)
                print('EnvName:', 'MicroGrid', 'seed:', random_seed, 'steps: {}k'.format(int(total_steps / 1000)),
                      'score:', score, 'progress:', i_ep / Max_train_epsiodes)
                if not best_score or best_score < score:
                    best_score = score
                    model.save()
                    print('saving model with score {:.3f}'.format(score))


        i_ep += 1
        if i_ep % 20 == 0:
            Rewards.append(ep_r)

            if ma_rewards:
                ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * Rewards[-1])
            else:
                ma_rewards.append(Rewards[-1])
                best_reward = Rewards[-1]

            if ma_rewards[-1] > best_reward:
                best_reward = ma_rewards[-1]
                #model.save()

            print('Episode:{}/{}, Reward:{}, ma_rewards:{}'.format(i_ep, Max_train_epsiodes, Rewards[-1], ma_rewards[-1]))

    env.close()

    np.savetxt('./reward/rewards1.csv', Rewards, fmt ='%.2f',delimiter = ',' )
    np.savetxt('./reward/ma_rewards1.csv', ma_rewards, fmt ='%.2f',delimiter = ',' )
    sns.set()
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.plot(Rewards, label="$Train-rewards$")
    plt.plot(ma_rewards, label="$ma-rewards$")
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
