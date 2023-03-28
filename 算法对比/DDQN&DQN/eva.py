import gym
from Net import NetApproximator
import numpy as np
import torch

model_path = './dqn_model.ckpt'
env = gym.make('CartPole-v0')#MountainCar-v0
feats_shape = env.observation_space.shape
num_actions = env.action_space.n
hidden_dim = 256
behavior_Q = NetApproximator(input_dim = feats_shape[0], output_dim = num_actions, hidden_dim = hidden_dim)
behavior_Q.load_state_dict(torch.load(model_path))

for i_episode in range(5):
    s0 = env.reset()
    total_reward, time_in_episode = 0,0
    done = False
    for t in range(300):
    #while not done:
        env.render()
        action = int(np.argmax(behavior_Q(s0)))
        s1, reward, done, info = env.step(action)
        total_reward+= reward
        s0 = s1
        time_in_episode += 1
        if done:
            print("Episode finished after {} timesteps".format(time_in_episode))
            break
env.close()