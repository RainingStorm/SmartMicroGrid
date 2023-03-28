import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from Net import Actor, Critic, Lstm
from Memory import ReplayBuffer



class TDQN:
    def __init__(self, state_dim, actionD_dim, actionC_dim, max_action, min_action, cfg, env):
        self.device = cfg.device
        self.LSTM = Lstm().to(cfg.device)
        self.critic = Critic(state_dim+actionC_dim, actionD_dim, cfg.hidden_dim).to(cfg.device)
        self.actor = Actor(state_dim, actionC_dim, cfg.hidden_dim).to(cfg.device)
        self.target_critic = Critic(state_dim+actionC_dim, actionD_dim, cfg.hidden_dim).to(cfg.device)
        self.target_actor = Actor(state_dim, actionC_dim, cfg.hidden_dim).to(cfg.device)
        # e-greedy策略相关参数
        self.frame_idx = 0  # 用于epsilon的衰减计数
        self.epsilon = lambda frame_idx: cfg.epsilon_end + \
            (cfg.epsilon_start - cfg.epsilon_end) * \
            math.exp(-1. * frame_idx / cfg.epsilon_decay)
        # copy parameters to target net
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),  lr=cfg.critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.memory = ReplayBuffer(cfg.memory_capacity)
        self.batch_size = cfg.batch_size
        self.gaussian_exploration_noise = cfg.gaussian_exploration_noise
        self.soft_tau = cfg.soft_tau
        self.gamma = cfg.gamma
        self.max_action = max_action
        self.min_action = min_action
        self.policy_noise = cfg.policy_noise
        self.noise_clip = cfg.noise_clip
        self.policy_freq = cfg.policy_freq
        self.use_lstm = cfg.use_lstm
        self.env = env
        self.total_it = 0

    def evaluate(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if self.use_lstm:
                state = self.LSTM(state)
            continuous_action = self.actor(state).detach().cpu().numpy()
            c_a = torch.FloatTensor(continuous_action).unsqueeze(0).to(self.device)
            q_values = self.critic.Q(state, c_a)
            discrete_action = q_values.max(1)[1].item()

        return (discrete_action, continuous_action)

    def select_continuous_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if self.use_lstm:
            state = self.LSTM(state)
        # self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):
            with torch.no_grad():
                continuous_action = self.actor(state).detach().cpu().numpy() + np.random.normal(0, self.gaussian_exploration_noise)
        else:
            continuous_action = self.env.action_space.sample()[1]
        return continuous_action

    def select_discrete_action(self, state, continuous_action):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if self.use_lstm:
            state = self.LSTM(state)
        continuous_action = torch.FloatTensor(continuous_action).unsqueeze(0).to(self.device)
        self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):
            with torch.no_grad():
                q_values = self.critic.Q(state, continuous_action)
                discrete_action = q_values.max(1)[1].item()
        else:
            discrete_action = self.env.action_space.sample()[0]
        return discrete_action

    def update(self):
        self.total_it += 1
        if len(self.memory) < self.batch_size:
            return
        state, discrete_action, continuous_action, reward, next_state, done = self.memory.sample(
            self.batch_size)
        # convert variables to Tensor
        state = torch.FloatTensor(state).to(self.device)
        if self.use_lstm:
            state = self.LSTM(state)
        next_state = torch.FloatTensor(next_state).to(self.device)
        discrete_action = torch.tensor(discrete_action, device=self.device, dtype=torch.int64).unsqueeze(1)
        continuous_action = torch.FloatTensor(continuous_action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        # Get current Q estimates
        Q1, Q2 = self.critic(state, continuous_action)
        current_Q1 = Q1.gather(dim=1, index=discrete_action)
        current_Q2 = Q2.gather(dim=1, index=discrete_action)
        noise = (torch.randn_like(continuous_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_action = (self.target_actor(next_state).detach() + noise)#.clamp(self.min_action, self.max_action)
        target_Q1, target_Q2 = self.target_critic(next_state, next_action)
        next_target_values = torch.min(target_Q1, target_Q2)

        # next_target_values =self.target_critic(next_state, self.target_actor(next_state).detach())

        a_prime = torch.max(next_target_values, 1)[1].unsqueeze(1)

        next_target_q_value = next_target_values.gather(1, a_prime)

        expected_q_values = reward + self.gamma * next_target_q_value * (1 - done)

        value_loss = nn.MSELoss()(current_Q1, expected_q_values.detach()) + nn.MSELoss()(current_Q2, expected_q_values.detach())      # 计算 均方误差loss
        #value_loss = nn.SmoothL1Loss()(current_Q1, expected_q_values.detach()) + nn.SmoothL1Loss()(current_Q2, expected_q_values.detach())   # 计算 Huber loss
        self.critic_optimizer.zero_grad()
        value_loss.backward()   #retain_graph=True
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            policy_loss = self.critic.Q(state, self.actor(state))
            policy_loss = -policy_loss.mean()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.soft_tau) +
                    param.data * self.soft_tau
                )
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.soft_tau) +
                    param.data * self.soft_tau
                )

    def save(self, path1, path2):
        torch.save(self.critic.state_dict(), path1)
        torch.save(self.actor.state_dict(), path2)

