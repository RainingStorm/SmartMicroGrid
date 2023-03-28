import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import numpy as np
from memory import ReplayBuffer
from Net import NetApproximator

class DQN_Agent:
    def __init__(self, state_dim, action_dim, cfg):

        self.action_dim = action_dim  # 总的动作个数
        self.device = cfg.device  # 设备，cpu或gpu等
        self.gamma = cfg.gamma  # 奖励的折扣因子
        # e-greedy策略相关参数
        self.frame_idx = 0  # 用于epsilon的衰减计数
        self.epsilon = lambda frame_idx: cfg.epsilon_end + \
            (cfg.epsilon_start - cfg.epsilon_end) * \
            math.exp(-1. * frame_idx / cfg.epsilon_decay)
        self.batch_size = cfg.batch_size
        self.policy_net = NetApproximator(state_dim, action_dim,hidden_dim=cfg.hidden_dim).to(self.device)
        self.target_net = self.policy_net.clone()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.memory = ReplayBuffer(cfg.memory_capacity)
        self.epochs = cfg.epochs
        

    def choose_action(self, state):
        '''选择动作
        '''
        self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):
            action = self.predict(state)
        else:
            #action = cfg.env.action_space.sample()
            action = random.randrange(self.action_dim)
        return action

    def predict(self,state):
        with torch.no_grad():
            state = torch.tensor(
                [state], device=self.device, dtype=torch.float32)
            q_values = self.policy_net(state)
            action = q_values.max(1)[1].item()
            #action = int(np.argmax(q_values.data.numpy()))
        return action

    def update(self, shape):

        if len(self.memory) < self.batch_size:
            return
        # 从memory中随机采样transition
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size)

        state_batch = torch.tensor(state_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)  # 例如tensor([[1],...,[0]])
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)  # tensor([1., 1.,...,1])
        next_state_batch = torch.tensor(next_state_batch, device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device)

        '''计算当前(s_t,a)对应的Q(s_t, a)'''
        '''torch.gather:对于a=torch.Tensor([[1,2],[3,4]]),那么a.gather(1,torch.Tensor([[0],[1]]))=torch.Tensor([[1],[3]])'''
        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)  # 等价于self.forward

        next_q_values = self.policy_net(next_state_batch)
        a_prime = torch.max(next_q_values, 1)[1].unsqueeze(1)  # DDQN

        next_target_values = self.target_net(next_state_batch)
        #a_prime = torch.max(next_target_values, 1)[1].unsqueeze(1)   #DQN

        next_target_q_value = next_target_values.gather(1, a_prime).squeeze(1)
        #代入到next_target_values获得target net对应的next_q_value，即Q’(s_t|a=argmax Q(s_t‘, a))

        # 计算 expected_q_value
        # 对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
        q_target = reward_batch + self.gamma * next_target_q_value * (1-done_batch)



        # self.loss = F.smooth_l1_loss(q_values,q_target.unsqueeze(1)) # 计算 Huber loss
        #loss = nn.MSELoss()(q_values, q_target.unsqueeze(1))  # 计算 均方误差loss
        #loss = self.criterion(q_values.squeeze(1), q_target)
        loss = (q_target - q_values.squeeze(1)).pow(2)
        loss = loss.mean()

        self.policy_net.fit(loss, self.optimizer, self.epochs)





        

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

