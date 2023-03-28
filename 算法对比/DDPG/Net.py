import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, n_obs, output_dim, hidden_size, init_w=3e-3):
        super(Critic, self).__init__()
        
        self.linear1 = nn.Linear(n_obs + output_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        # 随机初始化为较小的值
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        # 按维数1拼接
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class Actor(nn.Module):
    def __init__(self, n_obs, output_dim, hidden_size, init_w=3e-3):
        super(Actor, self).__init__()  
        self.linear1 = nn.Linear(n_obs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.linear4 = nn.Linear(hidden_size, 1)
        self.linear5 = nn.Linear(hidden_size, 1)
        self.linear6 = nn.Linear(hidden_size, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)
        self.linear5.weight.data.uniform_(-init_w, init_w)
        self.linear5.bias.data.uniform_(-init_w, init_w)
        self.linear6.weight.data.uniform_(-init_w, init_w)
        self.linear6.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x1 = torch.tanh(self.linear3(x))
        x2 = torch.tanh(self.linear4(x))
        x3 = torch.tanh(self.linear5(x))
        x4 = torch.tanh(self.linear6(x))
        x1 = (0.5 * x1 + 0.5) * 121
        x2 = (0.5 * x2 + 0.5) * 30
        x3 = (0.5 * x3 + 0.5) * 20
        x4 = x4 * 50
        x = torch.cat([x1, x2], 1)
        #print(x)
        x = torch.cat([x, x3], 1)
        x = torch.cat([x, x4], 1)
        return x.squeeze(0)