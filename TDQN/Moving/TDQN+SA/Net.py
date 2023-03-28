import torch
import torch.nn as nn
import torch.nn.functional as F

class Lstm(nn.Module):
    def __init__(self):
        super(Lstm, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=24, num_layers=1, batch_first=True)
    def forward(self, state):
        x = state[:, 0:24]
        l = x.shape[0]
        x = x.reshape(l, 24, 1)
        y, _ = self.lstm(x, None)
        y = y[:, -1, :]
        out = torch.cat([y, state[:, 24:27]], 1)
        return out

class Critic(nn.Module):
    def __init__(self, n_obs, output_dim, hidden_size, init_w=3e-3, d_model=128):
        super(Critic, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=24, num_layers=1, batch_first=True)

        # Q1 architecture
        self.linear1 = nn.Linear(n_obs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.adv1 = nn.Linear(hidden_size, hidden_size)
        self.adv2 = nn.Linear(hidden_size, output_dim)

        self.val1 = nn.Linear(hidden_size, hidden_size)
        self.val2 = nn.Linear(hidden_size, 1)

        # 随机初始化为较小的值
        self.adv2.weight.data.uniform_(-init_w, init_w)
        self.adv2.bias.data.uniform_(-init_w, init_w)
        self.val2.weight.data.uniform_(-init_w, init_w)
        self.val2.bias.data.uniform_(-init_w, init_w)

        # Q2 architecture
        self.linear3 = nn.Linear(n_obs, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.adv3 = nn.Linear(hidden_size, hidden_size)
        self.adv4 = nn.Linear(hidden_size, output_dim)

        self.val3 = nn.Linear(hidden_size, hidden_size)
        self.val4 = nn.Linear(hidden_size, 1)

        self.prenet = nn.Linear(2, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=256, nhead=2)
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
        
    def forward(self, state, action):
        out = self.prenet(action.unsqueeze(0))
        out = self.encoder_layer(out)
        out = out.transpose(0, 1)
        stats = out.mean(dim=1)
        sa_out = self.pred_layer(stats)
        sa = torch.cat([state, sa_out], 1)

        q1 = F.relu(self.linear1(sa))
        q1 = F.relu(self.linear2(q1))
        adv_1 = F.relu(self.adv1(q1))
        adv_1 = self.adv2(adv_1)
        adv_ave1 = torch.mean(adv_1, dim=1, keepdim=True)
        val_1 = F.relu(self.val1(q1))
        val_1 = self.val2(val_1)
        Q1 = val_1 + adv_1 - adv_ave1

        q2 = F.relu(self.linear3(sa))
        q2 = F.relu(self.linear4(q2))
        adv_2 = F.relu(self.adv3(q2))
        adv_2 = self.adv4(adv_2)
        adv_ave2 = torch.mean(adv_2, dim=1, keepdim=True)
        val_2 = F.relu(self.val3(q2))
        val_2 = self.val4(val_2)
        Q2 = val_2 + adv_2 - adv_ave2
        return Q1, Q2

    def Q(self, state, action):
        Q1, Q2 = self.forward(state, action)
        adv_q1 = torch.mean(Q1, dim=1, keepdim=True)
        adv_q2 = torch.mean(Q2, dim=1, keepdim=True)
        # Q = Q1 + Q2 - adv_q1 - adv_q2
        # Q1 = torch.softmax(Q1, dim=1)
        # Q2 = torch.softmax(Q2, dim=1)
        Q = Q1 + Q2
        return Q


class Actor(nn.Module):
    def __init__(self, n_obs, output_dim, hidden_size, init_w=3e-3):
        super(Actor, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=24, num_layers=1, batch_first=True)
        self.linear1 = nn.Linear(n_obs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.linear4 = nn.Linear(hidden_size, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x1 = torch.tanh(self.linear3(x))
        x2 = torch.tanh(self.linear4(x))

        x1 = (0.5 * x1 + 0.5)
        x2 = x2

        x = torch.cat([x1, x2], 1)

        return x.squeeze(0)

if __name__ == "__main__":
    actor = Actor(10, 2, 64)
    critic = Critic(12, 3, 64)
    state = [-0.1686474595029397, 0.32527802314528653, 0, 0.9867845982128716, 0.16203751642709702, -0.7942254349627129, 0.4490119383950545, 0.6376973303718508, 0, 0.005]
    state = torch.FloatTensor(state).unsqueeze(0)
    a_c = actor(state).detach().cpu().numpy()
    print(a_c)
    a_c = torch.FloatTensor(a_c).unsqueeze(0)
    #print(torch.cat([state,a_c],1).shape)
    q_values = critic(state, a_c)
    print(q_values)
    a_d = q_values.max(1)[1].item()
    print(a_d)
    #print(actor(state).item.numpy())
    #print(actor(state)[:][0][0])