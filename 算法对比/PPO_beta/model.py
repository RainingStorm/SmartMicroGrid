import torch.nn as nn
from torch.distributions import Beta, Normal, Categorical


class Actor(nn.Module):
    def __init__(self, state_dim, d_dim, c_dim, hidden_dim):
        super(Actor, self).__init__()

        self.liner = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            #nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
            #nn.ReLU()
        )

        self.d_prob = nn.Sequential(
            nn.Linear(hidden_dim, d_dim),
            nn.Softmax(dim=-1)
        )

        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim, c_dim),
            nn.Softplus()
        )

        self.beta_head = nn.Sequential(
            nn.Linear(hidden_dim, c_dim),
            nn.Softplus()
        )

    def forward(self, state):
        x = self.liner(state)
        d_probs = self.d_prob(x)
        alpha = self.alpha_head(x) + 1.0
        beta = self.beta_head(x) + 1.0
        return d_probs, alpha, beta

    def get_dist(self, state):
        d_probs, alpha, beta = self.forward(state)
        d_dist = Categorical(d_probs)
        c_dist = Beta(alpha, beta)
        return d_dist, c_dist


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        value = self.critic(state)
        return value
