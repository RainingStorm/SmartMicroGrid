import torch.nn as nn
from torch.distributions import Normal, Categorical


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

        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, c_dim),
            nn.Sigmoid()
        )

        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_dim, c_dim),
            nn.Softplus()
        )

    def forward(self, state):
        x = self.liner(state)
        d_probs = self.d_prob(x)
        mu = self.mu_head(x)
        sigma = self.sigma_head(x)
        return d_probs, mu, sigma

    def get_dist(self, state):
        d_probs, mu, sigma = self.forward(state)
        d_dist = Categorical(d_probs)
        c_dist = Normal(mu, sigma)
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
