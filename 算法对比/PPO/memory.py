import torch
import numpy as np


class PPOMemory:
    def __init__(self, device, env_with_Dead):
        self.states = []
        self.d_actions = []
        self.c_actions = []
        self.rewards = []
        self.states_ = []
        self.logprobs_d = []
        self.logprobs_c = []
        self.dones = []
        self.dws = []
        self.device = device
        self.env_with_Dead = env_with_Dead

    def sample(self):
        with torch.no_grad():
            s, d, c, r, s_prime, logprob_d, logprob_c, done_mask, dw_mask = \
                torch.tensor(self.states, dtype=torch.float).to(self.device), \
                torch.tensor(self.d_actions, dtype=torch.int64).to(self.device), \
                torch.tensor(self.c_actions, dtype=torch.float).to(self.device), \
                torch.tensor(self.rewards, dtype=torch.float).to(self.device), \
                torch.tensor(self.states_, dtype=torch.float).to(self.device), \
                torch.tensor(self.logprobs_d, dtype=torch.float).to(self.device), \
                torch.tensor(self.logprobs_c, dtype=torch.float).to(self.device), \
                torch.tensor(self.dones, dtype=torch.float).to(self.device), \
                torch.tensor(self.dws, dtype=torch.float).to(self.device)
        return s, d, c, r, s_prime, logprob_d, logprob_c, done_mask, dw_mask

    def push(self, state, d_action, c_action, reward, state_, logprob_d, logprob_c, done, dw):
        self.states.append(state)
        self.d_actions.append(d_action)
        self.c_actions.append(c_action)
        self.logprobs_d.append(logprob_d)
        self.logprobs_c.append(logprob_c)
        self.states_.append(state_)
        self.rewards.append([reward])
        self.dones.append([done])
        self.dws.append([dw])

        if not self.env_with_Dead:
            self.dws = (np.array(self.dws) * False).tolist()

    def clear(self):
        self.states = []
        self.d_actions = []
        self.c_actions = []
        self.rewards = []
        self.states_ = []
        self.logprobs_d = []
        self.logprobs_c = []
        self.dones = []
        self.dws = []
