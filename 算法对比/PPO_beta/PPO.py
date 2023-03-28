import copy
import numpy as np
import torch
import math
from model import Actor, Critic
from memory import PPOMemory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPO(object):
    def __init__(
            self,
            state_dim,
            d_dim,
            c_dim,
            env_with_Dead,
            gamma=0.99,
            lambd=0.95,
            clip_rate=0.2,
            K_epochs=10,
            net_width=256,
            a_lr=3e-4,
            c_lr=3e-4,
            l2_reg=1e-3,
            batch_size=64,
            entropy_coef=0,
            entropy_coef_decay=0.9998
    ):

        self.actor = Actor(state_dim, d_dim, c_dim, net_width).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)

        self.critic = Critic(state_dim, net_width).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=c_lr)

        self.env_with_Dead = env_with_Dead
        #self.action_dim = action_dim
        self.clip_rate = clip_rate
        self.gamma = gamma
        self.lambd = lambd
        self.clip_rate = clip_rate
        self.K_epochs = K_epochs
        self.l2_reg = l2_reg
        self.optim_batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay
        self.memory = PPOMemory(device, env_with_Dead)

    def select_action(self, state):  # only used when interact with the env
        with torch.no_grad():
            state = torch.FloatTensor(state).reshape(1, -1).to(device)
            d_dist, c_dist = self.actor.get_dist(state)
            d = d_dist.sample()
            logprob_d = d_dist.log_prob(d).cpu().numpy().flatten()
            d = d.item()

            c_buffer = c_dist.sample()
            c = torch.clamp(c_buffer, 0, 1)
            logprob_c = c_dist.log_prob(c_buffer).cpu().numpy().flatten()
            c = c.cpu().numpy().flatten()
            c_buffer = c_buffer.cpu().numpy().flatten()
            return d, logprob_d, c, logprob_c, c_buffer

    # 这里返回的动作是1维的，这样输入memory中才是2维的

    def evaluate(self, state):  # only used when evaluate the policy.Making the performance more stable
        with torch.no_grad():
            state = torch.FloatTensor(state).reshape(1, -1).to(device)

            d_probs, alpha, beta = self.actor(state)
            d = torch.argmax(d_probs).item()
            mode = (alpha) / (alpha + beta)
            c = mode.cpu().numpy().flatten()

            return d, c  # 检验策略时不再采样动作，而是选均值作为动作

    def train(self):
        self.entropy_coef *= self.entropy_coef_decay
        s, d, c, r, s_prime, logprob_d, logprob_c, done_mask, dw_mask = self.memory.sample()
        #s, a, r, s_prime, logprob_a, done_mask, dw_mask = self.memory.sample()
        #print((s.shape, d.shape, c.shape, r.shape, s_prime.shape, logprob_d.shape, logprob_c.shape, done_mask.shape, dw_mask.shape))
        #print(c.shape)

        ''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_prime)

            '''dw for TD_target and Adv'''
            deltas = r + self.gamma * vs_ * (1 - dw_mask) - vs

            deltas = deltas.cpu().flatten().numpy()
            adv = [0]

            '''done for GAE'''
            for dlt, mask in zip(deltas[::-1], done_mask.cpu().flatten().numpy()[::-1]):
                advantage = dlt + self.gamma * self.lambd * adv[-1] * (1 - mask)
                adv.append(advantage)
            adv.reverse()
            adv = copy.deepcopy(adv[0:-1])
            adv = torch.tensor(adv).unsqueeze(1).float().to(device)
            td_target = adv + vs
           # adv = (adv - adv.mean()) / ((adv.std() + 1e-4))  # sometimes helps    不一定好

        """Slice long trajectopy into short trajectory and perform mini-batch PPO update"""
        optim_iter_num = int(math.ceil(s.shape[0] / self.optim_batch_size))
        for i in range(self.K_epochs):

            # Shuffle the trajectory, Good for training
            perm = np.arange(s.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(device)
            s, d, c, td_target, adv, logprob_d, logprob_c = \
                s[perm].clone(), d[perm].clone(), c[perm].clone(), td_target[perm].clone(), adv[perm].clone(), logprob_d[perm].clone(), logprob_c[perm].clone()
            logprob_a = torch.cat((logprob_d, logprob_c), 1)
            #print(logprob_a.shape)
            #print(td_target.shape)
            '''update the actor'''
            for i in range(optim_iter_num):
                index = slice(i * self.optim_batch_size, min((i + 1) * self.optim_batch_size, s.shape[0]))
                d_dist, c_dist = self.actor.get_dist(s[index])
                # print(distribution.entropy().shape)
                dist_entropy_d = d_dist.entropy().reshape(-1, 1).sum(1, keepdim=True)
                dist_entropy_c = c_dist.entropy().sum(1, keepdim=True)
                dist_entropy = dist_entropy_d + dist_entropy_c
                #print(dist_entropy.shape)

                logprob_d_now = d_dist.log_prob(d[index]).reshape(-1, 1)
                logprob_c_now = c_dist.log_prob(c[index])
                logprob_a_now = torch.cat((logprob_d_now, logprob_c_now), 1)
                #print(logprob_a_now.shape)

                # print(logprob_a[index])
                ratio = torch.exp(logprob_a_now.sum(1, keepdim=True) - logprob_a[index].sum(1, keepdim=True))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
                a_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                # print(a_loss)

                self.actor_optimizer.zero_grad()
                a_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor_optimizer.step()

                '''update the critic'''
                c_loss = (self.critic(s[index]) - td_target[index]).pow(2).mean()
                for name, param in self.critic.named_parameters():
                    if 'weight' in name:
                        c_loss += param.pow(2).sum() * self.l2_reg

                self.critic_optimizer.zero_grad()
                c_loss.backward()
                self.critic_optimizer.step()

        self.memory.clear()

    def save(self):
        torch.save(self.critic.state_dict(), "./model/critic_model.ckpt")
        torch.save(self.actor.state_dict(), "./model/actor_model.ckpt")
