import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.categorical import Categorical
import copy

class MLP(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        hidden = 64
        self.f = nn.Sequential(nn.Linear(input_dim, hidden),
                               nn.ReLU(),
                               nn.Linear(hidden, hidden),
                               nn.ReLU(),
                               nn.Linear(hidden, hidden),
                               nn.ReLU(),
                               nn.Linear(hidden, output_dim))

    def forward(self, x, goal=None):
        # check if x is a dictionary
        if isinstance(x, dict):
            # print shapes of hte dictionary
            # for key in x.keys():
            #     print(f"shape {key}: {x[key].shape}")
            x = torch.cat([x[key] for key in sorted(x.keys())], dim=-1)
        return self.f(x)
    
    
class MLP_Conditioned(nn.Module):

    def __init__(self, input_dim, goal_dim, output_dim):
        super().__init__()

        hidden = 64
        self._goal_conditioned = False
        if goal_dim > 0:
            self._goal_conditioned = True
        self.f = nn.Sequential(nn.Linear(input_dim+goal_dim, hidden),
                               nn.ReLU(),
                               nn.Linear(hidden, hidden),
                               nn.ReLU(),
                               nn.Linear(hidden, hidden),
                               nn.ReLU(),
                               nn.Linear(hidden, output_dim))

    def forward(self, x, goal):
        # check if x is a dictionary
        if isinstance(x, dict):
            x = torch.cat([x[key] for key in sorted(x.keys())], dim=-1)
        if self._goal_conditioned:
            if isinstance(goal, dict):
                goal = torch.cat([goal[key] for key in sorted(goal.keys())], dim=-1)
            x = torch.cat([x, goal], -1)
        return self.f(x)
    

class Policy(nn.Module):

    def __init__(self, input_dim, cond_dim, output_dim, policy_def):
        super().__init__()

        n_layers = 2 #3
        hidden_units = 32 #256 #128

        self.type = policy_def['type']
        self.var = policy_def['var']
        self.max_action = policy_def['max_action']
        self.bias_action = policy_def['bias_action']

        self.f = nn.ModuleList()
        self.f.append(nn.Linear(input_dim+cond_dim, hidden_units))
        self.f.append(nn.ReLU())
        for _ in range(n_layers):
            self.f.append(nn.Linear(hidden_units, hidden_units))
            self.f.append(nn.ReLU())
        self.f.append(nn.Linear(hidden_units, output_dim))

    def get_mean(self, x, cond=None):
        
        h = torch.cat([x, cond], -1) if cond is not None else x
        for layer in self.f:
            h = layer(h)

        mu = torch.softmax(h, -1) if self.type == 'discrete' else torch.tanh(h) * self.max_action + self.bias_action

        return mu

    def get_log_prob(self, x, a, c=None):

        mu = self.get_mean(x, c)
        dist = Categorical(probs=mu) if self.type == 'discrete' else Normal(loc=mu, scale=self.var)
        return dist.log_prob(a[:,0]) if self.type == 'discrete' else dist.log_prob(a).sum(-1), dist.entropy() # N.log_prob(a[:,0]), N.entropy()

    def sample_action(self, x, c=None):
        mu = self.get_mean(x, c)
        dist = Categorical(probs=mu) if self.type == 'discrete' else Normal(loc=mu, scale=self.var)
        return dist.sample().float()


class MetricRL(nn.Module):

    def __init__(self, input_dim, goal_dim, z_dim, a_dim, device=None):
        super().__init__()

        self.device = device

        self.phi = MLP(input_dim, z_dim)

        policy_def = {'type': 'continuous',  # 'discrete' or 'continuous'
                      'var': 1.0,
                      'max_action': 1.0,  # max value action
                      'bias_action': 0.0}  # middle value action
        
        self.pi = Policy(input_dim, goal_dim, a_dim, policy_def)

        self.opt_critic = torch.optim.Adam(self.phi.parameters(), lr=1e-3)
        self.opt_actor = torch.optim.Adam(self.pi.parameters(), lr=1e-3)

    def get_distance(self, z1, z2):
        dist = torch.linalg.vector_norm(z1 - z2, ord=2, dim=-1)
        return dist

    def predict(self, s, s_g):

        mu = self.pi.get_mean(s, s_g)
        return torch.squeeze(mu).detach().cpu().numpy()

    def get_value(self, s, s_g):

        z = self.phi(s)
        z_g = self.phi(s_g)
        neg_dist = - self.get_distance(z, z_g)
        return neg_dist

    def critic_loss(self, st, st1):

        z = self.phi(st)
        z1 = self.phi(st1)

        action_distance = 1
        L_pos = torch.mean((self.get_distance(z1, z) - action_distance) ** 2)

        idx = torch.randperm(z.shape[0])
        z1_shuffle = z1[idx]
        dist_z_perm = - torch.log(self.get_distance(z, z1_shuffle) + 1e-6)
        L_neg = torch.mean(dist_z_perm)

        return L_pos+L_neg

    def actor_loss(self, st, s_g, st1, at):

        log_prob, entropy = self.pi.get_log_prob(st, at, s_g)
        Vt = self.get_value(st, s_g)
        Vt1 = self.get_value(st1, s_g)
        Adv = (Vt1 - Vt).detach()
        tot_loss = - torch.mean(Adv * log_prob)

        return tot_loss

    def update(self, batch):

        st, gt, at, st1 = batch

        critic_loss = self.critic_loss(st, st1)

        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        actor_loss = self.actor_loss(st, gt, st1, at)

        self.opt_actor.zero_grad()
        actor_loss.backward()
        # if self.policy_clip is not None:
        #     torch.nn.utils.clip_grad_norm_(self.pi.parameters(), self.policy_clip)
        self.opt_actor.step()

        return


