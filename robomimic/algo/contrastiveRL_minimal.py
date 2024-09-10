import numpy as np
import torch
import torch.nn as nn
from my_utils.model_utils import Policy
from torch.utils.data import Dataset
from torch.distributions.geometric import Geometric


class Dataset_Custom(Dataset):

    def __init__(self, data, gamma):

        self.GEOM = Geometric(1 - gamma)

        # Assume we have data as list of tuples of np arrays where each tuple describes a trajectory

        self.formatted_demos = data

        self.n_trj = len(self.formatted_demos)


    def __len__(self):
        return  # TODO

    def __getitem__(self, idx):

        trj_i = np.random.randint(0, self.n_trj)
        len_trj = self.formatted_demos[trj_i][0].shape[0]

        t = int(torch.clamp(self.GEOM.sample(), max=len_trj - 1).item())
        i = np.random.randint(0, len_trj - t)

        s = self.formatted_demos[trj_i][0][i]
        g = self.formatted_demos[trj_i][2][-1]  # Assuming the goal is the last state of the next states in the trajectory
        a = self.formatted_demos[trj_i][1][i]
        s1 = self.formatted_demos[trj_i][2][i+t]

        return s, g, a, s1





class MLP(nn.Module):

    def __init__(self, input_dim, cond_dim, output_dim):
        super().__init__()

        hidden = 64
        self.f = nn.Sequential(nn.Linear(input_dim+cond_dim, hidden),
                               nn.ReLU(),
                               nn.Linear(hidden, hidden),
                               nn.ReLU(),
                               nn.Linear(hidden, hidden),
                               nn.ReLU(),
                               nn.Linear(hidden, output_dim))

    def forward(self, x, c=None):
        h = x if c is None else torch.cat([x, c], -1)
        return self.f(h)


class ContrastiveRL(nn.Module):

    def __init__(self, input_dim, goal_dim, z_dim, a_dim, offline_reg, device=None):
        super().__init__()

        self.offline_reg = offline_reg  # this is a behavioral cloning regularization on the policy
        self.device = device
        self.input_dim = input_dim

        self.phi = MLP(input_dim, a_dim, z_dim)
        self.psi = MLP(input_dim, 0, z_dim)

        policy_def = {'type': 'continuous',  # 'discrete' or 'continuous'
                      'var': 1.0,
                      'max_action': 1.0,  # max value action
                      'bias_action': 0.0}  # middle value action

        self.pi = Policy(input_dim, goal_dim, a_dim, policy_def)

        self.opt_critic = torch.optim.Adam(list(self.phi.parameters()) + list(self.psi.parameters()), lr=1e-3)
        self.opt_actor = torch.optim.Adam(self.pi.parameters(), lr=1e-3)

    def predict(self, s, s_g):
        mu = self.pi.get_mean(s, s_g)
        return torch.squeeze(mu).detach().cpu().numpy()

    def get_value(self, s, a, g):
        z_sa = self.phi(s, a)
        z_g = self.psi(g)
        logits = torch.einsum('ik, ik->i', z_sa, z_g)
        return logits

    def critic_loss(self, st, at, st1):

        z_sa = self.phi(st, at)
        z_g = self.psi(st1)

        logits = torch.einsum('ik, jk->ij', z_sa, z_g)
        L_pos = nn.functional.binary_cross_entropy_with_logits(logits, torch.eye(logits.shape[0]).to(self.device))

        return L_pos

    def actor_loss(self, st, at, gt):

        a = self.pi.sample_action(st, gt)
        z_sa = self.phi(st, a)
        z_g = self.psi(gt)
        logits = torch.einsum('ik, ik->i', z_sa, z_g)

        log_prob_a_orig, _ = self.pi.get_log_prob(st, at, gt)

        tot_loss = (1 - self.offline_reg) * torch.mean(-1.0 * logits) - self.offline_reg * torch.mean(log_prob_a_orig)

        return tot_loss

    def update(self, batch=None):

        st, gt, at, st1 = batch

        critic_loss = self.critic_loss(st, at, st1)

        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        actor_loss = self.actor_loss(st, at, gt)

        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        return
