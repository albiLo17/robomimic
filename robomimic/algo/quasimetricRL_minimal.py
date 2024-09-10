import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchqmet
from typing import *


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


class GradMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, mult: Union[float, torch.Tensor]) -> torch.Tensor:
        ctx.mult_is_tensor = isinstance(mult, torch.Tensor)
        if ctx.mult_is_tensor:
            assert not mult.requires_grad
            ctx.save_for_backward(mult)
        else:
            ctx.mult = mult
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.mult_is_tensor:
            mult, = ctx.saved_tensors
        else:
            mult = ctx.mult
        return grad_output * mult, None


def grad_mul(x: torch.Tensor, mult: Union[float, torch.Tensor]) -> torch.Tensor:
    if not isinstance(mult, torch.Tensor) and mult == 0:
        return x.detach()
    return GradMul.apply(x, mult)


def softplus_inv_float(y: float) -> float:
    threshold: float = 20.
    if y > threshold:
        return y
    else:
        return np.log(np.expm1(y))


class QuasiMetric(nn.Module):

    def __init__(self, input_dim, goal_dim, z_dim, a_dim, device=None):
        super().__init__()
        # TODO: https://github.com/quasimetric-learning/torch-quasimetric SCARICAMI!!!

        self.device = device

        self.input_dim = input_dim
        self.goal_dim = goal_dim
        self.z_dim = z_dim
        self.a_dim = a_dim

        self.epsilon = 0.25
        self.step_cost = 1.0
        init_lagrange_multiplier = 1.0
        self.raw_lagrange_multiplier = nn.Parameter(
            torch.tensor(softplus_inv_float(init_lagrange_multiplier), dtype=torch.float32))
        self.softplus_offset = 500
        self.softplus_beta = 0.01
        self.weight = 0.1
        self.w_pi = 0.05

        self.iqe = torchqmet.IQE(z_dim, int(z_dim // 16)) # 4

        self.n_critics = 2

        policy_def = {'type': 'continuous',  # 'discrete' or 'continuous'
                      'var': 1.0,
                      'max_action': 1.0,  # max value action
                      'bias_action': 0.0}  # middle value action

        self.f = nn.ModuleList([MLP(input_dim, 0, z_dim) for _ in range(self.n_critics)])
        # self.pi = Policy(input_dim, goal_dim, a_dim, policy_def)
        self.pi = MLP(input_dim, goal_dim, a_dim)

        self.g = nn.ModuleList([MLP(z_dim, 0, z_dim) for _ in range(self.n_critics)])
        self.T = MLP(z_dim, a_dim, z_dim)

        self.opt_critic = torch.optim.Adam(list(self.f.parameters())+list(self.T.parameters())+list(self.g.parameters())+[self.raw_lagrange_multiplier], lr=5e-4)
        self.opt_actor = torch.optim.Adam(self.pi.parameters(), lr=3e-5)

    def get_distance(self, x, y, i):
        return self.iqe(self.g[i](x), self.g[i](y))

    def predict(self, s, s_g):
        mu = self.pi(s, s_g)
        return torch.squeeze(mu).detach().cpu().numpy()

    def get_trans(self, z, a):
        return self.T(z, a) + z

    def critic_loss(self, st, at, st1):

        zt = torch.cat([torch.unsqueeze(self.f[i](st), 0) for i in range(self.n_critics)], 0)
        zt1 = torch.cat([torch.unsqueeze(self.f[i](st1), 0) for i in range(self.n_critics)], 0)
        dist_local = torch.cat([torch.unsqueeze(self.get_distance(zt[i], zt1[i], i), 0) for i in range(self.n_critics)], 0)
        lagrange_mult = F.softplus(self.raw_lagrange_multiplier)
        lagrange_mult = grad_mul(lagrange_mult, -1)
        sq_deviation = (dist_local - self.step_cost).relu().square().mean()
        violation = (sq_deviation - self.epsilon ** 2)
        loss_positive = violation * lagrange_mult

        zr = torch.roll(zt1, 1, dims=0)
        dist_global = torch.cat([torch.unsqueeze(self.get_distance(zt[i], zr[i], i), 0) for i in range(self.n_critics)], 0)
        loss_negative = F.softplus(self.softplus_offset - dist_global, beta=self.softplus_beta)
        loss_negative = loss_negative.mean()

        zt1_hat = torch.cat([torch.unsqueeze(self.get_trans(zt[i], at), 0) for i in range(self.n_critics)], 0)
        dist_z_z1 = torch.cat([torch.unsqueeze(self.get_distance(zt1_hat[i], zt1[i], i), 0) for i in range(self.n_critics)], 0)
        dist_z1_z = torch.cat([torch.unsqueeze(self.get_distance(zt1[i], zt1_hat[i], i), 0) for i in range(self.n_critics)], 0)
        loss_trans = self.weight * (dist_z_z1 + dist_z1_z).square().mean()

        return loss_positive, loss_negative, loss_trans

    def actor_loss(self, st, gt, at):

        z = torch.cat([torch.unsqueeze(self.f[i](st), 0) for i in range(self.n_critics)], 0)
        z_g = torch.cat([torch.unsqueeze(self.f[i](gt), 0) for i in range(self.n_critics)], 0)

        a = self.pi(st, gt)
        neg_Qs = torch.cat([torch.unsqueeze(self.get_distance(self.get_trans(z[i].detach(), a), z_g[i].detach(), i), 0) for i in range(self.n_critics)], 0)
        actor_loss = neg_Qs.mean(-1).max()

        # log_prob, entropy = self.pi.get_log_prob(st, a, gt)
        # alpha = grad_mul(self.raw_entropy_weight.exp(), -1)
        # actor_loss += alpha * (self.target_entropy - entropy.mean())

        actor_loss += self.w_pi * torch.mean((a - at) ** 2)

        return actor_loss


    def update(self, batch=None):

        st, gt, at, st1 = batch

        L_local, L_global, L_trans = self.critic_loss(st, at, st1)
        critic_loss = L_local + L_global + L_trans

        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        L_pi = self.actor_loss(st, gt, at)

        self.opt_actor.zero_grad()
        L_pi.backward()
        self.opt_actor.step()

        return

