import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from robomimic.algo.metric_minimal import MLP
from robomimic.algo.additional_architectures import TemporalUnet
import time
from torch.distributions.uniform import Uniform
from torch.autograd.functional import jacobian as nabla
import matplotlib.pyplot as plt
from collections import OrderedDict


class DiffuserAlgo(nn.Module):

    def __init__(self, input_dim, goal_dim, z_dim, a_dim, norms=None, gamma=1.0, H=128, T=100, device=None):
        super().__init__()

        self.device = device

        self.input_dim = input_dim
        self.goal_dim = goal_dim
        self.z_dim = z_dim
        self.a_dim = a_dim

        # this is needed only if the data is not normalized
        if norms is None:
            self.s_norm = torch.tensor([0, 1]).float().to(self.device)
            self.a_norm = torch.tensor([0, 1]).float().to(self.device)
        else:
            self.s_norm, self.a_norm = norms[0], norms[1]

        self.gradients_cumulations = 10
        self.gradients_iters = 0

        self.gamma = gamma

        self.T = T
        self.H = H #128
        self.all_time = torch.linspace(1, self.T, self.T).float().to(self.device)

        # simplified version 
        self.input_dim = input_dim
        self.beta = torch.linspace(0.0001, 0.001, self.T).to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cat([torch.prod(self.alpha[:i]).view(1) for i in range(1, self.T+1)])

        # their version
        # s = 0.008
        # x = np.linspace(0, self.T+1, self.T+1)
        # alphas_cumprod = np.cos(((x / self.T+1) + s) / (1 + s) * np.pi * 0.5) ** 2
        # alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        # self.alpha_bar = torch.tensor(alphas_cumprod).float().to(device)
        # betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        # self.beta = torch.tensor(np.clip(betas, a_min=0, a_max=0.999)).float().to(device)
        # self.alpha = 1 - self.beta

        self.drift_const = 0.001

        self.f = TemporalUnet(32, input_dim+a_dim, 1, dim=32, dim_mults=(1, 2, 4, 8), attention=False,) # (1, 2, 4, 8)
        self.J = MLP(input_dim+a_dim, 1)
        self.opt = torch.optim.Adam(list(self.f.parameters())+list(self.J.parameters()), lr=1e-3)
        
    def forward(self, x):
        return self.predict(x)


    def predict(self, s):
        # TODO: need to be rewritten 
        st = torch.from_numpy(s[:, :self.input_dim]).float().to(self.device)
        gt = torch.from_numpy(self.get_goal(s[:, self.input_dim:])).float().to(self.device)

        with torch.no_grad():
            x = torch.randn((1, self.input_dim+self.a_dim, self.H)).to(self.device)
            for t in range(self.T, 0, -1):
                # the first and last state musth be valid states
                x[0, :self.input_dim, 0] = (st - self.s_norm[0]) / self.s_norm[1]
                x[0, :self.input_dim, -1] = (gt - self.s_norm[0]) / self.s_norm[1]
                x = self.reverse_diffusion_step(x, t)
                # TODO: reinitialize the first and last state to st and gt respectively
        a = torch.squeeze(x[0, self.input_dim:, 0]).detach().cpu().numpy()

        # if self.convert_to_hot:
        return a
        # else:
            # return (a * self.a_norm[1].detach().cpu().numpy()) + self.a_norm[0].detach().cpu().numpy()

    def reverse_diffusion_step(self, xt, t):

        z = torch.randn_like(xt)
        z = z * 0 if t == 1 else z
        sigma_t = (self.beta[t - 1] ** 2)

        epsilon_hat = self.f(xt, 0, self.all_time[t-1:t])

        xt1 = (1 / torch.sqrt(self.alpha[t - 1])) * (xt - (1 - self.alpha[t - 1]) / torch.sqrt(1 - self.alpha_bar[t - 1]) * epsilon_hat)

        # this is one of the versions of the paper -
        # nabla_J = torch.transpose(nabla(lambda x: torch.sum(self.J(x)), torch.transpose(xt, 2, 1)), 2, 1)
        # xt1 += self.drift_const * sigma_t * nabla_J
        xt1 += z * sigma_t
        return xt1

    def get_diffusion_loss(self, st, at, st1):

        s = torch.transpose(torch.cat([self.state_preprocessing(st), self.state_preprocessing(st1)[:, -1:]], 1), 1, 2)
        a = torch.transpose(torch.cat([at, at[:, -1:] * 0], 1), 1, 2)
        trj_0 = torch.cat([s, a], 1)
        s0 = trj_0[0, :self.input_dim, 0]
        sg = trj_0[0, :self.input_dim, -1]

        t = torch.randint(1, self.T+1, size=(1,)).to(self.device)
        epsilon = torch.randn(size=trj_0.shape).to(self.device)

        alpha_bar_t1 = self.alpha_bar[t-1]
        trj_t = torch.sqrt(alpha_bar_t1) * trj_0 + torch.sqrt(1 - alpha_bar_t1) * epsilon

        trj_t[0, :self.input_dim, 0] = s0
        trj_t[0, :self.input_dim, -1] = sg

        epsilon_hat = self.f(trj_t, 0, t)

        loss = torch.mean((epsilon - epsilon_hat) ** 2)
        
        info = OrderedDict()

        info["actor/tot_loss"] = loss

        return loss, info 

    def get_reward_loss(self, st, at, rt):

        rt_reversed = torch.flip(rt, dims=(1,))
        discount = torch.cat([torch.tensor([[self.gamma ** t]]).float() for t in range(rt.shape[1])], -1).to(self.device)
        R_reversed = torch.cumsum(discount * rt_reversed, 1)
        R = torch.flip(R_reversed, dims=(1,))

        trj = torch.cat([self.state_preprocessing(st)[0], at[0]], 1)
        R_hat = self.J(trj)

        return torch.mean((R_hat[:, 0] - R[0])**2)


    def update(self, batch):

        st, at, rt, st1 = batch

        st = (st - self.s_norm[0]) / self.s_norm[1]
        st1 = (st1 - self.s_norm[0]) / self.s_norm[1]
        at = (at - self.a_norm[0]) / self.a_norm[1]

        diff_loss, reward_loss = None, None
        for i in range(st.shape[1] + 1 - self.H):

            diff_loss = self.get_diffusion_loss(st[:,i:i+self.H-1], at[:,i:i+self.H-1], st1[:,i:i+self.H-1])
            reward_loss = self.get_reward_loss(st[:,i:i+self.H-1], at[:,i:i+self.H-1], rt[:,i:i+self.H-1])
            tot_loss = diff_loss + reward_loss

            if self.gradients_iters == 0:
                self.opt.zero_grad()
            tot_loss.backward()
            self.gradients_iters += 1
            if self.gradients_iters == self.gradients_cumulations:
                self.opt.step()
                self.gradients_iters = 0

        return
