"""
Implementation of Metric RL
"""
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.models.policy_nets as PolicyNets
from robomimic.algo.metric_minimal import Policy
import robomimic.models.value_nets as ValueNets
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import register_algo_factory_func, ValueAlgo, PolicyAlgo
from robomimic.algo.metric_minimal import MLP, Policy, MLP_Conditioned

import torchqmet
from typing import *

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



@register_algo_factory_func("qrl")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the MetricRL algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    return QuasimetricRL, {}

class QuasimetricRL(PolicyAlgo, ValueAlgo):
    def __init__(self, algo_config, obs_config, global_config, obs_key_shapes, ac_dim, device=None):
        super().__init__(algo_config, obs_config,  global_config, obs_key_shapes, ac_dim, device=device)
        self.pre_trained = self.algo_config.pre_train
        self.pre_train_epochs = self.algo_config.pre_train_epochs
        
        # Setup optimizers
        self._setup_optimizers()

    def _create_networks(self):
        
        self.epsilon = 0.25
        self.step_cost = 1.0
        init_lagrange_multiplier = 1.0
        self.raw_lagrange_multiplier = nn.Parameter(
            torch.tensor(softplus_inv_float(init_lagrange_multiplier), dtype=torch.float32))
        self.softplus_offset = 500
        self.softplus_beta = 0.01
        self.weight = 0.1
        self.w_pi = 0.05

        self.iqe = torchqmet.IQE(self.algo_config.phi_dim, int(self.algo_config.phi_dim // 16)).to(self.device) # 4

        self.n_critics = 2
        # Create nets
        self.nets = nn.ModuleDict()
        

        # Assemble args to pass to actor
        actor_args = dict(self.algo_config.actor.net.common)

        # Add network-specific args and define network class
        if self.algo_config.actor.net.type == "gaussian":
            actor_cls = PolicyNets.MRLGaussianActorNetwork
            actor_args.update(dict(self.algo_config.actor.net.gaussian))
        elif self.algo_config.actor.net.type == "gmm":
            actor_cls = PolicyNets.GMMActorNetwork
            actor_args.update(dict(self.algo_config.actor.net.gmm))
        else:
            # Unsupported actor type!
            raise ValueError(f"Unsupported actor requested. "
                             f"Requested: {self.algo_config.actor.net.type}, "
                             f"valid options are: {['gaussian', 'gmm']}")

        # Actor
        # TODO: check dimensions
        state_dim = np.asarray([v for v in self.obs_shapes.values()]).sum()
        self.nets["pi"] = MLP_Conditioned(input_dim=state_dim,
                                 goal_dim=state_dim,
                                 output_dim=self.ac_dim)
        
        # critic
        self.nets["f"] = nn.ModuleList([MLP(state_dim, self.algo_config.phi_dim) for _ in range(self.n_critics)])
        self.nets["g"] = nn.ModuleList([MLP(self.algo_config.phi_dim, self.algo_config.phi_dim) for _ in range(self.n_critics)])
        self.nets["T"] = MLP_Conditioned(self.algo_config.phi_dim, self.ac_dim, self.algo_config.phi_dim)
        
        
        #actor_cls(
        #     obs_shapes=self.obs_shapes,
        #     goal_shapes=self.goal_shapes,
        #     ac_dim=self.ac_dim,
        #     mlp_layer_dims=self.algo_config.actor.layer_dims,
        #     encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        #     **actor_args,
        # )

        # Critics
        # TODO: evnetually support ensamble critics
        # input_dim = np.asarray([v for v in self.obs_shapes.values()]).sum()
        # self.nets["critic"] = MLP(input_dim, self.algo_config.phi_dim)
        # TODO: extend ActionValue to support higher dimensional value output
        # self.nets["critic"] = ValueNets.ActionValueNetwork(
        #             obs_shapes=self.obs_shapes,
        #             ac_dim=self.ac_dim,
        #             mlp_layer_dims=self.algo_config.critic.layer_dims,
        #             value_bounds=self.algo_config.critic.value_bounds,
        #             goal_shapes=self.goal_shapes,
        #             encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        #         )
        # Send networks to appropriate device

        self.nets = self.nets.float().to(self.device)

        
    def _create_optimizers(self):
        """
        Setup optimizers for actor and critic networks.
        """
        self.optimizers = {
            "critic": torch.optim.Adam(list(self.nets["f"].parameters())+list(self.nets["T"].parameters())+list(self.nets["g"].parameters())+[self.raw_lagrange_multiplier], lr=self.optim_params.critic.learning_rate.initial),
            "actor": torch.optim.Adam(self.nets["pi"].parameters(), lr=self.optim_params.actor.learning_rate.initial),
        }
        
    def _setup_optimizers(self):
        """
        Setup optimizers for actor and critic networks.
        """
        self.optimizers = {
            "critic": torch.optim.Adam(list(self.nets["f"].parameters())+list(self.nets["T"].parameters())+list(self.nets["g"].parameters())+[self.raw_lagrange_multiplier], lr=self.optim_params.critic.learning_rate.initial),
            "actor": torch.optim.Adam(self.nets["pi"].parameters(), lr=self.optim_params.actor.learning_rate.initial),
        }


    def get_distance(self, z1, z2, i):
        return self.iqe(self.nets["g"][i](z1), self.nets["g"][i](z2))

    def predict(self, s, s_g):
        mu = self.nets["actor"](s, s_g)
        return torch.squeeze(mu).detach().cpu().numpy()
    
    def get_trans(self, z, a):
        return self.nets["T"](z, a) + z


    def critic_loss(self, st, st1, at):
        info = OrderedDict()

        
        zt = torch.cat([torch.unsqueeze(self.nets["f"][i](st), 0) for i in range(self.n_critics)], 0)
        zt1 = torch.cat([torch.unsqueeze(self.nets["f"][i](st1), 0) for i in range(self.n_critics)], 0)
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

        
        info["critic/l_pos"] = loss_positive
        info["critic/l_neg"] = loss_negative
        info["critic/loss_trans"] = loss_trans
        
        tot_loss = loss_positive + loss_negative + loss_trans

        return tot_loss, info

    def actor_loss(self, st, s_g, st1, at):
        info = OrderedDict()
        

        z = torch.cat([torch.unsqueeze(self.nets["f"][i](st), 0) for i in range(self.n_critics)], 0)
        z_g = torch.cat([torch.unsqueeze(self.nets["f"][i](s_g), 0) for i in range(self.n_critics)], 0)

        a = self.nets["pi"](st, s_g)
        neg_Qs = torch.cat([torch.unsqueeze(self.get_distance(self.get_trans(z[i].detach(), a), z_g[i].detach(), i), 0) for i in range(self.n_critics)], 0)
        actor_loss = neg_Qs.mean(-1).max()
        
        actor_loss_bc = self.w_pi * torch.mean((a - at) ** 2)
        tot_loss = actor_loss + actor_loss_bc

        
        # info["actor/adv"] = Adv
        # info["actor/log_prob"] = log_prob
        # info["actor/entropy"] = entropy
        info["actor/actor_loss"] = actor_loss
        info["actor/loss_bc"] = actor_loss_bc
        info["actor/tot_loss"] = tot_loss

        return tot_loss, info 



    def update(self, batch, info=None, epoch=None):
        
        # TODO: make sure this is correct
        st, gt, at, st1 = batch['obs'], batch['goal_obs'], batch['actions'], batch['next_obs']

        critic_loss, critic_info = self.critic_loss(st, st1, at)
        self.optimizers["critic"].zero_grad()
        critic_loss.backward()
        self.optimizers["critic"].step()

        if not self.pre_trained or epoch > self.pre_train_epochs:
            actor_loss, actor_info = self.actor_loss(st, gt, st1, at)
            self.optimizers["actor"].zero_grad()
            actor_loss.backward()
            self.optimizers["actor"].step()
        
            # Update info
            info.update(actor_info)
            
        info.update(critic_info)
        
        return info

    def train_on_batch(self, batch, epoch, validate=False):
        self.epoch = epoch
        info = OrderedDict()

        # Set the correct context for this training step
        with TorchUtils.maybe_no_grad(no_grad=validate):
            # Always run super call first
            info = super().train_on_batch(batch, epoch, validate=validate)
            info = self.update(batch, info, epoch)

        return info
    
    def train_critic(self, batch, epoch, validate=False):
        self.epoch = epoch
        info = OrderedDict()

        # Set the correct context for this training step
        with TorchUtils.maybe_no_grad(no_grad=validate):
            # Always run super call first
            info = super().train_on_batch(batch, epoch, validate=validate)
                    # TODO: make sure this is correct
            st, gt, at, st1 = batch['obs'], batch['goal_obs'], batch['actions'], batch['next_obs']

            critic_loss, critic_info = self.critic_loss(st, st1, gt)
            self.optimizers["critic"].zero_grad()
            critic_loss.backward()
            self.optimizers["critic"].step()
                
            info.update(critic_info)
        
        return info

    def process_batch_for_training(self, batch):
        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, 0, :] for k in batch["obs"]}
        input_batch["next_obs"] = {k: batch["next_obs"][k][:, 0, :] for k in batch["next_obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None)
        input_batch["actions"] = batch["actions"][:, 0, :]
        input_batch["dones"] = batch["dones"][:, 0]
        input_batch["rewards"] = batch["rewards"][:, 0]

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def get_action(self, obs_dict, goal_dict=None):
        assert not self.nets.training
        return self.nets["actor"](obs_dict=obs_dict, goal_dict=goal_dict)

    def log_info(self, info, only_critic=False):
        log = OrderedDict()
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = OrderedDict()
        
        if not only_critic:
            if not self.pre_trained or self.epoch > self.pre_train_epochs:
                log["actor/tot_loss"] = info["actor/tot_loss"].item()
                log["actor/actor_loss"] = info["actor/actor_loss"].item()
                log["actor/loss_bc"] = info["actor/loss_bc"].item()

                # self._log_data_attributes(log, info, "actor/log_prob")
                # self._log_data_attributes(log, info, "actor/entropy")
                # self._log_data_attributes(log, info, "actor/adv")
            
        
        log["critic/l_pos"] = info["critic/l_pos"].item()
        log["critic/l_neg"] = info["critic/l_neg"].item()
        log["critic/tot_loss"] = info["critic/loss_trans"].item()


        return log
    
    def _log_data_attributes(self, log, info, key):
        """
        Helper function for logging statistics. Moodifies log in-place

        Args:
            log (dict): existing log dictionary
            log (dict): existing dictionary of tensors containing raw stats
            key (str): key to log
        """
        log[key + "/max"] = info[key].max().item()
        log[key + "/min"] = info[key].min().item()
        log[key + "/mean"] = info[key].mean().item()
        log[key + "/std"] = info[key].std().item()

    def on_epoch_end(self, epoch):
        """
        Called at the end of each epoch.
        """
        pass

        # # LR scheduling updates
        # for lr_sc in self.lr_schedulers["critic"]:
        #     if lr_sc is not None:
        #         lr_sc.step()

        # if self.lr_schedulers["vf"] is not None:
        #     self.lr_schedulers["vf"].step()

        # if self.lr_schedulers["actor"] is not None:
        #     self.lr_schedulers["actor"].step()