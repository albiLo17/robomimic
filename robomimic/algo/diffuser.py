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
from robomimic.algo.metric_minimal import MLP, Policy
from robomimic.algo.diffuser_minimal import DiffuserAlgo


@register_algo_factory_func("diffuser")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the diffuser algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    return Diffuser, {}

class Diffuser(PolicyAlgo, ValueAlgo):
    def __init__(self, algo_config, obs_config, global_config, obs_key_shapes, ac_dim, device=None):
        super().__init__(algo_config, obs_config,  global_config, obs_key_shapes, ac_dim, device=device)


    
    def _create_networks(self):
        self.input_dim = int(np.asarray([v for v in self.obs_shapes.values()]).sum())
        self.goal_dim = int(np.asarray([v for v in self.obs_shapes.values()]).sum())


        self.gamma = self.algo_config.gamma
        
        self.z_dim = self.algo_config.phi_dim

        self.T = self.algo_config.T
        self.H = self.algo_config.H 


        ##################################
        # Create nets
        self.nets = nn.ModuleDict()     


        # Actor
        self.nets["actor"] = DiffuserAlgo(input_dim=self.input_dim, goal_dim=self.goal_dim, 
                                          z_dim=self.z_dim, a_dim=self.ac_dim, norms=None, gamma=self.gamma, 
                                          H=self.H, T=self.T, device=self.device)

        # Send networks to appropriate device
        self.nets = self.nets.float().to(self.device)

        
    def _setup_optimizers(self):
        """
        Setup optimizers for actor and critic networks.
        """
        self.optimizers = {
            "actor": self.nets["actor"].opt,
        }


    def predict(self, s, s_g):
        return self.nets["actor"].predict(s, s_g)



    def update(self, batch, info=None, epoch=None):
        
        # TODO: make sure this is correct
        st, gt, at, st1 = batch['obs'], batch['goal_obs'], batch['actions'], batch['next_obs']
        
        diff_loss, reward_loss = None, None
        for i in range(st.shape[1] + 1 - self.H):
            
            # TODO: What about rewards???
            diff_loss, actor_info = self.nets["actor"].get_diffusion_loss(st[:,i:i+self.H-1], at[:,i:i+self.H-1], st1[:,i:i+self.H-1])
            reward_loss = self.nets["actor"].get_reward_loss(st[:,i:i+self.H-1], at[:,i:i+self.H-1], rt[:,i:i+self.H-1])
            tot_loss = diff_loss + reward_loss

            if self.gradients_iters == 0:
                self.optimizers["actor"].zero_grad()
            tot_loss.backward()
            self.gradients_iters += 1
            if self.gradients_iters == self.gradients_cumulations:
                self.optimizers["actor"].step()
                self.gradients_iters = 0

    
        # Update info
        # TODO: understand how to do this properly
        info.update(actor_info)
        
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
        

        log["actor/tot_loss"] = info["actor/tot_loss"].item()


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