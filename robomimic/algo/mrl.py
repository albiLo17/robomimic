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


@register_algo_factory_func("mrl")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the MetricRL algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    return MetricRL, {}

class MetricRL(PolicyAlgo, ValueAlgo):
    def __init__(self, algo_config, obs_config, global_config, obs_key_shapes, ac_dim, device=None):
        super().__init__(algo_config, obs_config,  global_config, obs_key_shapes, ac_dim, device=device)
        self.pre_trained = self.algo_config.pre_train
        self.pre_train_epochs = self.algo_config.pre_train_epochs

    #     # Create the networks in the Robomimic style
    #     self._create_networks()

    #     # Setup optimizers
    #     self._setup_optimizers()

    # def _create_networks(self):
    #     """
    #     Creates networks and places them into @self.nets.
    #     Networks for this algo: actor, critic (as phi network in this case)
    #     """
    #     self.nets = nn.ModuleDict()

    #     # MLP for encoding state to latent space (phi network) as critic
    #     self.nets["critic"] = MLP(
    #         input_dim=self.obs_shapes["obs"]["shape"][0],
    #         output_dim=self.algo_config.phi_dim,
    #         hidden_units=64
    #     )

    #     # Policy network
    #     policy_def = {
    #         'type': 'continuous',  # 'discrete' or 'continuous'
    #         'var': 1.0,
    #         'max_action': 1.0,  # max value action
    #         'bias_action': 0.0  # middle value action
    #     }
    #     self.nets["actor"] = Policy(
    #         input_dim=self.obs_shapes["obs"]["shape"][0],
    #         cond_dim=self.goal_shapes["goal"]["shape"][0],
    #         output_dim=self.ac_dim,
    #         policy_def=policy_def
    #     )

    #     # Send networks to the correct device
    #     self.nets = self.nets.float().to(self.device)
    
    def _create_networks(self):
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
        self.nets["actor"] = actor_cls(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor.layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **actor_args,
        )

        # Critics
        # TODO: evnetually support ensamble critics
        # input_dim = np.asarray([v for v in self.obs_shapes.values()]).sum()
        # self.nets["critic"] = MLP(input_dim, self.algo_config.phi_dim)
        self.nets["critic"] = ValueNets.MRLValueNetwork(
                    obs_shapes=self.obs_shapes,
                    mlp_layer_dims=self.algo_config.critic.layer_dims,
                    goal_shapes=None,
                    output_shape=self.algo_config.phi_dim,
                    encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
                )
        
        # Send networks to appropriate device
        self.nets = self.nets.float().to(self.device)

        
    def _setup_optimizers(self):
        """
        Setup optimizers for actor and critic networks.
        """
        self.optimizers = {
            "critic": torch.optim.Adam(self.nets["critic"].parameters(), lr=self.algo_config.critic.lr),
            "actor": torch.optim.Adam(self.nets["actor"].parameters(), lr=self.algo_config.actor.lr),
        }

    def get_distance(self, z1, z2):
        dist = torch.linalg.vector_norm(z1 - z2, ord=2, dim=-1)
        return dist

    def predict(self, s, s_g):
        mu = self.nets["actor"].get_mean(s, s_g)
        return torch.squeeze(mu).detach().cpu().numpy()

    def get_value(self, s, s_g):
        z = self.nets["critic"](s, s_g)
        z_g = self.nets["critic"](s_g, s_g)
        neg_dist = -self.get_distance(z, z_g)
        return neg_dist

    def critic_loss(self, st, st1, gt):
        info = OrderedDict()
        
        z = self.nets["critic"](st, gt)
        z1 = self.nets["critic"](st1, gt)
        
        # check if any of the elements st1 (so second dimension skipping the batch) is a vector of all zeros
        # if torch.any(torch.all(torch.eq(st1["robot0_eef_pos"], 0), dim=1)):
        #     print(" ****************** Zero vector found ******************")

        action_distance = 1
        L_pos = torch.mean((self.get_distance(z1, z) - action_distance) ** 2)

        # ### VERSION 2 - 4 - F
        idx = torch.randperm(z.shape[0])
        z1_shuffle = z1[idx]
        dist_z_perm = -torch.log(self.get_distance(z, z1_shuffle) + 1e-6)
        L_neg = torch.mean(dist_z_perm)
        
        ### VERSION 3 - 5
        # idx = np.zeros((z.shape[0], z.shape[0] - 1))
        # for i in range(z.shape[0]):
        #     idx[i] = np.delete(np.arange(z.shape[0]), i)
        # z1_rep = torch.cat([torch.unsqueeze(z1[i], 0) for i in idx], 0)
        # dist_z_perm = - torch.log(torch.cdist(torch.unsqueeze(z, 1), z1_rep) + 1e-6)
        # L_neg = torch.mean(dist_z_perm)
        
        info["critic/l_pos"] = L_pos
        info["critic/l_neg"] = L_neg
        info["critic/tot_loss"] = L_neg + L_pos

        return L_pos + L_neg, info

    def actor_loss(self, st, s_g, st1, at):
        info = OrderedDict()
        
        log_prob, entropy = self.nets["actor"].get_log_prob(st, at, s_g)
        Vt = self.get_value(st, s_g)
        Vt1 = self.get_value(st1, s_g)
        Adv = (Vt1 - Vt)
        
        # Version 1 and F
        # tot_loss = -torch.mean(Adv.detach()* log_prob)
        
        # compute weights # Versiosn 2,3,4,5
        weights = self._get_adv_weights(Adv.reshape(-1, 1))
        tot_loss = -torch.mean(weights.detach() * log_prob)
        
        info["actor/adv"] = Adv
        info["actor/log_prob"] = log_prob
        info["actor/entropy"] = entropy
        info["actor/tot_loss"] = tot_loss

        return tot_loss, info 
    
    def _get_adv_weights(self, adv):
        """
        Helper function for computing advantage weights. Called by @_compute_actor_loss

        Args:
            adv (torch.Tensor): raw advantage estimates

        Returns:
            weights (torch.Tensor): weights computed based on advantage estimates,
                in shape (B,) where B is batch size
        """
        
        # clip raw advantage values
        if self.algo_config.adv.clip_adv_value is not None:
            adv = adv.clamp(max=self.algo_config.adv.clip_adv_value)

        # compute weights based on advantage values
        beta = self.algo_config.adv.beta # temprature factor        
        weights = torch.exp(adv / beta)

        # clip final weights
        if self.algo_config.adv.use_final_clip is True:
            weights = weights.clamp(-100.0, 100.0)

        # reshape from (B, 1) to (B,)
        return weights[:, 0]


    def update(self, batch, info=None, epoch=None):
        
        # TODO: make sure this is correct
        st, gt, at, st1 = batch['obs'], batch['goal_obs'], batch['actions'], batch['next_obs']

        critic_loss, critic_info = self.critic_loss(st, st1, gt)
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

                self._log_data_attributes(log, info, "actor/log_prob")
                self._log_data_attributes(log, info, "actor/entropy")
                self._log_data_attributes(log, info, "actor/adv")
            
        
        log["critic/l_pos"] = info["critic/l_pos"].item()
        log["critic/l_neg"] = info["critic/l_neg"].item()
        log["critic/tot_loss"] = info["critic/tot_loss"].item()


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