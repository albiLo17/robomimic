"""
Implementation of Contrastive RL
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


@register_algo_factory_func("crl")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the MetricRL algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    return ContrastiveRL, {}

class ContrastiveRL(PolicyAlgo, ValueAlgo):
    def __init__(self, algo_config, obs_config, global_config, obs_key_shapes, ac_dim, device=None):
        super().__init__(algo_config, obs_config,  global_config, obs_key_shapes, ac_dim, device=device)
        self.pre_trained = self.algo_config.pre_train
        self.pre_train_epochs = self.algo_config.pre_train_epochs
        
        self.offline_reg = 0.05

    #     # Create the networks in the Robomimic style
    #     self._create_networks()

        # Setup optimizers
        self._setup_optimizers()

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
        self.nets["phi"] = ValueNets.MRLActionValueNetwork(
                    obs_shapes=self.obs_shapes,
                    ac_dim=self.ac_dim,
                    mlp_layer_dims=self.algo_config.critic.layer_dims,
                    goal_shapes=None,
                    output_shape=self.algo_config.phi_dim,
                    encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
                )
        self.nets["psi"] = ValueNets.MRLValueNetwork(
                    obs_shapes=self.obs_shapes,
                    mlp_layer_dims=self.algo_config.critic.layer_dims,
                    goal_shapes=None,
                    output_shape=self.algo_config.phi_dim,
                    encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
                )
        
        # Send networks to appropriate device
        self.nets = self.nets.float().to(self.device)

        
    def _create_optimizers(self):
        """
        Setup optimizers for actor and critic networks.
        """
        self.optimizers = {
            "critic": torch.optim.Adam(list(self.nets["psi"].parameters()) + list(self.nets["phi"].parameters()), lr=self.optim_params.critic.learning_rate.initial),
            "actor": torch.optim.Adam(self.nets["actor"].parameters(), lr=self.optim_params.actor.learning_rate.initial),
        }
        
    def _setup_optimizers(self):
        """
        Setup optimizers for actor and critic networks.
        """
        self.optimizers = {
            "critic": torch.optim.Adam(list(self.nets["psi"].parameters()) + list(self.nets["phi"].parameters()), lr=self.optim_params.critic.learning_rate.initial),
            "actor": torch.optim.Adam(self.nets["actor"].parameters(), lr=self.optim_params.actor.learning_rate.initial),
        }


    def get_distance(self, z1, z2):
        dist = torch.linalg.vector_norm(z1 - z2, ord=2, dim=-1)
        return dist

    def predict(self, s, s_g):
        mu = self.nets["actor"].get_mean(s, s_g)
        return torch.squeeze(mu).detach().cpu().numpy()

    def get_value(self, s, a, s_g):
        z_sa = self.nets["phi"](s, a)
        z_g = self.nets["psi"](s_g, s_g)
        logits = torch.einsum('ik, ik->i', z_sa, z_g)
        return logits

    def critic_loss(self, st, st1, at, gt):
        info = OrderedDict()
        
        z_sa = self.nets["phi"](st, at)
        z_g = self.nets["psi"](st1, st1)
        

        logits = torch.einsum('ik, jk->ij', z_sa, z_g)
        L_pos = nn.functional.binary_cross_entropy_with_logits(logits, torch.eye(logits.shape[0]).to(self.device))
        
        info["critic/l_pos"] = L_pos
        # info["critic/l_neg"] = L_neg
        # info["critic/tot_loss"] = L_neg + L_pos

        return L_pos, info

    def actor_loss(self, st, s_g, st1, at):
        info = OrderedDict()
        
        # sample action from actor
        a = self.nets["actor"](st, s_g)
        z_sa = self.nets["phi"](st, a)
        z_g = self.nets["psi"](s_g, s_g)
        logits = torch.einsum('ik, ik->i', z_sa, z_g)
        
        log_prob_a_orig, entropy = self.nets["actor"].get_log_prob(st, at, s_g)

        tot_loss = (1 - self.offline_reg) * torch.mean(-1.0 * logits) - self.offline_reg * torch.mean(log_prob_a_orig)

        
        # info["actor/adv"] = Adv
        # info["actor/log_prob"] = log_prob
        # info["actor/entropy"] = entropy
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

        st, gt, at, st1 = batch['obs'], batch['goal_obs'], batch['actions'], batch['next_obs']

        critic_loss, critic_info = self.critic_loss(st, st1, at, gt)
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

                # self._log_data_attributes(log, info, "actor/log_prob")
                # self._log_data_attributes(log, info, "actor/entropy")
                # self._log_data_attributes(log, info, "actor/adv")
            
        
        log["critic/l_pos"] = info["critic/l_pos"].item()
        # log["critic/l_neg"] = info["critic/l_neg"].item()
        # log["critic/tot_loss"] = info["critic/tot_loss"].item()


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