"""
Config for MetricRL algorithm.
"""

from robomimic.config.base_config import BaseConfig
from robomimic.config.config import Config
from copy import deepcopy

class ContrastiveRLConfig(BaseConfig):
    ALGO_NAME = "crl"

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the 
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config` 
        argument to the constructor. Any parameter that an algorithm needs to determine its 
        training and test-time behavior should be populated here.
        """
        super(ContrastiveRLConfig, self).algo_config()
        
       
        # ================== Custom Config ===================
        self.algo.phi_dim = 64                                        # dimension of phi network output
        self.algo.pre_train = False                                    # whether to pre-train phi network    
        self.algo.pre_train_epochs = 500                              # number of pre-training epochs
        
        # optimization parameters        
        self.algo.optim_params.critic.learning_rate.initial = 1e-4          # critic learning rate
        self.algo.optim_params.critic.learning_rate.decay_factor = 0.0      # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.critic.learning_rate.epoch_schedule = []     # epochs where LR decay occurs
        self.algo.optim_params.critic.regularization.L2 = 0.00              # L2 regularization strength

        self.algo.optim_params.vf.learning_rate.initial = 1e-4              # vf learning rate
        self.algo.optim_params.vf.learning_rate.decay_factor = 0.0          # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.vf.learning_rate.epoch_schedule = []         # epochs where LR decay occurs
        self.algo.optim_params.vf.regularization.L2 = 0.00                  # L2 regularization strength

        self.algo.optim_params.actor.learning_rate.initial = 1e-4           # actor learning rate
        self.algo.optim_params.actor.learning_rate.decay_factor = 0.0       # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.actor.learning_rate.epoch_schedule = []      # epochs where LR decay occurs
        self.algo.optim_params.actor.regularization.L2 = 0.00               # L2 regularization strength

        # target network related parameters
        self.algo.discount = 0.99                                           # discount factor to use
        self.algo.target_tau = 0.01                                         # update rate for target networks

        # ================== Actor Network Config ===================
        # Actor network settings
        self.algo.actor.net.type = "gaussian"                               # Options are currently ["gaussian", "gmm"]

        # Actor network settings - shared
        self.algo.actor.net.common.std_activation = "softplus"              # Activation to use for std output from policy net
        self.algo.actor.net.common.low_noise_eval = True                    # Whether to use deterministic action sampling at eval stage
        self.algo.actor.net.common.use_tanh = False                         # Whether to use tanh at output of actor network

        # Actor network settings - gaussian
        self.algo.actor.net.gaussian.init_last_fc_weight = 0.001            # If set, will override the initialization of the final fc layer to be uniformly sampled limited by this value
        self.algo.actor.net.gaussian.init_std = 0.3                         # Relative scaling factor for std from policy net
        self.algo.actor.net.gaussian.fixed_std = False                      # Whether to learn std dev or not

        self.algo.actor.net.gmm.num_modes = 5                               # number of GMM modes
        self.algo.actor.net.gmm.min_std = 0.0001                            # minimum std output from network

        self.algo.actor.layer_dims = (300, 400)                             # actor MLP layer dimensions

        self.algo.actor.max_gradient_norm = None                            # L2 gradient clipping for actor

        # ================== Critic Network Config ===================
        # critic ensemble parameters
        self.algo.critic.ensemble.n = 1                                     # number of Q networks in the ensemble
        self.algo.critic.layer_dims = (300, 400)                            # critic MLP layer dimensions
        self.algo.critic.use_huber = False                                  # Huber Loss instead of L2 for critic
        self.algo.critic.max_gradient_norm = None                           # L2 gradient clipping for actor

        # ================== Adv Config ==============================
        self.algo.adv.clip_adv_value = None                                 # whether to clip raw advantage estimates
        self.algo.adv.beta = 1.0                                            # temperature for operator
        self.algo.adv.use_final_clip = True                                 # whether to clip final weight calculations

        self.algo.vf_quantile = 0.9                                         # quantile factor in quantile regression


        
    def experiment_config(self):
        """
        Update from subclass to set paper defaults for gym envs.
        """
        super(ContrastiveRLConfig, self).experiment_config()
        self.experiment.name = "MetricRL_2"                               # added weights
        # self.experiment.name = "MetricRL_3"                               # new loss
        # self.experiment.name = "MetricRL_4"                               # old loss, new critic architecture
        # self.experiment.name = "MetricRL_5"                               # new loss, new critic architecture
        # self.experiment.name = "MetricRL_F"                                 # No weights or loss changes, Gaussian policy

        # # no validation and no video rendering
        # self.experiment.validate = False
        # self.experiment.render_video = False

        # # evaluate with normal environment rollouts
        self.experiment.rollout.enabled = False
        # self.experiment.rollout.n = 10              # paper uses 10, but we can afford to do 50
        # self.experiment.rollout.horizon = 100
        # self.experiment.rollout.rate = 10            # rollout every epoch to match paper

    def train_config(self):
        """
        Update from subclass to set paper defaults for gym envs.
        """
        super(ContrastiveRLConfig, self).train_config()

        # update to normalize observations
        # self.train.hdf5_normalize_obs = True 
        self.train.goal_mode = "last"
        

        # 200 epochs, with each epoch lasting 5000 gradient steps, for 1M total steps
        # self.train.num_epochs = 200
        
        # increase batch size to 1024 (found to work better for most manipulation experiments)
        self.train.batch_size = 1024
        
    def observation_config(self):
        """
        This function populates the `config.observation` attribute of the config, and is given 
        to the `Algo` subclass (see `algo/algo.py`) for each algorithm through the `obs_config` 
        argument to the constructor. This portion of the config is used to specify what 
        observation modalities should be used by the networks for training, and how the 
        observation modalities should be encoded by the networks. While this class has a 
        default implementation that usually doesn't need to be overriden, certain algorithm 
        configs may choose to, in order to have seperate configs for different networks 
        in the algorithm. 
        """

        # observation modalities
        self.observation.modalities.obs.low_dim = [             # specify low-dim observations for agent
            "robot0_eef_pos", 
            "robot0_eef_quat", 
            "robot0_gripper_qpos", 
            "object",
        ]
        self.observation.modalities.obs.rgb = []              # specify rgb image observations for agent
        self.observation.modalities.obs.depth = []
        self.observation.modalities.obs.scan = []
        self.observation.modalities.goal.low_dim = []           # specify low-dim goal observations to condition agent on
        self.observation.modalities.goal.rgb = []             # specify rgb image goal observations to condition agent on
        self.observation.modalities.goal.depth = []
        self.observation.modalities.goal.scan = []
        self.observation.modalities.obs.do_not_lock_keys()
        self.observation.modalities.goal.do_not_lock_keys()

        # observation encoder architectures (per obs modality)
        # This applies to all networks that take observation dicts as input

        # =============== Low Dim default encoder (no encoder) ===============
        self.observation.encoder.low_dim.core_class = None
        self.observation.encoder.low_dim.core_kwargs = Config()                 # No kwargs by default
        self.observation.encoder.low_dim.core_kwargs.do_not_lock_keys()

        # Low Dim: Obs Randomizer settings
        self.observation.encoder.low_dim.obs_randomizer_class = None
        self.observation.encoder.low_dim.obs_randomizer_kwargs = Config()       # No kwargs by default
        self.observation.encoder.low_dim.obs_randomizer_kwargs.do_not_lock_keys()

        # =============== RGB default encoder (ResNet backbone + linear layer output) ===============
        self.observation.encoder.rgb.core_class = "VisualCore"                  # Default VisualCore class combines backbone (like ResNet-18) with pooling operation (like spatial softmax)
        self.observation.encoder.rgb.core_kwargs = Config()                     # See models/obs_core.py for important kwargs to set and defaults used
        self.observation.encoder.rgb.core_kwargs.do_not_lock_keys()

        # RGB: Obs Randomizer settings
        self.observation.encoder.rgb.obs_randomizer_class = None                # Can set to 'CropRandomizer' to use crop randomization
        self.observation.encoder.rgb.obs_randomizer_kwargs = Config()           # See models/obs_core.py for important kwargs to set and defaults used
        self.observation.encoder.rgb.obs_randomizer_kwargs.do_not_lock_keys()

        # Allow for other custom modalities to be specified
        self.observation.encoder.do_not_lock_keys()

        # =============== Depth default encoder (same as rgb) ===============
        self.observation.encoder.depth = deepcopy(self.observation.encoder.rgb)

        # =============== Scan default encoder (Conv1d backbone + linear layer output) ===============
        self.observation.encoder.scan = deepcopy(self.observation.encoder.rgb)

        # Scan: Modify the core class + kwargs, otherwise, is same as rgb encoder
        self.observation.encoder.scan.core_class = "ScanCore"                   # Default ScanCore class uses Conv1D to process this modality
        self.observation.encoder.scan.core_kwargs = Config()                    # See models/obs_core.py for important kwargs to set and defaults used
        self.observation.encoder.scan.core_kwargs.do_not_lock_keys()
