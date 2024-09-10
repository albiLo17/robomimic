"""
The main entry point for training policies.

Args:
    config (str): path to a config json that will be used to override the default settings.
        If omitted, default settings are used. This is the preferred way to run experiments.

    algo (str): name of the algorithm to run. Only needs to be provided if @config is not
        provided.

    name (str): if provided, override the experiment name defined in the config

    dataset (str): if provided, override the dataset path defined in the config

    debug (bool): set this flag to run a quick training run for debugging purposes    
"""

import argparse
import json
import numpy as np
import time
import os
import shutil
import psutil
import sys
import socket
import traceback
import wandb
import robomimic.utils.log_utils as LogUtils
from tqdm import tqdm


from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

import robomimic
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings
import matplotlib.pyplot as plt
import h5py

def plot_representation(model, train_data, device, input_obs_dict, demo_index=0, file_name=None):
    model.set_eval()
    f = train_data.hdf5_file
    demos = list(f["data"].keys())
    # print(demos)
    demos.sort(key=lambda x: int(x.split('_')[1]))
    demo_key = demos[demo_index]
    # print(demo_key)
    demo_grp = f["data"][demo_key]

    # Retrieve the observations and actions
 # Get all timesteps for this modality
    obs = {}
    for k in input_obs_dict:
        obs[k] = torch.tensor(demo_grp["obs/{}".format(k)][:],dtype=torch.float32).to(device)  # Get all timesteps for this modality
    g_t = {k: obs[k][-1].unsqueeze(0) for k in input_obs_dict}

    with torch.no_grad():
        z = model.nets["critic"](obs, g_t)
        z_g = model.nets["critic"]( g_t, g_t)

        distances = model.get_distance(z, z_g).cpu().numpy()
    # Get predicted actions by the model
    # for t in range(len(ground_truth_actions)):
    #     # Prepare the observation for the model
    #     obs_t = {k: torch.tensor(obs[k][t],dtype=torch.float32).unsqueeze(0).to(device) for k in input_obs_dict}
    #     # print(obs_t["robot0_eef_pos"].shape)
    #     # print(g_t["robot0_eef_pos"].shape)
    #     # Model forward pass
    #     with torch.no_grad():
    #         z = model.nets["critic"](obs_t, g_t)
    #         z_g = model.nets["critic"]( g_t, g_t)
    #         distances.append(distance.cpu().numpy())    
            
    # plot distances from the goal of the representation
    plt.figure()
    distances = np.array(distances)
    plt.figure(figsize=(10, 6))
    plt.plot(distances, label='Distance between representation and goal')
    plt.xlabel('Timestep')
    plt.ylabel('Distance')
    plt.title(f'Distance for each timestep in trajectory {demo_index}')
    plt.legend()
    
    if file_name:
        plt.savefig(file_name)
        plt.close()  # Close the plot to avoid display during training
    else:
        plt.show()
        
    if z.shape[1] == 2:
        plt.figure()
        z = z.cpu().numpy()
        z_g = z_g.cpu().numpy()
        plt.figure()
        plt.scatter(z[:, 0], z[:, 1], c=range(z.shape[0]), cmap='viridis')
        plt.scatter(z_g[:, 0], z_g[:, 1], c='red', marker='x', label='Goal')
        # plt.colorbar()
        plt.title(f'Representation for trajectory {demo_index}')
        if file_name:
            plt.savefig(file_name.replace('/epoch', '/representation_epoch'))
            plt.close()
            
        # select other trajs to plot
        plt.figure()
        for i in range(1, 7):
            random_idx = np.random.randint(0, len(demos))
            # print(f"Random index: {random_idx}")
            demo_grp = f["data"][demos[random_idx]]
            obs = {}
            for k in input_obs_dict:
                obs[k] = torch.tensor(demo_grp["obs/{}".format(k)][:],dtype=torch.float32).to(device)  # Get all timesteps for this modality
            next_obs = {}
            for k in input_obs_dict:
                next_obs[k] = torch.tensor(demo_grp["next_obs/{}".format(k)][:],dtype=torch.float32).to(device) 
                
            anchor = {k: next_obs[k][-1].unsqueeze(0) for k in input_obs_dict}
            with torch.no_grad():
                z = model.nets["critic"](obs, anchor)
                z_g = model.nets["critic"]( anchor, anchor)
            
            z = z.cpu().numpy()
            z_g = z_g.cpu().numpy()
            plt.scatter(z[:, 0], z[:, 1], c=range(z.shape[0]), cmap='viridis')
            plt.scatter(z_g[:, 0], z_g[:, 1], c='red', marker='x', label='Goal')
            
        if file_name:
            plt.savefig(file_name.replace('/epoch', '/anchor_state_epoch'))
            plt.close()
            
    model.set_train()
                

def train(config, device):
    """
    Train a model using the algorithm.
    """

    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    torch.set_num_threads(2)

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")
    log_dir, ckpt_dir, video_dir = TrainUtils.get_exp_dir(config)
    # get the parent of checkpoint directory
    base_output_dir = os.path.dirname(ckpt_dir)
    fig_dir = os.path.join(base_output_dir, "images")
    if os.path.exists(fig_dir):
        shutil.rmtree(fig_dir)
    os.makedirs(fig_dir)

    # if config.experiment.logging.terminal_output_to_txt:
    #     # log stdout and stderr to a text file
    #     logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
    #     sys.stdout = logger
    #     sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # make sure the dataset exists
    dataset_path = os.path.expanduser(config.train.data)
    if not os.path.exists(dataset_path):
        raise Exception("Dataset at provided path {} not found!".format(dataset_path))

    # load basic metadata from training file
    print("\n============= Loaded Environment Metadata =============")
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=config.train.data)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data,
        all_obs_keys=config.all_obs_keys,
        verbose=True
    )

    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    # create environment
    envs = OrderedDict()
    if config.experiment.rollout.enabled:
        # create environments for validation runs
        env_names = [env_meta["env_name"]]

        if config.experiment.additional_envs is not None:
            for name in config.experiment.additional_envs:
                env_names.append(name)

        for env_name in env_names:
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                env_name=env_name, 
                render=False, 
                render_offscreen=config.experiment.render_video,
                use_image_obs=shape_meta["use_images"],
                use_depth_obs=shape_meta["use_depths"],
            )
            env = EnvUtils.wrap_env_from_config(env, config=config) # apply environment warpper, if applicable
            envs[env.name] = env
            print(envs[env.name])

    print("")

    # setup for a new training run
    data_logger = DataLogger(
        log_dir,
        config,
        log_tb=config.experiment.logging.log_tb,
        log_wandb=config.experiment.logging.log_wandb,
    )
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )
    
    # save the config as a json file
    with open(os.path.join(log_dir, '..', 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)

    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")

    # load training data
    trainset, validset = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"])
    train_sampler = trainset.get_dataset_sampler()
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")
    if validset is not None:
        print("\n============= Validation Dataset =============")
        print(validset)
        print("")

    # maybe retreve statistics for normalizing observations
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    # initialize data loaders
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True
    )

    if config.experiment.validate:
        # cap num workers for validation dataset at 1
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            drop_last=True
        )
    else:
        valid_loader = None

    # print all warnings before training begins
    print("*" * 50)
    print("Warnings generated by robomimic have been duplicated here (from above) for convenience. Please check them carefully.")
    flush_warnings()
    print("*" * 50)
    print("")

    # main training loop
    best_valid_loss = None
    best_return = {k: -np.inf for k in envs} if config.experiment.rollout.enabled else None
    best_success_rate = {k: -1. for k in envs} if config.experiment.rollout.enabled else None
    last_ckpt_time = time.time()

    # number of learning steps per epoch (defaults to a full dataset pass)
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps
    
    model.set_train()
    num_steps = len(train_loader)
    step_log_all = []
    timing_stats = dict(Data_Loading=[], Process_Batch=[], Train_Batch=[], Log_Info=[])
    
    
    progress_bar = tqdm(range(1, config.train.num_epochs + 1), desc="Training", leave=True)
    
    for epoch in progress_bar: # epoch numbers start at 1

        data_loader_iter = iter(train_loader)
        # for _ in LogUtils.custom_tqdm(range(num_steps)):        
        for _ in range(num_steps):
            batch = next(data_loader_iter)
            input_batch = model.process_batch_for_training(batch)
            input_batch = model.postprocess_batch_for_training(input_batch, obs_normalization_stats=obs_normalization_stats)
            # TODO: train only the critic
            info = model.train_critic(input_batch, epoch, validate=False)
            step_log_model = model.log_info(info, only_critic=True)
            step_log_all.append(step_log_model)
            
        # flatten and take the mean of the metrics
        step_log_dict = {}
        for i in range(len(step_log_all)):
            for k in step_log_all[i]:
                if k not in step_log_dict:
                    step_log_dict[k] = []
                step_log_dict[k].append(step_log_all[i][k])
        step_log = dict((k, float(np.mean(v))) for k, v in step_log_dict.items())

        ##############################
        model.on_epoch_end(epoch)

        # setup checkpoint path
        epoch_ckpt_name = "model_epoch_{}".format(epoch)

        # check for recurring checkpoint saving conditions
        should_save_ckpt = False
        if epoch % config.experiment.save.every_n_epochs == 0:
            should_save_ckpt = True
            ckpt_reason = "epochs"
            
        if epoch % 10 == 0:
            plot_filename = fig_dir + f"/epoch_{epoch}.png"
            # select random demo index
            demo_idx = np.random.randint(0, len(trainset.demos))
            # demo_idx = 167
            plot_representation(model, trainset, device, input_obs_dict=config.observation.modalities.obs.low_dim, demo_index=demo_idx, file_name=plot_filename)
            # data_logger._wandb_logger.log({f"Reprsentation": wandb.Image(plot_filename)})
        for k, v in step_log.items():
            data_logger.record("Train/{}".format(k), v, epoch)
                
        # Update the progress bar with loss values
        progress_bar.set_postfix({
            'L Neg': step_log["critic/l_neg"],
            'L Pos': step_log["critic/l_pos"]
        })


        # Save model checkpoints based on conditions (success rate, validation loss, etc)
        if should_save_ckpt:
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=obs_normalization_stats,
            )


    # terminate logging
    data_logger.close()


def main(args):

    if args.config is not None:
        ext_cfg = json.load(open(args.config, 'r'))
        config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        config = config_factory(args.algo)

    config.algo.optim_params.critic.learning_rate.initial = args.lr
    config.algo.phi_dim = args.phi_dim
    config.train.batch_size = args.batch_size
    config.train.num_epochs = args.epochs
    config.experiment.name = "Representation_lift_mg"
    config.experiment.name = config.experiment.name + f"_rep_z_{args.phi_dim}_lr_{args.lr}_bs_{args.batch_size}_epochs_{args.epochs}"
    
    if args.dataset is not None:
        config.train.data = args.dataset

    if args.name is not None:
        config.experiment.name = args.name

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)
    print("Using device: {}".format(device))

    # maybe modify config for debugging purposes
    if args.debug:
        # shrink length of training to test whether this run is likely to crash
        config.unlock()
        config.lock_keys()

        # train and validate (if enabled) for 3 gradient steps, for 2 epochs
        config.experiment.epoch_every_n_steps = 3
        config.experiment.validation_epoch_every_n_steps = 3
        config.train.num_epochs = 2

        # if rollouts are enabled, try 2 rollouts at end of each epoch, with 10 environment steps
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10

        # send output to a temporary directory
        config.train.output_dir = "/tmp/tmp_trained_models"

    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    try:
        train(config, device=device)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )

    # Algorithm Name
    parser.add_argument(
        "--algo",
        type=str,
        default="mrl",
        help="(optional) name of algorithm to run. Only needs to be provided if --config is not provided",
    )

    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset",
        type=str,
        # default="/home/omniverse/workspace/robomimic/datasets/lift/ph/low_dim_v141_augmented.hdf5",
        default="/home/omniverse/workspace/robomimic/datasets/lift/mg/low_dim_sparse_v141_augmented.hdf5",
        help="(optional) if provided, override the dataset path defined in the config",
    )

    # debug mode
    parser.add_argument(
        "--debug",
        action='store_true',
        help="set this flag to run a quick training run for debugging purposes"
    )
    
    
    parser.add_argument("--lr", type=float, default=1e-3,  help="learning rate",)
    parser.add_argument("--phi_dim", type=float, default=64,  help="representation dimension",)
    parser.add_argument("--batch_size", type=float, default=1024,  help="batch size",)
    parser.add_argument("--epochs", type=float, default=5000,  help="Numebr of training epochs",)

    args = parser.parse_args()
    main(args)

