"""
The main script for evaluating a policy in an environment.

Args:
    agent (str): path to saved checkpoint pth file

    horizon (int): if provided, override maximum horizon of rollout from the one 
        in the checkpoint

    env (str): if provided, override name of env from the one in the checkpoint,
        and use it for rollouts

    render (bool): if flag is provided, use on-screen rendering during rollouts

    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    camera_names (str or [str]): camera name(s) to use for rendering on-screen or to video

    dataset_path (str): if provided, an hdf5 file will be written at this path with the
        rollout data

    dataset_obs (bool): if flag is provided, and @dataset_path is provided, include 
        possible high-dimensional observations in output dataset hdf5 file (by default,
        observations are excluded and only simulator states are saved).

    seed (int): if provided, set seed for rollouts

Example usage:

    # Evaluate a policy with 50 rollouts of maximum horizon 400 and save the rollouts to a video.
    # Visualize the agentview and wrist cameras during the rollout.
    
    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --video_path /path/to/output.mp4 \
        --camera_names agentview robot0_eye_in_hand 

    # Write the 50 agent rollouts to a new dataset hdf5.

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5 --dataset_obs 

    # Write the 50 agent rollouts to a new dataset hdf5, but exclude the dataset observations
    # since they might be high-dimensional (they can be extracted again using the
    # dataset_states_to_obs.py script).

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5
"""
import argparse
import json
import h5py
import imageio
import numpy as np
from copy import deepcopy
import os 
import matplotlib.pyplot as plt
import glob

import torch
import multiprocessing

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.algo import RolloutPolicy

from concurrent.futures import ProcessPoolExecutor, as_completed

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu, theta=0.15, sigma=0.2, dt=1e-2, x0=None):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0 if x0 is not None else np.zeros_like(mu)
        self.reset()

    def reset(self):
        self.x_prev = self.x0

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
    
    
def random_rollout_brown(policy, actions, env, horizon, use_goals=False, render=False, video_writer=None, video_skip=5, return_obs=False, camera_names=None):
    """
    Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video, 
    and returns the rollout trajectory.

    Args:
        policy (instance of RolloutPolicy): policy loaded from a checkpoint]
        actions (np.array): actions to be executed during the rollout with noise
        env (instance of EnvBase): env loaded from a checkpoint or demonstration metadata
        horizon (int): maximum horizon for the rollout
        render (bool): whether to render rollout on-screen
        video_writer (imageio writer): if provided, use to write rollout to video
        video_skip (int): how often to write video frames
        return_obs (bool): if True, return possibly high-dimensional observations along the trajectoryu. 
            They are excluded by default because the low-dimensional simulation states should be a minimal 
            representation of the environment. 
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.

    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
        traj (dict): dictionary that corresponds to the rollout trajectory
    """
    assert isinstance(env, EnvBase)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))
    
    # Initialize Ornstein-Uhlenbeck noise generator for the actions
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros_like(actions[0]), theta=0.15, sigma=0.2, dt=1e-2)

    action_low = env.env.action_spec[0]
    action_high = env.env.action_spec[1]
    
    
    obs = env.reset()
    state_dict = env.get_state()

    # hack that is necessary for robosuite tasks for deterministic action playback
    obs = env.reset_to(state_dict)


    total_reward = 0.
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))
    try:
        for step_i in range(horizon):
            # sample random action in the action space
            # sample action between action_low and action_high
            act = actions[step_i]
            # Add Ornstein-Uhlenbeck noise to the action
            act += ou_noise.sample()
            
            act = np.clip(act, action_low, action_high)
            
            # play action
            next_obs, r, done, _ = env.step(act)

            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0])

            # collect transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            traj["states"].append(state_dict["states"])
            if return_obs:
                # Note: We need to "unprocess" the observations to prepare to write them to dataset.
                #       This includes operations like channel swapping and float to uint8 conversion
                #       for saving disk space.
                traj["obs"].append(ObsUtils.unprocess_obs_dict(obs))
                traj["next_obs"].append(ObsUtils.unprocess_obs_dict(next_obs))

            # break if done or if success
            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))

    if return_obs:
        # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return stats, traj


def rollout_worker(args, checkpoint, rollout_horizon, rollout_id, video_writer=None, ):
    """Worker function to run a single rollout, using CPU only."""
    try:
        # Set the device to CPU explicitly
        device = torch.device("cpu")

        # Load policy and environment inside the worker process to avoid pickling issues
        policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=checkpoint, device=device, verbose=False)
        
        # read rollout settings
        rollout_num_episodes = args.n_rollouts
        rollout_horizon = args.horizon
        if rollout_horizon is None:
            # read horizon from config
            config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
            rollout_horizon = config.experiment.rollout.horizon

        # create environment from saved checkpoint
        env, _ = FileUtils.env_from_checkpoint(
            ckpt_dict=ckpt_dict, 
            env_name=args.env, 
            render=args.render, 
            render_offscreen=(args.video_path is not None), 
            verbose=False,
        )
            
        video_writer = None
        
        # maybe open hdf5 to write rollouts
        write_dataset = (args.dataset_path is not None)
        if write_dataset:
            data_writer = h5py.File(args.dataset_path, "w")
            data_grp = data_writer.create_group("data")
            total_samples = 0

        print(f"Starting rollout {rollout_id} on CPU (Process: {os.getpid()})")
        
        # Perform the rollout
        stats, traj = random_rollout(
            policy=policy, 
            env=env, 
            horizon=rollout_horizon, 
            use_goals=config.use_goals,
            render=args.render, 
            video_writer=video_writer, 
            video_skip=args.video_skip, 
            return_obs=(write_dataset and args.dataset_obs),
            camera_names=args.camera_names,
        )
        
        if write_dataset:
            # store transitions
            ep_data_grp = data_grp.create_group("demo_{}".format(rollout_id))
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            if args.dataset_obs:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            # episode metadata
            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
            total_samples += traj["actions"].shape[0]
        
        print(f"Finished rollout {rollout_id} on CPU (Process: {os.getpid()})")
        return stats

    except Exception as e:
        # create a txt file to log the errors with naming the time and algo name
        algo_name = policy.policy.global_config.experiment.name
        txt_file = f"{policy.policy.global_config.train.output_dir}/error_logs_{algo_name}.txt"
        with open(txt_file, 'a') as f:
            f.write(f"Error in rollout {rollout_id}: {e}\n  ")
            
        print(f"Error in rollout {rollout_id}: {e}")
        return None


def run_data_collection(args, max_parallel_rollouts=1, write_video=None, ):
    """Run the trained agent with parallel rollouts, capped by `max_parallel_rollouts`, using CPU only."""
    rollout_num_episodes = args.n_rollouts
    
    ####################### TO BE PARALLELIIZED #######################
        # relative path to agent
    ckpt_path = args.agent_path

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

    # read rollout settings
    rollout_num_episodes = args.n_rollouts
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)

    # create environment from saved checkpoint
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, 
        env_name=args.env, 
        render=args.render, 
        render_offscreen=(args.video_path is not None), 
        verbose=True,
    )

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    ########################################################################
    
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)
    
    # maybe open hdf5 to write rollouts
    write_dataset = (args.dataset_path is not None)
    if write_dataset:
        os.makedirs(os.path.dirname(args.dataset_path), exist_ok=True)
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0
        
    rollout_stats = []
    
    # load demo dataset to load predefined actions
    dataset_path = args.demo_path
    f = h5py.File(dataset_path, "r")
    demos = list(f["data"].keys())
    actions = [f["data/{}/actions".format(ep)] for ep in demos]


    # Iterate through the epochs for agent evaluation
    for i in range(rollout_num_episodes):
        print(f"Rolling out agent at epoch {i}")

        # # Run rollouts in parallel with a limited number of workers
        # with ProcessPoolExecutor(max_workers=max_parallel_rollouts) as executor:
        #     futures = []
        #     for rollout_id in range(rollout_num_episodes):
        #         futures.append(executor.submit(rollout_worker, args, args.agent, args.horizon, rollout_id))

        #     for future in as_completed(futures):
        #         stats = future.result()  # Collect rollout stats
        #         if stats is not None:
        #             rollout_stats.append(stats)
        
        action_id = i % len(actions)
        rollout_horizon = len(actions[action_id])
        # Perform the rollout
        stats, traj = random_rollout_brown(
            policy=policy, 
            actions=actions[action_id],
            env=env, 
            horizon=rollout_horizon, 
            use_goals=config.use_goals,
            render=args.render, 
            video_writer=video_writer, 
            video_skip=args.video_skip, 
            return_obs=(write_dataset and args.dataset_obs),
            camera_names=args.camera_names,
        )
        
        rollout_stats.append(stats)
        
        if write_dataset:
            # store transitions
            ep_data_grp = data_grp.create_group("demo_{}".format(i))
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            if args.dataset_obs:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            # episode metadata
            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
            total_samples += traj["actions"].shape[0]
                   


    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = { k : np.mean(rollout_stats[k]) for k in rollout_stats }
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
    print("Average Rollout Stats")
    # print(json.dumps(avg_rollout_stats, indent=4))
    
    
    if write_dataset:
        # global metadata
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
        data_writer.close()
        print("Wrote dataset trajectories to {}".format(args.dataset_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ########## TO REMEMBER ##########
    # you need to have the correct environment namings!!!

    parser.add_argument(
        "--agent_path",
        default= "cluster_output/mrl_trained_models_ds/mrl_can_ph/20240913173309/models/model_epoch_2000.pth",
        type=str,
        help="path to saved checkpoint pth file",
    )
    
    # number of rollouts
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=1000,
        help="number of rollouts",
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="(optional) override maximum horizon of rollout from the one in the checkpoint",
    )

    # Env Name (to override the one stored in model checkpoint)
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="(optional) override name of env from the one in the checkpoint, and use\
            it for rollouts",
    )
        # Whether to render rollouts to screen
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    # Dump a video of the rollouts to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render rollouts to this video file path",
    )

    # How often to write video frames during the rollout
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=["agentview"],
        help="(optional) camera name(s) to use for rendering on-screen or to video",
    )

    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./datasets/can/random_brown/low_dim_sparse_v141.hdf5",
        help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
    )
    
    parser.add_argument(
        "--demo_path",
        type=str,
        default="./datasets/can/ph/low_dim_v141.hdf5",
        help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
    )



    # for seeding before starting rollouts
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) set seed for rollouts",
    )

    parser.add_argument(
        "--dataset_obs",
        type=bool,
        default=True,
        help="(optional) if flag is provided, include possible high-dimensional observations in output dataset hdf5 file",
    )
    args = parser.parse_args()
    run_data_collection(args)

