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

from albi.evaluate_model_different_stages import rollout

from concurrent.futures import ProcessPoolExecutor, as_completed


def rollout_worker(args, checkpoint, rollout_horizon, rollout_id):
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
            
        # Create the environment from the checkpoint
        env, _ = FileUtils.env_from_checkpoint(
            ckpt_dict=ckpt_dict, 
            env_name=args.env, 
            render=args.render, 
            render_offscreen=(args.video_path is not None), 
            verbose=False,
        )
        video_writer = None

        print(f"Starting rollout {rollout_id} on CPU (Process: {os.getpid()})")
        
        # Perform the rollout
        stats, traj = rollout(
            policy=policy, 
            env=env, 
            horizon=rollout_horizon, 
            use_goals=config.use_goals,
            render=args.render, 
            video_writer=video_writer, 
            video_skip=args.video_skip, 
            return_obs=False,
            camera_names=args.camera_names,
        )
        
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


def run_trained_agent(args, max_parallel_rollouts=4):
    """Run the trained agent with parallel rollouts, capped by `max_parallel_rollouts`, using CPU only."""
    rollout_num_episodes = args.n_rollouts
    trained_epochs = 2000
    step = 50
    success_rates = []
    
    max_success_rate = 0.0

    # Iterate through the epochs for agent evaluation
    for i in range(50, trained_epochs + 1, step):
        print(f"Rolling out agent at epoch {i}")

        # Find the correct checkpoint path for the current epoch
        args.agent = glob.glob(args.agent_path + f"/model_epoch_{i}*")
        args.agent.sort()
        args.agent = args.agent[0]

        episode_success = []
        
        if max_success_rate < 1.0:
            # Run rollouts in parallel with a limited number of workers
            with ProcessPoolExecutor(max_workers=max_parallel_rollouts) as executor:
                futures = []
                potential_success_rate = 1.0
                for rollout_id in range(rollout_num_episodes):
                    futures.append(executor.submit(rollout_worker, args, args.agent, args.horizon, rollout_id))

                for future in as_completed(futures):
                    stats = future.result()  # Collect rollout stats
                    if stats is not None:
                        success_rate = stats["Success_Rate"]
                        success_rates.append(success_rate)
                        episode_success.append(success_rate)

                        # based on the past success rates, calculate the potential average success rate (averages over all the rollout_num_episodes)
                        potential_success_rate = (1 * (rollout_num_episodes - len(episode_success)) + np.asarray(episode_success).sum()) / rollout_num_episodes
                        
                    # if success rate is 1. or the potential success rate is less than the MAX success rate, break the loop and cancel the remaining rollouts
                    if potential_success_rate < max_success_rate:
                        for f in futures:
                            if not f.done():
                                f.cancel()  # Attempt to cancel remaining rollouts
                        break
                    
        # if the lenght of the success rates is less than the rollout_num_episodes, fill the remaining with 0
        while len(success_rates) < rollout_num_episodes * (int(i/step)):  
            success_rates.append(0.0)
            
        episode_success_rate = np.mean(np.asarray(episode_success))
        if episode_success_rate > max_success_rate:
            max_success_rate = np.mean(np.asarray(episode_success))      
                          


    # Save and plot results as before
    success_rates = np.array(success_rates)
    success_rates = success_rates.reshape(-1, rollout_num_episodes)
    save_and_plot_results(success_rates, args.agent_path, trained_epochs, step)
    
    # get the best success rate and the epoch
    best_success_rate = np.max(success_rates.mean(axis=1))
    best_epoch = np.argmax(success_rates.mean(axis=1)) * step + 50
    return best_success_rate, best_epoch


def save_and_plot_results(success_rates, agent_path, trained_epochs, step):
    """Save success rates to disk and generate plots."""
    # success_rates = np.array(success_rates)
    parent_folder = os.path.dirname(os.path.dirname(agent_path))
    agent_results_folder = os.path.join(parent_folder, "agent_results")
    os.makedirs(agent_results_folder, exist_ok=True)
    print(f"Saving results to {agent_results_folder}")

    # Save success rates to a file
    np.save(os.path.join(agent_results_folder, "success_rates.npy"), success_rates)

    # Plot success rates over epochs
    plt.figure()
    x = np.arange(len(success_rates))
    plt.plot(x, success_rates.mean(axis=1))
    plt.xticks(x, np.arange(50, trained_epochs + 1, step), rotation=45)
    plt.xlabel("Epochs")
    plt.ylabel("Success Rate")
    plt.title("Success Rate vs Epochs")
    plt.savefig(os.path.join(agent_results_folder, "success_rates.png"))

    # Plot histogram of the last 10 success rates
    epochs = np.arange(50, trained_epochs + 1, step)
    last_10_epochs = epochs[-10:]
    best_success_rate = np.max(success_rates.mean(axis=1))
    last_10_success_rates = success_rates.mean(axis=1)[-10:]

    plt.figure()
    plt.axhline(y=best_success_rate, color='r', linestyle='-', label="Best Success Rate")
    plt.bar(range(len(last_10_epochs)), last_10_success_rates, tick_label=last_10_epochs)
    plt.xlabel("Last 10 Success Rates")
    plt.ylabel("Success Rate")
    plt.title("Histogram of Last 10 Success Rates")
    plt.savefig(os.path.join(agent_results_folder, "last_10_success_rates_histogram.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--agent_path",
        default= "cluster_output/mrl_trained_models_ds/mrl_can_ph/20240910174015/models",
        type=str,
        help="path to saved checkpoint pth file",
    )
    
    # number of rollouts
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=30,
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
        default=None,
        help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
    )


    # for seeding before starting rollouts
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) set seed for rollouts",
    )

    args = parser.parse_args()
    run_trained_agent(args, max_parallel_rollouts=5)

