import os
import h5py
import numpy as np

def update_dataset(original_file_path, new_file_path, proficient_dataset_path, random_data_percentage, goal_anchor=False):
    # Open the original and proficient datasets
    with h5py.File(original_file_path, 'r') as original_file, h5py.File(proficient_dataset_path, 'r') as proficient_file:
        random_demos = list(original_file["data"].keys())
        proficient_demos = list(proficient_file["data"].keys())
        
        # Merge proficient dataset into the original dataset
        # merged_demos = demos + proficient_demos
        
        # # Determine the last demo number in the original file
        # last_demo_number = max([int(demo.split('_')[-1]) for demo in demos])
        
        # Determine the number of random trajectories to replace
        total_demos = len(proficient_demos)
        num_random_demos = int((random_data_percentage / 100.0) * total_demos)
        
        # Randomly select demos to be replaced with random data
        random_demos_indices = np.random.choice(range(total_demos), size=num_random_demos, replace=False)

        # Open a new file in write mode
        with h5py.File(new_file_path, 'w') as new_file:
            # Copy all items from the original file to the new file
            for item in proficient_file:
                proficient_file.copy(item, new_file)
                
                
            obs_keys = list(proficient_file["data"][proficient_demos[0]]["obs"].keys())
            
            # store the average of the last observations
            obs_sum = {k: np.zeros(new_file["data"][proficient_demos[0]]["obs"][k][-1].shape) for k in obs_keys}
            obs_goal = {k: np.zeros(new_file["data"][proficient_demos[0]]["obs"][k][-1].shape) for k in obs_keys}
            all_obs_goal = {}
            num_demos_used = 0
                
            # Process the merged demos
            added_random_demos = 0
            for i, ep in enumerate(proficient_demos):
                # if i is not in the random demos indices, then copy the proficient data
                if i in random_demos_indices:
                    # delete current demo in the new file and suibstitute with the random data ep keeping the original name
                    del new_file["data"][ep]
                    new_file.copy(original_file["data"][random_demos[random_demos_indices[added_random_demos]]], new_file["data"], name=ep)
                    added_random_demos += 1
                    
                # add key demo to all obs
                all_obs_goal[ep] = {}
                for k in  obs_keys:
                    if goal_anchor:
                        all_obs_goal[ep][k] = np.zeros_like(new_file["data"][ep]["obs"][k][-1])
                    else:
                        all_obs_goal[ep][k] = new_file["data"][ep]["obs"][k][-1]
                # first check that the last done is 1 that in our case means that the episode is successful
                if new_file["data"][ep]["dones"][-1] == 1:
                    num_demos_used += 1
                    # update the average
                    for k in  obs_keys:
                        obs_sum[k] += new_file["data"][ep]["obs"][k][-1]
                        obs_goal[k] = new_file["data"][ep]["obs"][k][-1]

                    
            # compute the average
            obs_avg = {k: obs_sum[k]/num_demos_used for k in obs_keys}
            obs_last = {k: obs_goal[k] for k in obs_keys}
            obs_anchor = {k: np.zeros_like(obs_goal[k]) for k in obs_keys}
            
            print(f"Average of the last observations computed over {num_demos_used}/{len(proficient_demos)} successful episodes")

            # create the "goal" group under "data"
            if "goal_obs" in new_file:
                del new_file["goal_obs"]
            new_file.create_group("goal_obs")
            for ep in proficient_demos:
                new_file["goal_obs"].create_group(ep)
                for k in obs_keys:
                    new_file["goal_obs"][ep].create_dataset(f"{k}", data=all_obs_goal[ep][k])
            # for k in obs_keys:
            #     new_file["goal_obs"].create_dataset(k, data=obs_last[k])
                
            # Now you can modify the new file
            for ep in proficient_demos:
                # do it twice to make sure that the anchor state will be propertly loaded for training the representation
                for i in range(1):
                    # Read the data
                    actions = new_file["data"][ep]["actions"][:]
                    dones = new_file["data"][ep]["dones"][:]
                    rewards = new_file["data"][ep]["rewards"][:]
                    states = new_file["data"][ep]["states"][:]
                    obs = {k: v[:] for k, v in new_file["data"][ep]["obs"].items()}
                    next_obs = {k: v[:] for k, v in new_file["data"][ep]["next_obs"].items()}
                    
                    # only append if the last done is 1
                    if dones[-1] == 1:                        
                        new_file["data"][ep].attrs["num_samples"] += 1
                        # Augment the data
                        augmented_actions = np.concatenate([actions, np.zeros((1, actions[0].shape[0]))], axis=0)
                        augmented_dones = np.concatenate([dones, np.ones((1))], axis=0)
                        augmented_rewards = np.concatenate([rewards, rewards[-1][None]], axis=0)
                        augmented_states = np.concatenate([states, states[-1][None,:]], axis=0)
                        augmented_obs = {k: np.concatenate([v, next_obs[k][-1][None,:]], axis=0) for k, v in obs.items()}
                        augmented_next_obs = {k: np.concatenate([v, np.zeros_like(v[-1])[None,:]], axis=0) for k, v in next_obs.items()}

                        # Delete the old datasets
                        del new_file["data"][ep]["actions"]
                        del new_file["data"][ep]["dones"]
                        del new_file["data"][ep]["rewards"]
                        del new_file["data"][ep]["states"]
                        for k in new_file["data"][ep]["obs"].keys():
                            del new_file["data"][ep]["obs"][k]
                            del new_file["data"][ep]["next_obs"][k]

                        # Create new datasets with the augmented data
                        new_file["data"][ep].create_dataset("actions", data=augmented_actions)
                        new_file["data"][ep].create_dataset("dones", data=augmented_dones)
                        new_file["data"][ep].create_dataset("rewards", data=augmented_rewards)
                        new_file["data"][ep].create_dataset("states", data=augmented_states)
                        for k in augmented_obs.keys():
                            new_file["data"][ep]["obs"].create_dataset(k, data=augmented_obs[k])
                            new_file["data"][ep]["next_obs"].create_dataset(k, data=augmented_next_obs[k])
                            
            new_file["data"].attrs["total"] = np.asarray([new_file["data"][ep].attrs["num_samples"] for ep in proficient_demos]).sum()
                        
            print("Dataset updated successfully")
            print(f"New dataset saved at {new_file_path}")
            
            # now create a new h5py file in the same folder containing only the
            # average of the last observations
            new_file_path = new_file_path.replace(".hdf5", "_goal.hdf5")
            with h5py.File(new_file_path, 'w') as goal_file:
                for k, v in obs_avg.items():
                    goal_file.create_dataset(k, data=v)
                print(f"Goal file saved at {new_file_path}")
    
    # open original file as write to add the goal
    with h5py.File(original_file_path, 'a') as original_file:
        # chekc if original file has the group goal_obs
        if "goal_obs" in original_file:
            del original_file["goal_obs"]
        original_file.create_group("goal_obs")
        for k in obs_keys:
            if goal_anchor:
                original_file["goal_obs"].create_dataset(k, data=obs_anchor[k])
            else:
                original_file["goal_obs"].create_dataset(k, data=obs_last[k])
            
    
if __name__ == "__main__":
    # download the dataset
    
        
    random_data_percentage = 10  # Set this to control the percentage of random data
    
    for random_data_percentage in [10, 20, 50, 70]:
        dataset_type = f"random"
        # dataset_type = f"random_brown"
        hdf5_type = "low_dim_sparse"
        
        if not os.path.exists(os.path.join("./datasets", "can", f"{dataset_type}_{random_data_percentage}")):
            os.makedirs(os.path.join("./datasets", "can", f"{dataset_type}_{random_data_percentage}"), exist_ok=True)
            # copy the dataset in random folder in this folder
            os.system(f"cp ./datasets/can/{dataset_type}/{hdf5_type}_v141.hdf5 ./datasets/can/{dataset_type}_{random_data_percentage}/{hdf5_type}_v141.hdf5")
            
        
        # for task in ["transport", "lift", "square", "can"]:
        for task in ["can"]:
            dataset_path = os.path.join("./datasets", task, f"{dataset_type}_{random_data_percentage}", f"{hdf5_type}_v141.hdf5")
            proficient_dataset_path = os.path.join("./datasets", task, "ph", f"low_dim_v141.hdf5")
            new_dataset_path = os.path.join("./datasets", task, f"{dataset_type}_{random_data_percentage}", f"{hdf5_type}_v141_augmented.hdf5")
            
            update_dataset(dataset_path, new_dataset_path, proficient_dataset_path, random_data_percentage, goal_anchor=True)
