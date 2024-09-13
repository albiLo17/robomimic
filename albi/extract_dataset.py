import os
import h5py
import numpy as np


# def load_dataset(dataset_path):
#     print("********* REMEMBER TO CHANGE THE OBS KEYS *********")
#     # open file
#     f = h5py.File(dataset_path, "r")
#     # each demonstration is a group under "data"
#     demos = list(f["data"].keys())
#     num_demos = len(demos)
#     print("hdf5 file {} has {} demonstrations".format(dataset_path, num_demos))
    
#     # each demonstration is named "demo_#" where # is a number.
#     # Let's put the demonstration list in increasing episode order
#     inds = np.argsort([int(elem[5:]) for elem in demos])
#     demos = [demos[i] for i in inds]

#     obs_keys = [             # specify low-dim observations for agent
#             "robot0_eef_pos", 
#             "robot0_eef_quat", 
#             "robot0_gripper_qpos", 
#             "object",
#         ]
#     formatted_demos = []
#     # store the average of the observations
#     # obs_sum = {k: np.zeros(f["data"][demos[0]]["obs"][k].shape) for k in obs_keys}
#     for ep in demos:
#         # augment all the observations with one more final state
#         actions = f["data"][ep]["actions"][:]
#         # concatenate the observations with obs_keys
#         obs = np.concatenate([f["data"][ep]["obs"][key][:] for key in obs_keys], axis=-1)
#         next_obs = np.concatenate([f["data"][ep]["next_obs"][key][:] for key in obs_keys], axis=-1)
#         formatted_demos.append((obs, actions, next_obs))

        
        
def update_dataset(original_file_path, new_file_path):
    # Open the original file in read-only mode
    with h5py.File(original_file_path, 'r') as original_file:
        demos = list(original_file["data"].keys())
        # Open a new file in write mode
        with h5py.File(new_file_path, 'w') as new_file:
            # Copy all items from the original file to the new file
            for item in original_file:
                original_file.copy(item, new_file)
                
            obs_keys = list(original_file["data"][demos[0]]["obs"].keys())
                
            # store the average of the last observations
            obs_sum = {k: np.zeros(new_file["data"][demos[0]]["obs"][k][-1].shape) for k in obs_keys}
            obs_goal = {k: np.zeros(new_file["data"][demos[0]]["obs"][k][-1].shape) for k in obs_keys}
            all_obs_goal = {}
            num_demos_used = 0
            for ep in demos:
                # add key demo to all obs
                all_obs_goal[ep] = {}
                for k in  obs_keys:
                    all_obs_goal[ep][k] = new_file["data"][ep]["obs"][k][-1]
                # first check that the last done is 1 that in our case means that the episode is successful
                if new_file["data"][ep]["dones"][-1] == 1:
                    num_demos_used += 1
                    # update the average
                    for k in  obs_keys:
                        # print(f"obs shape: {new_file['data'][ep]['obs'][k].shape}")
                        # print(f"last shape: {new_file['data'][ep]['obs'][k][-1].shape}")
                        obs_sum[k] += new_file["data"][ep]["obs"][k][-1]
                        obs_goal[k] = new_file["data"][ep]["obs"][k][-1]
                    
            # compute the average
            obs_avg = {k: obs_sum[k]/num_demos_used for k in obs_keys}
            obs_last = {k: obs_goal[k] for k in obs_keys}
            
            print(f"Average of the last observations computed over {num_demos_used}/{len(demos)} successful episodes")
            
            # add to the new_file["data"] the average of the last observations
            # create the "goal" group under "data"
            if "goal_obs" in new_file:
                del new_file["goal_obs"]
            new_file.create_group("goal_obs")
            for ep in demos:
                new_file["goal_obs"].create_group(ep)
                for k in obs_keys:
                    new_file["goal_obs"][ep].create_dataset(f"{k}", data=all_obs_goal[ep][k])
            # for k in obs_keys:
            #     new_file["goal_obs"].create_dataset(k, data=obs_last[k])
                
            # Now you can modify the new file
            for ep in demos:
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
            original_file["goal_obs"].create_dataset(k, data=obs_last[k])
            
    


if __name__=="__main__":
    # download the dataset
    task = "transport" # lift square transport lift
    dataset_type = "mh"
    hdf5_type = "low_dim"
    
    for task in ["transport", "lift", "square", "can"]:
        dataset_path = os.path.join("./datasets", task, dataset_type, f"{hdf5_type}_v141.hdf5")
        
        new_dataset_path = os.path.join("./datasets", task, dataset_type, f"{hdf5_type}_v141_augmented.hdf5")
        
        update_dataset(dataset_path, new_dataset_path)
    
    dataset_type = "ph"
    hdf5_type = "low_dim"
    
    for task in ["transport", "lift", "square", "can"]:
        dataset_path = os.path.join("./datasets", task, dataset_type, f"{hdf5_type}_v141.hdf5")
        
        new_dataset_path = os.path.join("./datasets", task, dataset_type, f"{hdf5_type}_v141_augmented.hdf5")
        
        update_dataset(dataset_path, new_dataset_path)
    
    dataset_type = "mg"
    hdf5_type = "low_dim_sparse"
    
    # for task in ["transport", "lift", "square", "can"]:
    for task in ["lift", "can"]:
        dataset_path = os.path.join("./datasets", task, dataset_type, f"{hdf5_type}_v141.hdf5")
        
        new_dataset_path = os.path.join("./datasets", task, dataset_type, f"{hdf5_type}_v141_augmented.hdf5")
        
        update_dataset(dataset_path, new_dataset_path)
    

    # augment all the observations with one more final state
    # for ep in demos:
    #     # concatenate a dummy action
    #     f["data"][ep]["actions"] = np.concatenate([f["data"][ep]["actions"][:], np.zeros((1, f["data"][ep]["actions"][0].shape[0]))], axis=0)
    #     f["data"][ep]["dones"] = np.concatenate([f["data"][ep]["dones"][:], np.ones((1, 1))], axis=0)
    #     f["data"][ep]["rewards"] = np.concatenate([f["data"][ep]["rewards"][:], f["data"][ep]["rewards"][-1]*np.ones((1, 1))], axis=0)
    #     f["data"][ep]["states"] = np.concatenate([f["data"][ep]["states"][:], f["data"][ep]["states"][-1][None,:]], axis=0)
    #     for k in f["data"][ep]["obs"].keys():
    #         f["data"][ep]["obs"][k] = np.concatenate([f["data"][ep]["obs"][k][:], f["data"][ep]["next_obs"][k][-1][None,:]], axis=0)
    #         f["data"][ep]["next_obs"][k] = np.concatenate([f["data"][ep]["next_obs"][k][:], np.zeros_like(f["data"][ep]["next_obs"][k][-1])[None,:]], axis=0)
    # # save npz array
    # save_path = os.path.join("./datasets", task, dataset_type, "low_dim_v141.npz")
    
    # # save the dataset
    # np.savez(save_path, demos=formatted_demos)
    
    # save_path = os.path.join("./datasets", task, dataset_type, "low_dim_v141.hdf5")

    # # save the dataset
    # with h5py.File(save_path, 'w') as f:
    #     # You may need to adapt this depending on the structure of formatted_demos
    #     for i, demo in enumerate(formatted_demos):
    #         f.create_dataset(f'demo_{i}', data=demo)
