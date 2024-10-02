import os
import glob
from albi.parallel_eval_different_stages import run_trained_agent
import argparse
from datetime import datetime
import logging
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from albi.parallel_eval_different_stages import rollout_worker

def run_trained_agent_last(args, max_parallel_rollouts=4, trained_epochs=2000):
    """Run the trained agent with parallel rollouts, capped by `max_parallel_rollouts`, using CPU only."""
    rollout_num_episodes = args.n_rollouts

    print(f"Rolling out agent at epoch {trained_epochs}")

    # Find the correct checkpoint path for the current epoch
    args.agent = glob.glob(args.agent_path + f"/model_epoch_{trained_epochs}*")
    args.agent.sort()
    args.agent = args.agent[0]

    episode_success = []

    # Run rollouts in parallel with a limited number of workers
    with ProcessPoolExecutor(max_workers=max_parallel_rollouts) as executor:
        futures = []
        for rollout_id in range(rollout_num_episodes):
            futures.append(executor.submit(rollout_worker, args, args.agent, args.horizon, rollout_id))

        for future in as_completed(futures):
            stats = future.result()  # Collect rollout stats
            if stats is not None:
                success_rate = stats["Success_Rate"]
                episode_success.append(success_rate)

        
    episode_success_rate = np.mean(np.asarray(episode_success))
    
    save_results(episode_success_rate, args.agent_path)

    return episode_success_rate

def save_results(success_rates, agent_path):
    """Save success rates to disk and generate plots."""
    # success_rates = np.array(success_rates)
    parent_folder = os.path.dirname(os.path.dirname(agent_path))
    agent_results_folder = os.path.join(parent_folder, "agent_last_results")
    os.makedirs(agent_results_folder, exist_ok=True)
    print(f"Saving results to {agent_results_folder}")

    
    # Save success rates to a file
    np.save(os.path.join(agent_results_folder, "last_success_rate.npy"), success_rates)


def get_args(agent_path, n_rollouts=30):
    args = argparse.Namespace(
    agent_path=agent_path,
    n_rollouts=n_rollouts,
    horizon=None,
    env=None,
    render=False,
    video_path=None,
    video_skip=5,
    camera_names=["agentview"],
    dataset_path=None,
    seed=None
    )
    return args

# Function to log results in the log file
def log_results(method, task, dataset, best_success_rate, best_epoch):
    with open(log_file, 'a') as f:
        # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = (
            f"Method: {method}, Task: {task}, Dataset: {dataset}, "
            f"Best Success Rate: {best_success_rate}, Best Epoch: {best_epoch}\n"
        )
        f.write(log_entry)
        print(log_entry)  # Print to console as well for real-time feedback

def print_message(message, log_file):
    with open(log_file, 'a') as f:
        print(message)  # Print to the console
        f.write(message + "\n")  # Write to the log file with a newline
        

def run_evaluation(methods, tasks, datasets, cluster_folder, training_epochs, parallel_rollouts, log_txt_file):
    for method in methods:
        folder_path = f"{method}_trained_models_ds"
        print_message(f"********* Evaluating {method} *********", log_txt_file)
        # logging.info()
        for task in tasks:
            for dataset in datasets:
                results_path = f"{cluster_folder}/{folder_path}/{method}_{task}_{dataset}"
                if not os.path.exists(results_path):
                    print_message(f"Skipping {method}_{task}_{dataset} as folder does not exist", log_txt_file)
                    continue
                else:
                    # check if the folder agent_results exists, if yes, do not repeat the evaluation
                    if os.path.exists(os.path.join(results_path, "agent_last_results")):
                        success_rates = np.load(os.path.join(results_path, "agent_last_results", "last_success_rate.npy"), allow_pickle=True)
                        best_success_rate = success_rates
                                            
                        # Log the best success rate and epoch
                        print_message(f"{method} | {task} | {dataset} = {best_success_rate} at epoch {training_epochs}", log_txt_file) 
                        log_results(method, task, dataset, best_success_rate, training_epochs)
                        continue
                    
                    # check now if the model finished training, which means that the last element in the folder is the model with epoch 2000
                    nested_folder = os.listdir(results_path)
                    nested_folder.sort()
                    nested_folder = nested_folder[0]
                    models_folder =os.path.join(results_path, nested_folder, "models")
                    all_models = glob.glob(os.path.join(results_path, nested_folder, "models", "*.pth"))
                    if not any(f"model_epoch_{training_epochs}.pth" in m for m in all_models):
                        print_message(f"Model for {method}_{task}_{dataset} has not finished training, skipping evaluation", log_txt_file)
                        continue
                    
                    # if the model has finished training, then run the evaluation
                    args = get_args(agent_path=models_folder, n_rollouts=30)
                    best_success_rate = run_trained_agent_last(args, max_parallel_rollouts=parallel_rollouts, trained_epochs=training_epochs)
                        
                    # Log the best success rate and epoch               
                    print_message(f"{method} | {task} | {dataset} = {best_success_rate} at epoch {training_epochs}", log_txt_file)
                    log_results(method, task, dataset, best_success_rate, training_epochs)
            
                    


if __name__=="__main__":
    training_epochs = 1000
    # training_epochs = 200
    parallel_rollouts = 15
    
    cluster_folder = "cluster_output"
    log_file = os.path.join(cluster_folder, f"evaluation_results_LAST_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    # create output text file based on the day and time
    log_txt_file = os.path.join(cluster_folder, f"evaluation_results_LAST_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")
    
    # methods = ["mrl", "bc", "cql", "crl", "qrl", "bcq", "iql"]
    # #methods = ["mrl", "bc"]
    # tasks = ["can", "lift"]
    # datasets = ["mg"]
    
    # # logging.info("Starting evaluation of the trained models")
    # print_message(f"Starting evaluation DATASET: {datasets[0]}", log_txt_file)
    
    # run_evaluation(methods, tasks, datasets, cluster_folder, training_epochs,  parallel_rollouts, log_txt_file)
    
    # methods = ["mrl", "bc", "cql", "crl", "qrl", "bcq", "iql"]
    # #methods = ["mrl", "bc", ]
    # tasks = ["can", "lift", "square", "transport"]
    # datasets = ["ph"]
    
    # # logging.info("Starting evaluation of the trained models")
    # print_message(f"Starting evaluation DATASET: {datasets[0]}", log_txt_file)
    
    # run_evaluation(methods, tasks, datasets, cluster_folder, training_epochs, parallel_rollouts, log_txt_file)
        

    # methods = ["mrl", "bc", "cql", "crl", "qrl", "bcq", "iql"]
    # #methods = ["mrl", "bc", ]
    # tasks = ["can", "lift", "square", "transport"]
    # datasets = ["mh"]
    
    # # logging.info("Starting evaluation of the trained models")
    # print_message(f"Starting evaluation DATASET: {datasets[0]}", log_txt_file)
    
    # run_evaluation(methods, tasks, datasets, cluster_folder, training_epochs, parallel_rollouts, log_txt_file)
    
    methods = ["mrl", "bc"]
    #methods = ["mrl", "bc", ]
    tasks = ["can"]
    datasets = ["random"]
    
    # logging.info("Starting evaluation of the trained models")
    print_message(f"Starting evaluation DATASET: {datasets[0]}", log_txt_file)
    
    run_evaluation(methods, tasks, datasets, cluster_folder, training_epochs, parallel_rollouts, log_txt_file)
        
                