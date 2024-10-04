import os
import glob
from albi.parallel_eval_different_stages import run_trained_agent
import argparse
from datetime import datetime
import logging
import numpy as np


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
        

def run_evaluation(methods, tasks, datasets, cluster_folder, training_epochs, step, parallel_rollouts, log_txt_file):
    for method in methods:
        folder_path = f"{method}_trained_models_ds"
        print_message(f"********* Evaluating {method} *********", log_txt_file)
        # logging.info()
        for task in tasks:
            for dataset in datasets:
                results_path = f"{cluster_folder}/{folder_path}/{method}_{task}_{dataset}"
                if not os.path.exists(results_path):
                    # logging.info(f"Skipping {method}_{task}_{dataset} as folder does not exist")
                    print_message(f"Skipping {method}_{task}_{dataset} as folder does not exist", log_txt_file)
                    continue
                # if task == "transport" and dataset == "mh" and method == "bc":
                #     print_message(f"Skipping {method}_{task}_{dataset} as folder does not exist", log_txt_file)
                #     continue
                else:
                    # check if the folder agent_results exists, if yes, do not repeat the evaluation
                    if os.path.exists(os.path.join(results_path, "agent_results")):
                        # logging.info(f"Results for {method}_{task}_{dataset} already exist, skipping evaluation")
                        # print_message(f"Results for {method}_{task}_{dataset} already exist, skipping evaluation", log_txt_file)
                        success_rates = np.load(os.path.join(results_path, "agent_results", "success_rates.npy"), allow_pickle=True)
                        best_success_rate = np.max(success_rates.mean(axis=1))
                        best_epoch = np.argmax(success_rates.mean(axis=1)) * step + 50
                                            
                        # Log the best success rate and epoch
                        print_message(f"{method} | {task} | {dataset} = {best_success_rate} at epoch {best_epoch}", log_txt_file) 
                        log_results(method, task, dataset, best_success_rate, best_epoch)
                        continue
                    
                    # check now if the model finished training, which means that the last element in the folder is the model with epoch 2000
                    nested_folder = os.listdir(results_path)[0]
                    models_folder =os.path.join(results_path, nested_folder, "models")
                    all_models = glob.glob(os.path.join(results_path, nested_folder, "models", "*.pth"))
                    if not any(f"model_epoch_{training_epochs}.pth" in m for m in all_models):
                        # logging.info(f"Model for {method}_{task}_{dataset} has not finished training, skipping evaluation")
                        print_message(f"Model for {method}_{task}_{dataset} has not finished training, skipping evaluation", log_txt_file)
                        continue
                    
                    # if the model has finished training, then run the evaluation
                    args = get_args(agent_path=models_folder, n_rollouts=30)
                    best_success_rate, best_epoch = run_trained_agent(args, max_parallel_rollouts=parallel_rollouts, trained_epochs=training_epochs, step=step)
                    # logging.info(f"{method} | {task} | {dataset} = {best_success_rate} at epoch {best_epoch}")
                        
                    # Log the best success rate and epoch               
                    print_message(f"{method} | {task} | {dataset} = {best_success_rate} at epoch {best_epoch}", log_txt_file)
                    log_results(method, task, dataset, best_success_rate, best_epoch)
            
                    

    

if __name__=="__main__":
    training_epochs = 1000
    step = 50
    parallel_rollouts = 15
    
    cluster_folder = "cluster_output"
    log_file = os.path.join(cluster_folder, f"evaluation_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    # create output text file based on the day and time
    log_txt_file = os.path.join(cluster_folder, f"evaluation_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")  
      
    methods = ["mrl", "bc", "cql", "crl", "qrl", "bcq", "iql"]
    #methods = ["mrl", "bc", ]
    tasks = ["can"]
    datasets = ["random"]
    
    all_datasets = [["random_10"], ["random_20"], ["random_50"], ["random_70"]]
    # all_datasets = [["random_brown_10"], ["random_brown_20"], ["random_brown_50"], ["random_brown_70"]]
    
    for datasets in all_datasets:
        # logging.info("Starting evaluation of the trained models")
        print_message(f"Starting evaluation DATASET: {datasets[0]}", log_txt_file)
        
        run_evaluation(methods, tasks, datasets, cluster_folder, training_epochs,step,  parallel_rollouts, log_txt_file)

      
    # methods = ["mrl", "bc"]
    # #methods = ["mrl", "bc", ]
    # tasks = ["can"]
    # datasets = ["fullrandom"]
    
    # # logging.info("Starting evaluation of the trained models")
    # print_message(f"Starting evaluation DATASET: {datasets[0]}", log_txt_file)
    
    # run_evaluation(methods, tasks, datasets, cluster_folder, training_epochs,step,  parallel_rollouts, log_txt_file)
    

    # methods = ["mrl", "bc", "cql", "crl", "qrl", "bcq", "iql"]
    # #methods = ["mrl", "bc"]
    # tasks = ["can", "lift"]
    # datasets = ["mg"]
    
    # # logging.info("Starting evaluation of the trained models")
    # print_message(f"Starting evaluation DATASET: {datasets[0]}", log_txt_file)
    
    # run_evaluation(methods, tasks, datasets, cluster_folder, training_epochs, step, parallel_rollouts, log_txt_file)
    
    # methods = ["mrl", "bc", "cql", "crl", "qrl", "bcq", "iql"]
    # #methods = ["mrl", "bc", ]
    # tasks = ["can", "lift", "square", "transport"]
    # datasets = ["ph"]
    
    # # logging.info("Starting evaluation of the trained models")
    # print_message(f"Starting evaluation DATASET: {datasets[0]}", log_txt_file)
    
    # run_evaluation(methods, tasks, datasets, cluster_folder, training_epochs, step, parallel_rollouts, log_txt_file)
        

    # methods = ["mrl", "bc", "cql", "crl", "qrl", "bcq", "iql"]
    # #methods = ["mrl", "bc", ]
    # tasks = ["can", "lift", "square", "transport"]
    # datasets = ["mh"]
    
    # # logging.info("Starting evaluation of the trained models")
    # print_message(f"Starting evaluation DATASET: {datasets[0]}", log_txt_file)
    
    # run_evaluation(methods, tasks, datasets, cluster_folder, training_epochs, step, parallel_rollouts, log_txt_file)
        
                