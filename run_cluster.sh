#!/usr/bin/env bash

# Define the tasks and datasets
# # methods=("mrl" "bc" "cql" "crl" "qrl" "bcq" "iql")
methods=("mrl" "bc" )
tasks=("can")
datasets=("random")

# Loop over each method
for method in "${methods[@]}"
do
    # Loop over each task
    for task in "${tasks[@]}"
    do
        # Loop over each dataset
        for dataset in "${datasets[@]}"
        do
            model_dir="/Midgard/home/longhini/robomimic/output/${method}_trained_models_ds/${method}_${task}_${dataset}"
            if [ -d "$model_dir" ]; then
            
                rm -rf "$model_dir"
                echo "DEBUG: Removed existing model directory: $model_dir"
            else
                echo "DEBUG: Model directory $model_dir does not exist, proceeding with training."
            fi

            # Construct the config file path
            config_file="/Midgard/home/longhini/robomimic/robomimic/exps/RAL/${method}/${task}_${dataset}.json"
            sbatch --export=CONFIG_FILE=$config_file robomimic.sbatch
            sleep 1

        done
    done
done

# # methods=("mrl" "bc" "cql" "crl" "qrl" "bcq" "iql")
# methods=("cql" "crl" "qrl" "bcq" "iql")
# tasks=("can" "lift" "square" "transport")
# datasets=("ph" "mh")

# # Loop over each method
# for method in "${methods[@]}"
# do
#     # Loop over each task
#     for task in "${tasks[@]}"
#     do
#         # Loop over each dataset
#         for dataset in "${datasets[@]}"
#         do
#             model_dir="/Midgard/home/longhini/robomimic/output/${method}_trained_models_ds/${method}_${task}_${dataset}"
#             if [ -d "$model_dir" ]; then
            
#                 rm -rf "$model_dir"
#                 echo "DEBUG: Removed existing model directory: $model_dir"
#             else
#                 echo "DEBUG: Model directory $model_dir does not exist, proceeding with training."
#             fi

#             # Construct the config file path
#             config_file="/Midgard/home/longhini/robomimic/robomimic/exps/RAL/${method}/${task}_${dataset}.json"
#             sbatch --export=CONFIG_FILE=$config_file robomimic.sbatch
#             sleep 1

#         done
#     done
# done