#!/usr/bin/env bash

# Define the tasks and datasets
methods=("mrl")
tasks=("can" "lift")
datasets=("mg")

# Loop over each method
for method in "${methods[@]}"
do
    # Loop over each task
    for task in "${tasks[@]}"
    do
        # Loop over each dataset
        for dataset in "${datasets[@]}"
        do
            # Construct the config file path
            config_file="/Midgard/home/longhini/robomimic/exps/RAL/${method}/${task}_${dataset}.json"
            sbatch --export=CONFIG_FILE=$config_file robomimic.sbatch
            sleep 1

        done
    done
done