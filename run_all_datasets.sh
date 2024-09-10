
# Run the missing lift ph and can mg
# python robomimic/scripts/train.py --config robomimic/exps/RAL/mrl_can_mg.json
# python robomimic/scripts/train.py --config robomimic/exps/RAL/mrl_lift_ph.json

# models_dir="output/mrl_trained_models/MetricRL_can_PH/"

# # Get the name of the first folder inside the models directory
# model_folder=$(ls -d "$models_dir"*/ | head -1)
# model_folder="${model_folder}models" 
# echo "Model folder: $model_folder"

# # Run the evaluation script using the found model folder
# python albi/evaluate_model_different_stages.py --agent "$model_folder"



# Define the tasks and datasets
methods=("mrl")
# methods=("mrl" "crl")
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
            config_file="robomimic/exps/RAL/${method}/${task}_${dataset}.json"
            
            # Print the current config file being used
            echo "Running training with config: $config_file"
            
            # Run the training script with the current config
            python robomimic/scripts/train.py --config "$config_file"

            # Construct the models directory path
            models_dir="output/mrl_trained_models_ds/${method}_${task}_${dataset}/"

            # Get the name of the first folder inside the models directory
            model_folder=$(ls -d "$models_dir"*/ | head -1)
            model_folder="${model_folder}models"
            
            # Verify that the model folder exists
            if [ ! -d "$model_folder" ]; then
                echo "Model folder does not exist: $model_folder"
                continue
            fi
            
            # Print the model folder path
            echo "Model folder: $model_folder"
            
            # Run the evaluation script using the found model folder
            echo "Running evaluation for model folder: $model_folder"
            python albi/parallel_eval_different_stages.py --agent "$model_folder"
        done
    done
done



# Define the tasks and datasets
methods=("mrl")
# methods=("mrl" "crl")
tasks=("can" "lift") # "square" "transport")
datasets=("ph" "mh")
# Loop over each method
for method in "${methods[@]}"
do
    # Loop over each task
    for task in "${tasks[@]}"
    do
        # Loop over each dataset
        for dataset in "${datasets[@]}"
        do
            Construct the config file path
            config_file="robomimic/exps/RAL/${method}/${task}_${dataset}.json"
            
            # Print the current config file being used
            echo "Running training with config: $config_file"
            
            # Run the training script with the current config
            python robomimic/scripts/train.py --config "$config_file"

            # Construct the models directory path
            models_dir="output/mrl_trained_models_ds/${method}_${task}_${dataset}/"

            # Get the name of the first folder inside the models directory
            model_folder=$(ls -d "$models_dir"*/ | head -1)
            model_folder="${model_folder}models"
            
            # Verify that the model folder exists
            if [ ! -d "$model_folder" ]; then
                echo "Model folder does not exist: $model_folder"
                continue
            fi
            
            # Print the model folder path
            echo "Model folder: $model_folder"
            
            # Run the evaluation script using the found model folder
            echo "Running evaluation for model folder: $model_folder"
            python albi/parallel_eval_different_stages.py --agent "$model_folder"
        done
    done
done
