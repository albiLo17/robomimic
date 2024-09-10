import robomimic
from robomimic.config import BCConfig, CQLConfig, IQLConfig, TD3_BCConfig
from robomimic.utils.train_utils import train
import os

# Define paths
dataset_path = "/home/omniverse/workspace/robomimic/datasets/lift/ph/low_dim_v141.hdf5"  # Path to your Lift dataset
output_dir = "/home/omniverse/workspace/robomimic/output"  # Directory where models and logs will be saved

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# List of model configurations to train
model_configs = {
    "BC": BCConfig(),
    "CQL": CQLConfig(),
    "IQL": IQLConfig(),
    "TD3": TD3_BCConfig(),
}

# Loop through the model configurations and train each one
for model_name, config in model_configs.items():
    # Set dataset path in config
    config.train.data = dataset_path
    
    # Set output directory in config
    config.train.output_dir = os.path.join(output_dir, model_name)
    
    # Set environment name
    config.train.env = "Lift"
    
    # Train the model
    print(f"Training {model_name} model...")
    train(config=config)
    print(f"{model_name} model training complete!")

print("All models trained successfully!")