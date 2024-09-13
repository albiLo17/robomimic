import os
import json

def update_nested_attribute(json_data, key_path, value):
    """
    Update a nested attribute in the JSON data using a dot-separated key path.
    
    :param json_data: Dictionary loaded from a JSON file.
    :param key_path: Dot-separated key path (e.g., "train.data").
    :param value: New value to set for the given key.
    """
    keys = key_path.split('.')
    sub_dict = json_data
    for key in keys[:-1]:
        sub_dict = sub_dict.get(key, {})
    sub_dict[keys[-1]] = value

def update_json_attributes(json_data, new_attributes):
    """
    Updates specific attributes in the JSON data.
    
    :param json_data: Dictionary loaded from a JSON file.
    :param new_attributes: Dictionary of new attributes to update.
    :return: Updated JSON data.
    """
    for k in new_attributes.keys():
        # Update the algorithm name
        if 'algo_name' in k:
            update_nested_attribute(json_data, 'algo_name', new_attributes['algo_name'])
            # json_data['algo_name'] = new_attributes['algo_name']
            # keep the rest after the spitt
            new_name = json_data["experiment"]["name"].replace(json_data["experiment"]["name"].split('_')[0], new_attributes['algo_name'])
            update_nested_attribute(json_data, "experiment.name", new_name)
            new_output = json_data["train"]["output_dir"].replace(json_data["train"]["output_dir"].split('/')[-1].split('_')[0], new_attributes['algo_name'])
            update_nested_attribute(json_data, "train.output_dir", new_output)
            # json_data['experiment.name'] = json_data['experiment.name'].replace(json_data['experiment.name'].split('_')[0], new_attributes['algo_name'])
        
        else:
        # Update the number of epochs
            update_nested_attribute(json_data, k, new_attributes[k])
            # json_data[k] = new_attributes[k]
    
    # Update other attributes as needed (you can add more here)
    
    return json_data

def delete_nested_attribute(json_data, key_path):
    """
    Delete a nested attribute in the JSON data using a dot-separated key path.
    
    :param json_data: Dictionary loaded from a JSON file.
    :param key_path: Dot-separated key path (e.g., "train.epochs").
    """
    keys = key_path.split('.')
    sub_dict = json_data
    for key in keys[:-1]:
        sub_dict = sub_dict.get(key, {})
    
    if keys[-1] in sub_dict:
        del sub_dict[keys[-1]]
        
def delete_json_attributes(json_data, delete_attributes):
    """
    Deletes specific attributes in the JSON data.
    
    :param json_data: Dictionary loaded from a JSON file.
    :param delete_attributes: List of dot-separated keys for attributes to delete.
    :return: Updated JSON data.
    """
    for key_path in delete_attributes:
        delete_nested_attribute(json_data, key_path)
    
    return json_data
        
def process_json_files_in_folder(folder_path, new_attributes, delete_attributes):
    """
    Processes all JSON files in the given folder, updates specific attributes, deletes attributes,
    and saves the changes.
    
    :param folder_path: Path to the folder containing JSON files.
    :param new_attributes: Dictionary of new attributes to update, with dot-separated keys.
    :param delete_attributes: List of attributes to delete, with dot-separated keys.
    """
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            
            # Load the JSON file
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            
            # Update the JSON data with new attributes
            updated_data = update_json_attributes(json_data, new_attributes)
            
            # Delete specific attributes
            updated_data = delete_json_attributes(updated_data, delete_attributes)
            
            # Save the updated JSON back to the file
            with open(file_path, 'w') as f:
                json.dump(updated_data, f, indent=4)
            
            print(f"Updated {filename}")

if __name__ == "__main__":
    # Path to the folder containing JSON files
    folder_path = "robomimic/exps/RAL/bc"
    
    # # Define the new attributes to update (customize as needed)
    new_attributes = {
        "algo_name": "bc",
        "train.num_epochs": 2000,  # Example of new epoch values
        "observation.modalities.goal.low_dim": []
    }
    delete_attributes = ["train.epochs"]    
    process_json_files_in_folder(folder_path, new_attributes, delete_attributes)
    
    algos = ["bc", "bcq", "cql", "crl", "iql", "mrl", "qrl"]
    
    for algo in algos:
        folder_path = f"robomimic/exps/RAL/{algo}"
        # Define the new attributes to update (customize as needed)
        new_attributes = {
            "algo_name": algo,
            "train.num_epochs": 2000,  # Example of new epoch values
            "experiment.logging.wandb_proj_name": "MetricRL_Baselines",
            "observation.modalities.goal.low_dim": []
        }
        
        # Define the attributes to delete (customize as needed)    
        delete_attributes = ["train.epochs"]    
        if algo not in ["mrl", "qrl", "crl"]:
            delete_attributes.append("algo.phi_dim")
            delete_attributes.append("algo.pre_train")
            delete_attributes.append("algo.pre_train_epochs")
            
        # Process all JSON files in the folder
        process_json_files_in_folder(folder_path, new_attributes, delete_attributes)