import yaml
import os
from typing import List

def load_api_keys_from_yaml(file_path: str = "config/api_keys.yaml") -> List[str]:
    """
    Load the API key list from a YAML file.
    
    Args:
        file_path: path to the YAML config file
        
    Returns:
        list of API keys
        
    Raises:
        FileNotFoundError: config file not found
        yaml.YAMLError: malformed YAML
        KeyError: required key missing in the config file
    """
    try:
        # Check whether the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        # Read the YAML file
        with open(file_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        # Validate the config structure
        if not isinstance(config, dict):
            raise ValueError("Malformed config file: the root node must be a dict")
        
        if 'gemini' not in config:
            raise KeyError("Missing 'gemini' section in the config file")
        
        if 'api_keys' not in config['gemini']:
            raise KeyError("Missing 'gemini.api_keys' in the config file")
        
        api_keys = config['gemini']['api_keys']
        
        # Validate the API key list
        if not isinstance(api_keys, list):
            raise ValueError("api_keys must be a list")
        
        if not api_keys:
            raise ValueError("API key list must not be empty")
        
        # Drop empty or invalid keys
        valid_keys = [key.strip() for key in api_keys if key and isinstance(key, str) and key.strip()]
        
        if not valid_keys:
            raise ValueError("No valid API keys found")
        
        print(f"Loaded {len(valid_keys)} API keys")
        return valid_keys
        
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML parsing error: {e}")
    except Exception as e:
        raise Exception(f"Error loading config file: {e}")


# Usage example
if __name__ == "__main__":
    try:
        # Load from the default path
        api_keys = load_api_keys_from_yaml()
        print("API keys:", [key[-8:] + "..." for key in api_keys])
        
        # Or pass a custom path
        # api_keys = load_api_keys_from_yaml("my_config/keys.yaml")
        
    except Exception as e:
        print(f"Error: {e}")