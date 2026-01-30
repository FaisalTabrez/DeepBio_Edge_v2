import yaml
import os
from pathlib import Path

def load_config(config_path: str = "configs/main.yaml") -> dict:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
    
    Returns:
        dict: The configuration dictionary.
    """
    # Get the project root directory (assuming this script is in src/utils/)
    # Adjusted to handle running from root or various depths if needed, 
    # but strictly following the request to look in configs/ relative to run dir or absolute.
    # For robustness, let's try to find relative to project root.
    
    # Assuming the standard structure: project_root/src/utils/config.py
    # So project_root is two levels up from this file's directory.
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # Check if path is absolute, if not prepend project root
    path_obj = Path(config_path)
    if not path_obj.is_absolute():
        full_path = project_root / config_path
    else:
        full_path = path_obj

    if not full_path.exists():
        # Fallback to checking from current working directory if project structure assumption fails
        if Path(config_path).exists():
             full_path = Path(config_path)
        else:
            raise FileNotFoundError(f"Configuration file not found at {full_path} or {config_path}")

    with open(full_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config

if __name__ == "__main__":
    try:
        config = load_config()
        print("Configuration loaded successfully:")
        print(config)
    except Exception as e:
        print(f"Error loading config: {e}")
