import yaml

def load_config(path):
    """
    Loads a YAML configuration file.
    Args:
        path (str): Path to the .yaml configuration file.
    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config
