"""
Databricks authentication using ~/.databrickscfg
This reads from the standard Databricks config file in the user's home directory.
"""

import os
from pathlib import Path
from configparser import ConfigParser


def get_databricks_auth(profile: str = "DEFAULT"):
    """
    Get Databricks authentication credentials from ~/.databrickscfg
    
    Args:
        profile: The profile name to use from .databrickscfg (default: "DEFAULT")
    
    Returns:
        dict: Authentication credentials with 'host', 'token', and 'method'
        
    Raises:
        FileNotFoundError: If ~/.databrickscfg doesn't exist
        ValueError: If the profile is missing or incomplete
    """
    config_path = Path.home() / ".databrickscfg"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Databricks config not found at {config_path}\n\n"
            "Please create ~/.databrickscfg with:\n"
            "[DEFAULT]\n"
            "host = https://your-workspace.cloud.databricks.com\n"
            "token = your-token-here\n"
        )
    
    config = ConfigParser()
    config.read(config_path)
    
    if profile not in config:
        available = ", ".join(config.sections())
        raise ValueError(
            f"Profile '{profile}' not found in {config_path}\n"
            f"Available profiles: {available}"
        )
    
    profile_config = config[profile]
    
    if 'host' not in profile_config or 'token' not in profile_config:
        raise ValueError(
            f"Profile '{profile}' is incomplete in {config_path}\n"
            "Required fields: host, token"
        )
    
    return {
        'host': profile_config['host'].rstrip('/'),
        'token': profile_config['token'],
        'method': 'databrickscfg'
    }



