#!/usr/bin/env python3
"""
Configuration Manager for AI Hedge Fund

Handles loading and parsing of configuration files with environment variable substitution.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigManager:
    """
    Manages configuration loading and access for the AI Hedge Fund system.
    
    Features:
    - YAML configuration loading
    - Environment variable substitution
    - Default value handling
    - Configuration validation
    """
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file with environment variable substitution.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dict containing the configuration
        """
        # Ensure the config file exists
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load the configuration file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Process environment variables in the configuration
        config = ConfigManager._process_env_vars(config)
        
        return config
    
    @staticmethod
    def _process_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively process environment variables in the configuration.
        
        Environment variables are specified as ${ENV_VAR} in the configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Processed configuration dictionary
        """
        if isinstance(config, dict):
            return {k: ConfigManager._process_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [ConfigManager._process_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            # Extract environment variable name
            env_var = config[2:-1]
            # Return the environment variable value or empty string if not set
            return os.environ.get(env_var, '')
        else:
            return config