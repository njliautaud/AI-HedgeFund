# AI Hedge Fund Utilities

This directory contains utility modules used throughout the AI Hedge Fund system.

## Modules

### ConfigManager

The `config_manager.py` module provides configuration loading and management functionality:

- YAML configuration loading
- Environment variable substitution
- Default value handling
- Configuration validation

Usage example:

```python
from utils.config_manager import ConfigManager

# Load configuration
config = ConfigManager.load_config('config.yaml')

# Access configuration values
dashboard_port = config.get('ui', {}).get('dashboard_port', 8080)
```