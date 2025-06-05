#!/usr/bin/env python3
"""
AI Hedge Fund - Configuration Checker

This script verifies that the configuration is set up correctly.
"""

import os
import sys
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"{text:^80}")
    print("=" * 80 + "\n")

def check_environment():
    """Check environment variables."""
    print_header("CHECKING ENVIRONMENT VARIABLES")
    
    # Load .env file if it exists
    env_file = PROJECT_ROOT / '.env'
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✓ Loaded environment variables from {env_file}")
    else:
        print(f"ℹ No .env file found at {env_file}")
    
    # Check required environment variables
    required_vars = [
        'DASHBOARD_PORT',
        'API_PORT',
        'DATABASE_URL'
    ]
    
    missing_vars = []
    for var in required_vars:
        if os.getenv(var) is None:
            missing_vars.append(var)
    
    if missing_vars:
        print("✗ Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these variables in your .env file.")
        return False
    
    print("✓ All required environment variables are set")
    return True

def check_config_file():
    """Check the main configuration file."""
    print_header("CHECKING CONFIGURATION FILE")
    
    config_file = PROJECT_ROOT / 'config.yaml'
    
    if not config_file.exists():
        print(f"✗ Configuration file not found: {config_file}")
        return False
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✓ Configuration file loaded successfully: {config_file}")
        
        # Check required sections
        required_sections = ['system', 'logging', 'trading', 'database', 'ui']
        missing_sections = [s for s in required_sections if s not in config]
        
        if missing_sections:
            print("✗ Missing required configuration sections:")
            for section in missing_sections:
                print(f"  - {section}")
            return False
        
        print("✓ All required configuration sections are present")
        return True
        
    except Exception as e:
        print(f"✗ Error loading configuration file: {e}")
        return False

def check_directories():
    """Check required directories."""
    print_header("CHECKING REQUIRED DIRECTORIES")
    
    required_dirs = [
        PROJECT_ROOT / 'data',
        PROJECT_ROOT / 'logs',
        PROJECT_ROOT / 'models',
    ]
    
    all_exist = True
    for directory in required_dirs:
        if directory.exists() and directory.is_dir():
            print(f"✓ Directory exists: {directory}")
        else:
            print(f"✗ Directory does not exist: {directory}")
            all_exist = False
    
    if not all_exist:
        print("\nPlease run 'python setup.py' to create missing directories.")
    
    return all_exist

def main():
    """Main function to run all checks."""
    print_header("AI HEDGE FUND CONFIGURATION CHECK")
    
    env_ok = check_environment()
    config_ok = check_config_file()
    dirs_ok = check_directories()
    
    print_header("VERIFICATION SUMMARY")
    
    if env_ok and config_ok and dirs_ok:
        print("✓ All checks passed! You're ready to run the AI Hedge Fund system.")
        print("\nTo start the dashboard, run:")
        print("  python run_dashboard.py")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above and try again.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
