#!/usr/bin/env python3
"""
AI Hedge Fund - Setup Script

This script helps set up the development environment for the AI Hedge Fund project.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Required directories
REQUIRED_DIRS = [
    PROJECT_ROOT / 'data',
    PROJECT_ROOT / 'logs',
    PROJECT_ROOT / 'models',
    PROJECT_ROOT / 'notebooks'
]

# Required environment files
ENV_FILES = {
    '.env.example': '.env'
}

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"{text:^80}")
    print("=" * 80 + "\n")

def create_directories():
    """Create required directories if they don't exist."""
    print_header("CREATING REQUIRED DIRECTORIES")
    
    for directory in REQUIRED_DIRS:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory: {directory}")
        except Exception as e:
            print(f"✗ Failed to create {directory}: {e}")

def setup_environment():
    """Set up environment files."""
    print_header("SETTING UP ENVIRONMENT FILES")
    
    for src, dest in ENV_FILES.items():
        src_path = PROJECT_ROOT / src
        dest_path = PROJECT_ROOT / dest
        
        if not src_path.exists():
            print(f"✗ Source file not found: {src_path}")
            continue
            
        if dest_path.exists():
            print(f"✓ {dest} already exists, skipping...")
            continue
            
        try:
            shutil.copy(src_path, dest_path)
            print(f"✓ Created {dest} from {src}")
        except Exception as e:
            print(f"✗ Failed to create {dest}: {e}")

def install_dependencies():
    """Install Python dependencies."""
    print_header("INSTALLING PYTHON DEPENDENCIES")
    
    requirements_file = PROJECT_ROOT / 'requirements.txt'
    
    if not requirements_file.exists():
        print(f"✗ Requirements file not found: {requirements_file}")
        return
    
    try:
        print("Installing dependencies from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")

def main():
    """Main setup function."""
    print_header("AI HEDGE FUND SETUP")
    print("This script will help you set up your development environment.\n")
    
    # Create required directories
    create_directories()
    
    # Set up environment files
    setup_environment()
    
    # Install dependencies
    install_dependencies()
    
    print_header("SETUP COMPLETE")
    print("\nNext steps:")
    print("1. Edit the .env file with your API keys and configuration")
    print("2. Run the dashboard with: python run_dashboard.py")
    print("3. Open your browser to: http://localhost:8088\n")

if __name__ == "__main__":
    main()
