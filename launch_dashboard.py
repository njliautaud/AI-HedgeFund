#!/usr/bin/env python3
"""
AI Hedge Fund - Dashboard Launcher

A simple script to launch the AI Hedge Fund dashboard without using the command line.
This provides a user-friendly way to start the system for users who prefer GUI interfaces.
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

# Ensure we're in the correct directory
os.chdir(Path(__file__).parent)

# Add the project directory to the path
sys.path.append(str(Path(__file__).parent))

# Import configuration
from utils.config_manager import ConfigManager

# Load configuration
config = ConfigManager.load_config('config.yaml')

# Get dashboard port
dashboard_port = config.get('ui', {}).get('dashboard_port', 8080)

# Dashboard URL
dashboard_url = f"http://localhost:{dashboard_port}"

def main():
    print("Starting AI Hedge Fund Dashboard...")
    print(f"Dashboard will be available at: {dashboard_url}")
    
    # Start the dashboard using streamlit
    try:
        # Launch the dashboard in a separate process
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "ui/dashboard.py", "--server.port", str(dashboard_port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a moment for the server to start
        time.sleep(2)
        
        # Open the dashboard in the default web browser
        webbrowser.open(dashboard_url)
        
        print("Dashboard is running. Close this window to stop the dashboard.")
        
        # Keep the script running until the user closes it
        process.wait()
        
    except KeyboardInterrupt:
        print("\nStopping dashboard...")
        process.terminate()
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()