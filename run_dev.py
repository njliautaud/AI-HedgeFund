#!/usr/bin/env python3
"""
AI Hedge Fund - Development Server

Run the dashboard in development mode with auto-reload.
"""

import os
import sys
import time
import subprocess
import signal
import webbrowser
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Configuration
DASHBOARD_URL = "http://localhost:8088"
RELOAD_PATTERNS = ["*.py", "*.yaml", "*.env"]

class DashboardRestartHandler(FileSystemEventHandler):
    """Handler for file system events that trigger dashboard restarts."""
    
    def __init__(self, callback):
        self.callback = callback
    
    def on_modified(self, event):
        if not event.is_directory and any(event.src_path.endswith(ext) for ext in [".py", ".yaml", ".env"]):
            print(f"\n\n🔄  File changed: {event.src_path}")
            self.callback()

class DashboardRunner:
    """Run and manage the dashboard process with auto-reload."""
    
    def __init__(self):
        self.process = None
        self.running = False
    
    def start(self):
        """Start the dashboard process."""
        if self.process:
            self.stop()
        
        print("🚀 Starting dashboard...")
        self.process = subprocess.Popen(
            [sys.executable, "run_dashboard.py"],
            stdout=sys.stdout,
            stderr=sys.stderr,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        self.running = True
        print(f"✅ Dashboard started (PID: {self.process.pid})")
        print(f"🌐 Open your browser to: {DASHBOARD_URL}")
    
    def stop(self):
        """Stop the dashboard process."""
        if not self.process:
            return
        
        print("🛑 Stopping dashboard...")
        if os.name == 'nt':
            # Windows
            import ctypes
            ctypes.windll.kernel32.GenerateConsoleCtrlEvent(1, self.process.pid)
        else:
            # Unix
            self.process.terminate()
        
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
        
        self.process = None
        self.running = False
        print("✅ Dashboard stopped")
    
    def restart(self):
        """Restart the dashboard process."""
        print("\n" + "=" * 80)
        print("🔄  Restarting dashboard...")
        print("=" * 80 + "\n")
        self.stop()
        self.start()

def setup_file_watcher(callback):
    """Set up file system watcher to detect changes."""
    event_handler = DashboardRestartHandler(callback)
    observer = Observer()
    
    # Watch for changes in Python files, YAML configs, and .env
    for pattern in ["*.py", "*.yaml", "*.env"]:
        observer.schedule(event_handler, ".", recursive=True)
    
    observer.start()
    return observer

def main():
    """Main function to run the development server."""
    print("=" * 80)
    print("AI HEDGE FUND - DEVELOPMENT SERVER".center(80))
    print("=" * 80 + "\n")
    
    # Create dashboard runner
    runner = DashboardRunner()
    
    # Set up signal handlers for clean shutdown
    def signal_handler(sig, frame):
        print("\nShutting down...")
        runner.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the dashboard
    runner.start()
    
    # Set up file watcher
    observer = setup_file_watcher(runner.restart)
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(2)  # Give the dashboard time to start
        webbrowser.open(DASHBOARD_URL)
    
    import threading
    threading.Thread(target=open_browser, daemon=True).start()
    
    print("\n👀 Watching for file changes...")
    print("Press Ctrl+C to stop\n")
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop()
        observer.stop()
        observer.join()

if __name__ == "__main__":
    main()
