#!/usr/bin/env python3
"""
AI Hedge Fund - Test Runner

This script runs the test suite and generates coverage reports.
"""

import os
import sys
import subprocess
import webbrowser
from pathlib import Path

def run_tests():
    """Run the test suite with coverage."""
    print("Running test suite with coverage...\n")
    
    # Create coverage directory if it doesn't exist
    coverage_dir = Path("coverage")
    coverage_dir.mkdir(exist_ok=True)
    
    # Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--cov=./",
        f"--cov-report=html:{coverage_dir}/html",
        "--cov-report=xml:coverage.xml",
        "--cov-report=term"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✓ Tests completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Tests failed with exit code {e.returncode}")
        return False

def open_coverage_report():
    """Open the HTML coverage report in the default web browser."""
    report_path = Path("coverage") / "html" / "index.html"
    
    if report_path.exists():
        print(f"\nOpening coverage report: {report_path}")
        webbrowser.open(f"file://{report_path.absolute()}")
    else:
        print("\nCoverage report not found. Run tests first.")

def main():
    """Main function to run tests and open coverage report."""
    print("=" * 80)
    print("AI HEDGE FUND - TEST RUNNER".center(80))
    print("=" * 80 + "\n")
    
    # Run tests
    success = run_tests()
    
    # Open coverage report if tests passed
    if success:
        open_coverage_report()
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
