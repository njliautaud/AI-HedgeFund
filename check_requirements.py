#!/usr/bin/env python3
"""
AI Hedge Fund - Requirements Checker

This script checks if all required Python packages are installed.
"""

import pkg_resources
import sys
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed."""
    # Path to requirements.txt
    req_file = Path(__file__).parent / 'requirements.txt'
    
    if not req_file.exists():
        print(f"Error: {req_file} not found")
        return False
    
    print(f"Checking requirements from {req_file}...\n")
    
    # Read requirements
    with open(req_file, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    missing_packages = []
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    
    # Check each requirement
    for req in requirements:
        try:
            req_obj = pkg_resources.Requirement.parse(req)
            pkg_name = req_obj.key
            spec = str(req_obj.specs[0]) if req_obj.specs else "any"
            
            if pkg_name in installed_packages:
                version = installed_packages[pkg_name]
                print(f"✓ {pkg_name} ({version}) is installed")
            else:
                print(f"✗ {pkg_name} is not installed")
                missing_packages.append(str(req_obj))
        except Exception as e:
            print(f"! Error checking {req}: {e}")
    
    # Print summary
    print("\n" + "=" * 50)
    if missing_packages:
        print(f"\nMissing {len(missing_packages)} package(s):")
        for pkg in missing_packages:
            print(f"- {pkg}")
        
        print("\nTo install missing packages, run:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print("\n✓ All required packages are installed!")
        return True

if __name__ == "__main__":
    if check_requirements():
        sys.exit(0)
    else:
        sys.exit(1)
