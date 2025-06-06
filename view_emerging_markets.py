#!/usr/bin/env python3
"""
Emerging Markets Viewer

A simple script to display emerging markets data in the terminal.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from src.data.emerging_markets import get_emerging_markets_summary

def main():
    """Display emerging markets data in the terminal."""
    try:
        print("\n" + "="*80)
        print("EMERGING MARKETS OVERVIEW".center(80))
        print("="*80 + "\n")
        
        # Get and print the emerging markets summary
        summary = get_emerging_markets_summary()
        print(summary)
        
        print("="*80)
        print("End of report".center(80))
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\nError: {str(e)}\n")
        print("Please make sure all required dependencies are installed.")
        print("You can install them by running: pip install -r requirements.txt\n")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
