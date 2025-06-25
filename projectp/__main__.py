#!/usr/bin/env python3
from .cli import main_cli
import argparse
            import os
import sys
"""
ProjectP - Main Module Entry Point
Run with: python -m projectp
"""


def main():
    """Main entry point for python -m projectp"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description = "ProjectP - Advanced AI Trading Pipeline")
    parser.add_argument(
        " -  - mode", 
        choices = ["full_pipeline", "debug_full_pipeline", "preprocess", "realistic_backtest", "robust_backtest", "realistic_backtest_live"], 
        help = "Pipeline mode to run directly"
    )
    parser.add_argument(
        " -  - env", 
        choices = ["dev", "prod"], 
        default = "dev", 
        help = "Environment config"
    )
    parser.add_argument(
        " -  - auto", 
        action = "store_true", 
        help = "Run in auto mode without interactive menu"
    )

    args = parser.parse_args()

    # If mode is specified, set it and run directly
    if args.mode:
        sys.argv = ['projectp', ' -  - mode', args.mode, ' -  - env', args.env]
        if args.auto:
            # Set auto mode to skip interactive menu
            os.environ['PROJECTP_AUTO_MODE'] = '1'

    # Call the main CLI function
    main_cli()

if __name__ == "__main__":
    main()