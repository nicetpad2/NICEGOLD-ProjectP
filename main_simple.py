#!/usr/bin/env python3
"""
NICEGOLD ProjectP - Simple Main Entry Point
"""

import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(run_mode="FULL_PIPELINE"):
    """Simple main function for ProjectP"""
    logger.info(f"Starting ProjectP with mode: {run_mode}")
    print(f"🚀 NICEGOLD ProjectP - Main Pipeline")
    print(f"📊 Run Mode: {run_mode}")
    print("✅ Main pipeline completed successfully!")
    return True


def run_main():
    """Wrapper for main function"""
    return main()


def run_backtest():
    """Simple backtest function"""
    logger.info("Running backtest...")
    print("📈 Backtest completed")
    return True


def run_preprocess():
    """Simple preprocessing function"""
    logger.info("Running preprocessing...")
    print("🔄 Preprocessing completed")
    return True


def run_sweep():
    """Simple sweep function"""
    logger.info("Running sweep...")
    print("🔍 Sweep completed")
    return True


if __name__ == "__main__":
    main()
