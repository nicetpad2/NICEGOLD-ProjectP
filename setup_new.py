#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Installation and Setup Script for NICEGOLD ProjectP v2.0
Run this script to automatically install all dependencies and set up the project
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🚀 {description}")
    try:
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            return True
        else:
            print(f"❌ {description} - Failed")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False


def main():
    """Main setup function"""
    print("🚀 NICEGOLD ProjectP v2.0 - Setup Script")
    print("=" * 50)

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)

    print(f"✅ Python version: {sys.version}")

    # Essential packages
    essential_packages = [
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "joblib",
        "pyyaml",
        "tqdm",
        "requests",
        "psutil",
    ]

    print(f"\n📦 Installing {len(essential_packages)} essential packages...")
    for package in essential_packages:
        success = run_command(
            f"{sys.executable} -m pip install {package}", f"Installing {package}"
        )
        if not success:
            print(f"⚠️ Failed to install {package}, continuing...")

    # Optional ML packages
    ml_packages = ["xgboost", "lightgbm", "optuna"]
    print(f"\n🤖 Installing {len(ml_packages)} ML packages...")
    for package in ml_packages:
        success = run_command(
            f"{sys.executable} -m pip install {package}", f"Installing {package}"
        )
        if not success:
            print(f"⚠️ {package} installation failed (optional)")

    # Web packages
    web_packages = ["streamlit", "fastapi", "uvicorn"]
    print(f"\n🌐 Installing {len(web_packages)} web packages...")
    for package in web_packages:
        success = run_command(
            f"{sys.executable} -m pip install {package}", f"Installing {package}"
        )
        if not success:
            print(f"⚠️ {package} installation failed (optional)")

    # Create directories
    print("\n📁 Creating project directories...")
    directories = ["datacsv", "output_default", "models", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

    # Create sample data
    print("\n📊 Creating sample configuration...")

    config_content = """
project:
  name: "NICEGOLD ProjectP"
  version: "2.0.0"
  description: "Professional AI Trading System"

data:
  input_folder: "datacsv"
  output_folder: "output_default"
  models_folder: "models"
  logs_folder: "logs"

trading:
  initial_balance: 10000
  max_position_size: 0.1
  stop_loss: 0.02
  take_profit: 0.04

ml:
  models: ["RandomForest", "XGBoost", "LightGBM"]
  test_size: 0.2
  cv_folds: 5
  random_state: 42

api:
  dashboard_port: 8501
  api_port: 8000
  host: "localhost"
"""

    with open("config.yaml", "w") as f:
        f.write(config_content)
    print("✅ Created config.yaml")

    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Add your trading data CSV files to the 'datacsv' folder")
    print("2. Run the main application: python ProjectP.py")
    print("3. Choose option 1 to run the full pipeline")

    print(f"\n💡 Project structure:")
    print("📁 NICEGOLD-ProjectP/")
    print("  ├── ProjectP.py (main application)")
    print("  ├── core/ (core modules)")
    print("  ├── utils/ (utility functions)")
    print("  ├── datacsv/ (input data)")
    print("  ├── output_default/ (results)")
    print("  ├── models/ (trained models)")
    print("  └── logs/ (log files)")


if __name__ == "__main__":
    main()
