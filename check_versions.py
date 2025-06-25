#!/usr/bin/env python3
        import click
        import joblib
        import matplotlib
        import mlflow
        import psutil
        import rich
        import seaborn
import sys
        import typer
        import wandb
        import yaml
"""
Quick version check for tracking dependencies
"""

def check_versions():
    print(" =  = = Tracking System Dependencies Check = =  = ")
    print(f"Python version: {sys.version}")

    try:
        print(f"✅ MLflow: {mlflow.__version__}")
    except ImportError:
        print("❌ MLflow: Not installed")

    try:
        print(f"✅ Weights & Biases: {wandb.__version__}")
    except ImportError:
        print("❌ Weights & Biases: Not installed")

    try:
        print(f"✅ Rich: {rich.__version__}")
    except ImportError:
        print("❌ Rich: Not installed")

    try:
        print(f"✅ Click: {click.__version__}")
    except ImportError:
        print("❌ Click: Not installed")

    try:
        print(f"✅ Typer: {typer.__version__}")
    except ImportError:
        print("❌ Typer: Not installed")

    try:
        print(f"✅ PyYAML: Available")
    except ImportError:
        print("❌ PyYAML: Not installed")

    try:
        print(f"✅ PSUtil: {psutil.__version__}")
    except ImportError:
        print("❌ PSUtil: Not installed")

    try:
        print(f"✅ Joblib: {joblib.__version__}")
    except ImportError:
        print("❌ Joblib: Not installed")

    try:
        print(f"✅ Matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("❌ Matplotlib: Not installed")

    try:
        print(f"✅ Seaborn: {seaborn.__version__}")
    except ImportError:
        print("❌ Seaborn: Not installed")

if __name__ == "__main__":
    check_versions()