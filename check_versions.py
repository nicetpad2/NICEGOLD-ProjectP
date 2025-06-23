#!/usr/bin/env python3
"""
Quick version check for tracking dependencies
"""
import sys

def check_versions():
    print("=== Tracking System Dependencies Check ===")
    print(f"Python version: {sys.version}")
    
    try:
        import mlflow
        print(f"✅ MLflow: {mlflow.__version__}")
    except ImportError:
        print("❌ MLflow: Not installed")
    
    try:
        import wandb
        print(f"✅ Weights & Biases: {wandb.__version__}")
    except ImportError:
        print("❌ Weights & Biases: Not installed")
    
    try:
        import rich
        print(f"✅ Rich: {rich.__version__}")
    except ImportError:
        print("❌ Rich: Not installed")
    
    try:
        import click
        print(f"✅ Click: {click.__version__}")
    except ImportError:
        print("❌ Click: Not installed")
    
    try:
        import typer
        print(f"✅ Typer: {typer.__version__}")
    except ImportError:
        print("❌ Typer: Not installed")
    
    try:
        import yaml
        print(f"✅ PyYAML: Available")
    except ImportError:
        print("❌ PyYAML: Not installed")
    
    try:
        import psutil
        print(f"✅ PSUtil: {psutil.__version__}")
    except ImportError:
        print("❌ PSUtil: Not installed")
    
    try:
        import joblib
        print(f"✅ Joblib: {joblib.__version__}")
    except ImportError:
        print("❌ Joblib: Not installed")
    
    try:
        import matplotlib
        print(f"✅ Matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("❌ Matplotlib: Not installed")
    
    try:
        import seaborn
        print(f"✅ Seaborn: {seaborn.__version__}")
    except ImportError:
        print("❌ Seaborn: Not installed")

if __name__ == "__main__":
    check_versions()
