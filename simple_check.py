#!/usr/bin/env python3
from pathlib import Path
import sys
"""
Simple System Check for NICEGOLD ProjectP
"""

print("🔍 NICEGOLD ProjectP - System Check")
print(" = " * 40)

# Check Python

print(
    f"🐍 Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
)

# Test essential packages
print("\n📦 Essential Packages:")
packages = [
    "pandas", 
    "numpy", 
    "sklearn", 
    "matplotlib", 
    "seaborn", 
    "joblib", 
    "yaml", 
    "tqdm", 
    "requests", 
    "evidently", 
]
for pkg in packages:
    try:
        __import__(pkg)
        print(f"   ✅ {pkg}")
    except ImportError:
        print(f"   ❌ {pkg}")

# Test ML packages
print("\n🤖 ML Packages:")
ml_packages = ["catboost", "xgboost", "lightgbm", "optuna", "shap", "ta"]
for pkg in ml_packages:
    try:
        __import__(pkg)
        print(f"   ✅ {pkg}")
    except ImportError:
        print(f"   ❌ {pkg}")

# Check data files
print("\n📊 Data Files:")

PROJECT_ROOT = Path(__file__).parent
data_files = ["datacsv/XAUUSD_M1.csv", "datacsv/XAUUSD_M15.csv", "config.yaml"]
for file_path in data_files:
    full_path = PROJECT_ROOT / file_path
    if full_path.exists():
        size_mb = full_path.stat().st_size / (1024 * 1024)
        print(f"   ✅ {file_path} ({size_mb:.1f} MB)")
    else:
        print(f"   ❌ {file_path}")

# Check directories
print("\n📁 Directories:")
dirs = ["src", "datacsv", "output_default", "models", "logs", "plots"]
for dir_name in dirs:
    dir_path = PROJECT_ROOT / dir_name
    if dir_path.exists():
        print(f"   ✅ {dir_name}/")
    else:
        print(f"   ❌ {dir_name}/")

print("\n" + " = " * 40)
print("🏁 System Check Complete!")