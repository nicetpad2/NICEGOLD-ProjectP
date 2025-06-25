#!/usr/bin/env python3
from pathlib import Path
        from src.real_data_loader import RealDataLoader
import os
import sys
"""
Quick Status Test for NICEGOLD ProjectP
Test system health without interactive menu
"""


# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


def test_quick_status():
    """Test basic system status quickly"""
    print("🔍 NICEGOLD ProjectP - Quick Status Check")
    print(" = " * 50)

    # Check Python version
    print(f"🐍 Python Version: {sys.version}")

    # Check working directory
    print(f"📁 Working Directory: {PROJECT_ROOT}")

    # Check if essential packages can be imported
    essential_packages = [
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

    print("\n📦 Essential Packages:")
    imported = 0
    for pkg in essential_packages:
        try:
            __import__(pkg)
            print(f"   ✅ {pkg}")
            imported += 1
        except ImportError:
            print(f"   ❌ {pkg} - MISSING")
        except Exception as e:
            print(f"   ⚠️  {pkg} - ERROR: {e}")

    print(f"\n📊 Package Status: {imported}/{len(essential_packages)} installed")

    # Check ML packages
    ml_packages = ["catboost", "xgboost", "lightgbm", "optuna", "shap", "ta"]
    print("\n🤖 ML Packages:")
    ml_imported = 0
    for pkg in ml_packages:
        try:
            __import__(pkg)
            print(f"   ✅ {pkg}")
            ml_imported += 1
        except ImportError:
            print(f"   ❌ {pkg} - MISSING")
        except Exception as e:
            print(f"   ⚠️  {pkg} - ERROR: {e}")

    print(f"\n🤖 ML Package Status: {ml_imported}/{len(ml_packages)} installed")

    # Check data files
    print("\n📊 Data Files:")
    data_files = ["datacsv/XAUUSD_M1.csv", "datacsv/XAUUSD_M15.csv", "config.yaml"]
    for file_path in data_files:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            size_mb = full_path.stat().st_size / (1024 * 1024)
            print(f"   ✅ {file_path} ({size_mb:.1f} MB)")
        else:
            print(f"   ❌ {file_path} - NOT FOUND")

    # Check directories
    print("\n📁 Required Directories:")
    required_dirs = ["src", "datacsv", "output_default", "models", "logs", "plots"]
    for dir_name in required_dirs:
        dir_path = PROJECT_ROOT / dir_name
        if dir_path.exists():
            print(f"   ✅ {dir_name}/")
        else:
            print(f"   ❌ {dir_name}/ - MISSING")
            # Try to create it
            try:
                dir_path.mkdir(parents = True, exist_ok = True)
                print(f"   🔨 Created {dir_name}/")
            except Exception as e:
                print(f"   💥 Failed to create {dir_name}/: {e}")

    # Test basic data loading
    print("\n🔍 Testing Data Loading:")
    try:

        loader = RealDataLoader("config.yaml")
        print("   ✅ RealDataLoader imported successfully")

        # Try to load small sample
        m1_data = loader.load_m1_data(limit_rows = 100)
        if m1_data is not None and len(m1_data) > 0:
            print(f"   ✅ M1 data loaded: {len(m1_data)} rows")
        else:
            print("   ❌ M1 data loading failed")

    except Exception as e:
        print(f"   ❌ Data loading test failed: {e}")

    print("\n" + " = " * 50)
    print("🏁 Quick Status Check Complete!")


if __name__ == "__main__":
    test_quick_status()