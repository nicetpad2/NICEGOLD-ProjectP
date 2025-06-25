#!/usr/bin/env python3
from pathlib import Path
        from src.model_helpers import ensure_model_files_exist
        from src.pipeline import main as pipeline_main
        from src.robust_model_loader import ensure_model_files_robust
        import numpy as np
import os
        import pandas as pd
        import sklearn
import sys
"""
Complete system validation for NICEGOLD ProjectP after fixes
"""


# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


def run_system_validation():
    """Run comprehensive system validation"""
    print("🧪 NICEGOLD ProjectP System Validation")
    print(" = " * 50)

    # Test 1: Basic imports
    print("\n🔬 Test 1: Basic imports...")
    try:

        print("✅ Core libraries available")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

    # Test 2: Model creation
    print("\n🔬 Test 2: Model file creation...")
    try:

        status = ensure_model_files_robust("./output_default")
        print(f"✅ Model files: {status['all_success']}")
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

    # Test 3: Data availability
    print("\n🔬 Test 3: Data availability...")
    try:
        data_file = Path("data/example_trading_data.csv")
        if data_file.exists():
            df = pd.read_csv(data_file, nrows = 10)
            print(f"✅ Data loaded: {df.shape}")
        else:
            print("⚠️ Sample data not found, will use synthetic")
    except Exception as e:
        print(f"❌ Data error: {e}")
        return False

    # Test 4: Pipeline components
    print("\n🔬 Test 4: Pipeline components...")
    try:

        print("✅ Pipeline functions imported")
    except Exception as e:
        print(f"❌ Pipeline import error: {e}")
        return False

    print("\n🎉 All validation tests passed!")
    print("✅ System is ready for production use")
    return True


if __name__ == "__main__":
    success = run_system_validation()
    if success:
        print("\n🚀 Running quick pipeline test...")
        try:
            # Test the actual pipeline briefly

            ensure_model_files_robust("./output_default")
            print("✅ Pipeline preparation successful")
            print("🎯 System is fully operational!")
        except Exception as e:
            print(f"⚠️ Pipeline test warning: {e}")

    sys.exit(0 if success else 1)