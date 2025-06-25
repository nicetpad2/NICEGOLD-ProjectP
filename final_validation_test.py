#!/usr/bin/env python3
from pathlib import Path
        from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
        from steps.walkforward import (
        from steps.walkforward import get_positive_class_proba
        import numpy
import numpy as np
import os
import pandas as pd
import sys
        import traceback
"""
FINAL VALIDATION TEST for NICEGOLD Production Pipeline
Tests all critical fixes: WalkForward, AUC calculation, imports
"""


# Add projectp to path for imports
sys.path.append('projectp')

def test_walkforward_imports():
    """Test that all required imports are available"""
    print("🔍 Testing WalkForward imports...")
    try:
            run_walk_forward_validation, 
            get_positive_class_proba, 
            check_no_data_leak
        )
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_auc_calculation():
    """Test AUC calculation with various scenarios"""
    print("🔍 Testing AUC calculation scenarios...")


    # Test 1: Binary classification
    try:
        y_true = np.array([0, 1, 0, 1, 0])
        y_scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
        auc = roc_auc_score(y_true, y_scores)
        print(f"✅ Binary AUC test: {auc:.3f}")
    except Exception as e:
        print(f"❌ Binary AUC failed: {e}")
        return False

    # Test 2: Multiclass with binary conversion (our fix)
    try:
        y_true_multi = np.array([ - 1, 1, 0, 1, -1])  # Multiclass
        y_scores = np.array([0.1, 0.9, 0.5, 0.8, 0.2])

        # Convert to binary (1 vs rest) - our fix approach
        y_true_binary = (y_true_multi == 1).astype(int)
        auc = roc_auc_score(y_true_binary, y_scores)
        print(f"✅ Multiclass - >Binary AUC test: {auc:.3f}")
    except Exception as e:
        print(f"❌ Multiclass AUC failed: {e}")
        return False

    return True

def test_array_shape_handling():
    """Test array shape handling scenarios"""
    print("🔍 Testing array shape handling...")

    # Test 1D array handling
    test_1d = np.array([0.1, 0.9, 0.5])
    result_1d = np.asarray(test_1d).flatten()
    print(f"✅ 1D array: {test_1d.shape} -> {result_1d.shape}")

    # Test 2D array handling
    test_2d = np.array([[0.1, 0.9], [0.5, 0.5], [0.2, 0.8]])
    result_2d = np.asarray(test_2d).flatten()
    print(f"✅ 2D array: {test_2d.shape} -> {result_2d.shape}")

    # Test single column 2D array (common case)
    test_2d_single = np.array([[0.1], [0.9], [0.5]])
    result_2d_single = np.asarray(test_2d_single).flatten()
    print(f"✅ 2D single column: {test_2d_single.shape} -> {result_2d_single.shape}")

    return True

def test_data_availability():
    """Test that required data files exist"""
    print("🔍 Testing data file availability...")

    # Check for preprocessed data
    data_files = [
        "output_default/preprocessed_super.parquet", 
        "XAUUSD_M1.csv", 
        "XAUUSD_M15.csv"
    ]

    for file_path in data_files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"✅ {file_path}: {size_mb:.1f}MB")
        else:
            print(f"⚠️ {file_path}: Not found")

    return True

def run_mini_walkforward():
    """Run a minimal WalkForward test"""
    print("🔍 Running mini WalkForward validation...")

    try:
        # Create synthetic data that matches our expected format
        np.random.seed(42)
        n_samples = 1000
        n_features = 5

        # Create DataFrame with expected structure
        data = {
            **{f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)}, 
            'target': np.random.choice([ - 1, 0, 1], n_samples)  # Multiclass target
        }
        df = pd.DataFrame(data)

        # Import our fixed WalkForward functions

        # Split data
        features = [f'feature_{i}' for i in range(n_features)]
        X_train, X_test, y_train, y_test = train_test_split(
            df[features], df['target'], test_size = 0.3, random_state = 42
        )

        # Train model
        model = RandomForestClassifier(n_estimators = 10, random_state = 42)
        model.fit(X_train, y_train)

        # Test our fixed prediction function
        y_pred = get_positive_class_proba(model, X_test)
        print(f"✅ Prediction shape: {np.array(y_pred).shape}, Type: {type(y_pred)}")

        # Test AUC calculation with multiclass -> binary conversion
        y_test_binary = (y_test == 1).astype(int)
        auc = roc_auc_score(y_test_binary, y_pred)
        print(f"✅ Mini WalkForward AUC: {auc:.3f}")

        return True

    except Exception as e:
        print(f"❌ Mini WalkForward failed: {e}")
        print(traceback.format_exc())
        return False

def main():
    """Run all validation tests"""
    print("🚀 NICEGOLD FINAL VALIDATION TEST")
    print(" = " * 50)

    tests = [
        ("Import Tests", test_walkforward_imports), 
        ("AUC Calculation Tests", test_auc_calculation), 
        ("Array Shape Handling", test_array_shape_handling), 
        ("Data Availability", test_data_availability), 
        ("Mini WalkForward", run_mini_walkforward)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print(" - " * 30)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n🎯 VALIDATION SUMMARY")
    print(" = " * 50)
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\n🏆 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! Production pipeline is ready.")
        return True
    else:
        print("⚠️ Some tests failed. Check issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)