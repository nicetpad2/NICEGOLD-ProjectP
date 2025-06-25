import importlib
import os
import sys
"""
A simple script to test if necessary imports are available in src.strategy
"""

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT_DIR)

def test_strategy_imports():
    """Test that src.strategy has all required imports"""
    strategy = importlib.import_module('src.strategy')

    # Test for utility functions
    print("\nTesting utility functions:")
    for func_name in ['safe_load_csv_auto', 'simple_converter']:
        has_attr = hasattr(strategy, func_name)
        print(f"  Has {func_name}: {has_attr}")

    # Test for SHAP helper functions
    print("\nTesting SHAP helper functions:")
    for func_name in ['select_top_shap_features', 'check_model_overfit', 
                     'analyze_feature_importance_shap', 'check_feature_noise_shap']:
        has_attr = hasattr(strategy, func_name)
        print(f"  Has {func_name}: {has_attr}")

if __name__ == "__main__":
    test_strategy_imports()