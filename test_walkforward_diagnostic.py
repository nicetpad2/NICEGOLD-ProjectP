#!/usr/bin/env python3
"""
Test Walkforward Validation - Quick Diagnostic
"""
import sys
import os
import traceback
import pandas as pd
import numpy as np

# Add project path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_walkforward_basic():
    """Test basic walkforward validation functionality."""
    try:
        print("=== Testing WalkForward Validation ===")
        
        # Import the walkforward module
        from projectp.steps.walkforward import get_positive_class_proba, run_walkforward
        from sklearn.ensemble import RandomForestClassifier
        
        print("✓ Successfully imported walkforward modules")
        
        # Test the get_positive_class_proba function with various scenarios
        print("\n--- Testing get_positive_class_proba function ---")
        
        # Create simple test data
        X_test = np.random.rand(100, 5)
        y_test = np.random.randint(0, 2, 100)
        
        # Create and train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_test, y_test)
        
        print("✓ Created and trained test model")
        
        # Test prediction function
        proba = get_positive_class_proba(model, X_test)
        print(f"✓ get_positive_class_proba returned shape: {proba.shape}")
        print(f"✓ Prediction range: [{proba.min():.3f}, {proba.max():.3f}]")
        
        # Test edge cases
        print("\n--- Testing edge cases ---")
        
        # Empty array
        proba_empty = get_positive_class_proba(model, np.array([]).reshape(0, 5))
        print(f"✓ Empty array test: shape={proba_empty.shape}")
        
        # Single sample
        proba_single = get_positive_class_proba(model, X_test[:1])
        print(f"✓ Single sample test: shape={proba_single.shape}")
        
        print("\n=== All basic tests PASSED ===")
        return True
        
    except Exception as e:
        print(f"❌ Error in walkforward test: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return False

def test_walkforward_full():
    """Test full walkforward validation with real data structure."""
    try:
        print("\n=== Testing Full WalkForward Validation ===")
        
        # Create synthetic data that mimics the real structure
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=1000, freq='1min')
        
        # Create a DataFrame with the expected structure
        df = pd.DataFrame({
            'timestamp': dates,
            'close': 2000 + np.cumsum(np.random.randn(1000) * 0.1),
            'volume': np.random.randint(100, 1000, 1000),
            'target_future_price_change': np.random.choice([0, 1], 1000, p=[0.6, 0.4])
        })
        
        # Add some features
        for i in range(5):
            df[f'feature_{i}'] = np.random.randn(1000)
        
        print(f"✓ Created synthetic dataset: {df.shape}")
        print(f"✓ Target distribution: {df['target_future_price_change'].value_counts().to_dict()}")
          # Import and test walkforward
        from projectp.steps.walkforward import run_walkforward
        
        try:
            results = run_walkforward()
            print(f"✓ WalkForward validation completed successfully")
            print(f"✓ Results type: {type(results)}")
            if hasattr(results, 'shape'):
                print(f"✓ Results shape: {results.shape}")
            return True
        except Exception as wfv_error:
            print(f"❌ WalkForward validation failed: {wfv_error}")
            print(f"Traceback:\n{traceback.format_exc()}")
            return False
            
    except Exception as e:
        print(f"❌ Error in full walkforward test: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("Starting WalkForward Validation Diagnostic Tests...")
    
    success_basic = test_walkforward_basic()
    success_full = test_walkforward_full()
    
    if success_basic and success_full:
        print("\n🎉 ALL TESTS PASSED - WalkForward validation is working correctly!")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED - WalkForward validation needs attention")
        sys.exit(1)
