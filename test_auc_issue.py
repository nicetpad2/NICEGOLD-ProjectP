#!/usr/bin/env python3
"""
Quick test of the exact AUC calculation issue
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def test_auc_issue():
    """Reproduce the exact AUC calculation issue"""
    print("Testing AUC calculation issue...")
    
    # Create test data similar to what we see in the logs
    np.random.seed(42)
    X = np.random.rand(1000, 5)
    
    # Create target with ONLY 0 and 1 (binary)
    y = np.random.choice([0, 1], size=1000, p=[0.6, 0.4])
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"y unique values: {np.unique(y)}")
    
    # Train a model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    print(f"Model classes: {model.classes_}")
    
    # Get predictions - this should return 2D array
    proba = model.predict_proba(X)
    print(f"Proba shape: {proba.shape}")
    print(f"Proba first few: {proba[:3]}")
    
    # Extract positive class probability (class 1)
    if 1 in model.classes_:
        idx = list(model.classes_).index(1)
        y_pred = proba[:, idx]
    else:
        y_pred = proba[:, -1]
    
    print(f"y_pred shape: {y_pred.shape}")
    print(f"y_pred range: [{y_pred.min():.3f}, {y_pred.max():.3f}]")
    
    # Now try AUC calculation
    try:
        auc = roc_auc_score(y, y_pred)
        print(f"✓ AUC calculation successful: {auc:.3f}")
        return True
    except Exception as e:
        print(f"❌ AUC calculation failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_multiclass_scenario():
    """Test a scenario where the model has more than 2 classes but target is binary"""
    print("\nTesting multiclass model with binary target...")
    
    np.random.seed(42)
    X = np.random.rand(1000, 5)
    
    # Create training target with 3 classes
    y_train_multi = np.random.choice([0, 1, 2], size=1000, p=[0.4, 0.3, 0.3])
    
    # Create test target with only 2 classes (this might cause the issue)
    y_test_binary = np.random.choice([0, 1], size=200, p=[0.6, 0.4])
    X_test = np.random.rand(200, 5)
    
    print(f"Training y unique: {np.unique(y_train_multi)}")
    print(f"Test y unique: {np.unique(y_test_binary)}")
    
    # Train model on multiclass data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y_train_multi)
    
    print(f"Model classes: {model.classes_}")
    
    # Get predictions on test data
    proba = model.predict_proba(X_test)
    print(f"Proba shape: {proba.shape}")
    
    # Extract positive class probability
    if 1 in model.classes_:
        idx = list(model.classes_).index(1)
        y_pred = proba[:, idx]
    else:
        y_pred = proba[:, -1]
    
    print(f"y_pred shape: {y_pred.shape}")
    
    # Try AUC calculation
    try:
        auc = roc_auc_score(y_test_binary, y_pred)
        print(f"✓ AUC calculation successful: {auc:.3f}")
        return True
    except Exception as e:
        print(f"❌ AUC calculation failed: {e}")
        print(f"y_test_binary shape: {y_test_binary.shape}")
        print(f"y_pred shape: {y_pred.shape}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success1 = test_auc_issue()
    success2 = test_multiclass_scenario()
    
    if success1 and success2:
        print("\n🎉 All AUC tests passed!")
    else:
        print("\n❌ Some AUC tests failed - need further investigation")
