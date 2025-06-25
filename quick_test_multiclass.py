#!/usr/bin/env python3
            from scipy.special import expit
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np
import os
import pandas as pd
"""
Quick test for multi_class fix
ทดสอบรวดเร็วสำหรับการแก้ไข multi_class
"""


print("🔧 Quick Multi - class Test...")

# Create test data
np.random.seed(42)
X = np.random.randn(100, 5)
y = np.random.choice([0, 1, -1], 100)

print(f"✅ Test data: X shape {X.shape}, y classes {len(set(y))}")

# Train LogisticRegression
model = LogisticRegression(random_state = 42, max_iter = 1000)
model.fit(X, y)
print("✅ Model trained")

# Test predict_proba
try:
    pred_proba = model.predict_proba(X)
    print(f"✅ predict_proba works: {pred_proba.shape}")

    # Get binary probabilities
    if pred_proba.shape[1] == 2:
        binary_proba = pred_proba[:, 1]
    else:
        binary_proba = pred_proba.max(axis = 1)

    print(f"✅ Binary probabilities: min = {binary_proba.min():.3f}, max = {binary_proba.max():.3f}")

except Exception as e:
    print(f"❌ predict_proba failed: {e}")

    # Test fallback method
    try:
        if hasattr(model, 'decision_function'):
            decision_scores = model.decision_function(X)
            print(f"✅ decision_function works: {decision_scores.shape}")

            # Convert to probabilities
            binary_proba = expit(decision_scores) if decision_scores.ndim == 1 else expit(decision_scores)[:, 1]
            print(f"✅ Converted probabilities: min = {binary_proba.min():.3f}, max = {binary_proba.max():.3f}")
        else:
            print("❌ No decision_function available")
    except Exception as e2:
        print(f"❌ Fallback failed: {e2}")

print("🎉 Quick test completed!")