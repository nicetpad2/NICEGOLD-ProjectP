#!/usr/bin/env python3
"""
Test predict.py multi_class issue specifically
ทดสอบปัญหา multi_class ในขั้นตอน predict
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

print("🔧 Testing predict.py multi_class fix...")

# Create dummy data similar to the preprocessed data
print("📊 Creating test data...")
np.random.seed(42)
n_samples = 1000

# Create features similar to your data
data = {
    'open': np.random.randn(n_samples),
    'volume': np.random.randn(n_samples),
    'returns': np.random.randn(n_samples),
    'volatility': np.random.randn(n_samples),
    'momentum': np.random.randn(n_samples),
    'rsi': np.random.randn(n_samples),
    'macd': np.random.randn(n_samples),
    'target': np.random.choice([0, 1, -1], n_samples, p=[0.8, 0.15, 0.05])
}

df = pd.DataFrame(data)
print(f"✅ Test data created: {df.shape}")
print(f"📊 Target distribution: {df['target'].value_counts().to_dict()}")

# Create and save test models
os.makedirs("output_default", exist_ok=True)

# Model 1: LogisticRegression (เป็นโมเดลที่มีปัญหา multi_class)
features = ['open', 'volume', 'returns', 'volatility', 'momentum', 'rsi', 'macd']
X = df[features]
y = df['target']

print("🤖 Training test models...")

# Train LogisticRegression
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X, y)

# Train RandomForest
rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
rf_model.fit(X, y)

# Save models
lr_path = os.path.join("output_default", "test_lr_model.pkl")
rf_path = os.path.join("output_default", "test_rf_model.pkl")

joblib.dump(lr_model, lr_path)
joblib.dump(rf_model, rf_path)

print(f"✅ Saved test models: {lr_path}, {rf_path}")

# Save test data
test_data_path = os.path.join("output_default", "preprocessed_super.parquet")
df.to_parquet(test_data_path)
print(f"✅ Saved test data: {test_data_path}")

# Save feature list
features_path = os.path.join("output_default", "train_features.txt")
with open(features_path, "w") as f:
    for feature in features:
        f.write(f"{feature}\n")
print(f"✅ Saved features list: {features_path}")

# Test prediction with both models
print("\n🧪 Testing prediction functions...")

def test_model_prediction(model_path, model_name):
    print(f"\n🔍 Testing {model_name}...")
    
    try:
        # Load model
        model = joblib.load(model_path)
        print(f"✅ Loaded {model_name}")
        
        # Test normal predict_proba
        try:
            pred_proba = model.predict_proba(X)
            print(f"✅ {model_name} predict_proba works: shape {pred_proba.shape}")
            return True
        except Exception as e:
            if "multi_class" in str(e):
                print(f"❌ {model_name} has multi_class issue: {e}")
                
                # Test fallback methods
                try:
                    if hasattr(model, 'decision_function'):
                        decision_scores = model.decision_function(X)
                        print(f"✅ {model_name} decision_function works: shape {decision_scores.shape}")
                        return True
                    else:
                        predictions = model.predict(X)
                        print(f"✅ {model_name} predict works: shape {predictions.shape}")
                        return True
                except Exception as e2:
                    print(f"❌ {model_name} all methods failed: {e2}")
                    return False
            else:
                print(f"❌ {model_name} unknown error: {e}")
                return False
                
    except Exception as e:
        print(f"❌ {model_name} loading failed: {e}")
        return False

# Test both models
lr_result = test_model_prediction(lr_path, "LogisticRegression")
rf_result = test_model_prediction(rf_path, "RandomForest")

print(f"\n📊 Test Results:")
print(f"  LogisticRegression: {'✅ PASS' if lr_result else '❌ FAIL'}")
print(f"  RandomForest: {'✅ PASS' if rf_result else '❌ FAIL'}")

# Now test the actual predict.py function
print(f"\n🚀 Testing actual predict.py function...")

try:
    # Import and run predict function
    sys.path.append('.')
    from projectp.steps.predict import run_predict
    
    # Test with LogisticRegression (problematic model)
    print(f"🔧 Testing with LogisticRegression model...")
    
    # Copy LR model to expected location
    expected_model_path = os.path.join("output_default", "catboost_model_best_cv.pkl")
    joblib.dump(lr_model, expected_model_path)
    
    # Run predict
    result = run_predict()
    print(f"✅ predict.py completed successfully!")
    print(f"📁 Output: {result}")
    
    # Check if predictions.csv exists and has correct columns
    if os.path.exists(result):
        pred_df = pd.read_csv(result)
        print(f"✅ Predictions file created: {pred_df.shape}")
        print(f"📊 Columns: {list(pred_df.columns)}")
        
        if 'pred_proba' in pred_df.columns:
            print(f"✅ pred_proba column exists")
            print(f"📊 pred_proba stats: min={pred_df['pred_proba'].min():.3f}, max={pred_df['pred_proba'].max():.3f}, mean={pred_df['pred_proba'].mean():.3f}")
        else:
            print(f"❌ pred_proba column missing")
            
    else:
        print(f"❌ Predictions file not created")
    
except Exception as e:
    print(f"❌ predict.py test failed: {e}")
    import traceback
    traceback.print_exc()

print(f"\n🎉 Multi-class test completed!")
