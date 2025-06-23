#!/usr/bin/env python3
"""
Test feature names fix for prediction
"""

import pandas as pd
import joblib
import os

print("🔧 Testing Feature Names Fix...")

# Load the model
model_path = os.path.join("output_default", "catboost_model.pkl")
if not os.path.exists(model_path):
    print(f"❌ Model not found: {model_path}")
    exit(1)

model = joblib.load(model_path)
print(f"✅ Model loaded: {type(model)}")

# Check what features the model expects
if hasattr(model, 'feature_names_'):
    model_features = model.feature_names_
    print(f"✅ Model expects features: {model_features}")
elif hasattr(model, 'feature_importances_') and hasattr(model, 'n_features_'):
    print(f"✅ Model has {model.n_features_} features (no feature_names_ available)")
    model_features = None
else:
    print("⚠️ Cannot determine model features")
    model_features = None

# Load the train_features.txt file
features_path = os.path.join("output_default", "train_features.txt")
if os.path.exists(features_path):
    with open(features_path, "r", encoding="utf-8") as f:
        feature_list = [line.strip() for line in f if line.strip()]
    print(f"✅ train_features.txt contains: {feature_list}")
else:
    print(f"❌ Features file not found: {features_path}")
    exit(1)

# Load some test data
data_path = os.path.join("output_default", "preprocessed_super.parquet")
if os.path.exists(data_path):
    df = pd.read_parquet(data_path)
    df.columns = [c.lower() for c in df.columns]  # Lowercase columns as done in predict.py
    print(f"✅ Data loaded: {df.shape}")
    print(f"✅ Data columns: {list(df.columns)}")
    
    # Apply the same feature mapping logic as in predict.py
    original_features = feature_list
    
    for orig_feat in original_features:
        lower_feat = orig_feat.lower()
        if lower_feat in df.columns and orig_feat not in df.columns:
            df.rename(columns={lower_feat: orig_feat}, inplace=True)
            print(f"✅ Renamed '{lower_feat}' -> '{orig_feat}'")
        elif orig_feat not in df.columns:
            df[orig_feat] = float('nan')
            print(f"⚠️ Added missing feature '{orig_feat}' with NaN")
    
    # Check if we have all features needed
    missing_features = [f for f in original_features if f not in df.columns]
    if missing_features:
        print(f"❌ Missing features: {missing_features}")
    else:
        print(f"✅ All features available: {original_features}")
    
    # Try prediction on a small sample
    try:
        test_data = df[original_features].head(5)
        test_data = test_data.fillna(0)  # Fill NaN for testing
        
        pred_proba = model.predict_proba(test_data)
        print(f"✅ Prediction successful! Shape: {pred_proba.shape}")
        print(f"✅ Sample probabilities: {pred_proba[:3, 1] if pred_proba.shape[1] > 1 else pred_proba[:3]}")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
else:
    print(f"❌ Data file not found: {data_path}")

print("🎉 Feature names test completed!")
