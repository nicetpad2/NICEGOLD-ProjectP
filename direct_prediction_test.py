#!/usr/bin/env python3
"""
Direct test of prediction logic to verify fix
"""

import os
import pandas as pd
import joblib

print("🔧 Direct Prediction Test...")

try:
    # Load model
    model_path = os.path.join("output_default", "catboost_model.pkl")
    model = joblib.load(model_path)
    print(f"✅ Model loaded: {type(model)}")

    # Load data
    data_path = os.path.join("output_default", "preprocessed_super.parquet")
    df = pd.read_parquet(data_path)
    print(f"✅ Data loaded: {df.shape}")

    # Apply same logic as predict.py
    df.columns = [c.lower() for c in df.columns]  # Lowercase all columns
    print(f"✅ Columns lowercased: {list(df.columns)[:5]}...")

    # Load features from train_features.txt
    features_path = os.path.join("output_default", "train_features.txt")
    with open(features_path, "r", encoding="utf-8") as f:
        original_features = [line.strip() for line in f if line.strip()]
    
    print(f"✅ Model expects features: {original_features}")

    # Map data columns to model features
    for orig_feat in original_features:
        lower_feat = orig_feat.lower()
        if lower_feat in df.columns and orig_feat not in df.columns:
            df.rename(columns={lower_feat: orig_feat}, inplace=True)
            print(f"✅ Renamed '{lower_feat}' -> '{orig_feat}'")

    # Verify all features are available
    missing_features = [f for f in original_features if f not in df.columns]
    if missing_features:
        print(f"❌ Missing features: {missing_features}")
        exit(1)
    else:
        print(f"✅ All features available: {original_features}")

    # Test prediction on small sample
    test_data = df[original_features].head(10)
    test_data = test_data.fillna(0)
    
    print(f"✅ Test data shape: {test_data.shape}")
    print(f"✅ Test data columns: {list(test_data.columns)}")

    # Predict
    pred_proba = model.predict_proba(test_data)
    print(f"✅ Prediction successful!")
    print(f"✅ Output shape: {pred_proba.shape}")
    print(f"✅ Sample probabilities: {pred_proba[:3, 1] if pred_proba.shape[1] > 1 else pred_proba[:3]}")

    print("🎉 Direct prediction test PASSED!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
