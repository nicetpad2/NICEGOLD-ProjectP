#!/usr/bin/env python3
"""
Quick diagnosis of feature names issue
"""

import os
import pandas as pd

print("🔧 Quick Feature Names Diagnosis...")

# 1. Check train_features.txt content
features_path = os.path.join("output_default", "train_features.txt")
if os.path.exists(features_path):
    with open(features_path, "r") as f:
        content = f.read()
    print(f"✅ train_features.txt content:\n{repr(content)}")
    
    feature_list = [line.strip() for line in content.split('\n') if line.strip()]
    print(f"✅ Parsed features: {feature_list}")
else:
    print(f"❌ Features file not found: {features_path}")

# 2. Check model file
model_path = os.path.join("output_default", "catboost_model.pkl")
print(f"✅ Model file exists: {os.path.exists(model_path)}")

# 3. Check data file and its columns
data_path = os.path.join("output_default", "preprocessed_super.parquet")
if os.path.exists(data_path):
    df = pd.read_parquet(data_path)
    print(f"✅ Data shape: {df.shape}")
    print(f"✅ Original columns: {list(df.columns)}")
    
    # Apply lowercase transformation
    df.columns = [c.lower() for c in df.columns]
    print(f"✅ Lowercase columns: {list(df.columns)}")
    
    # Check which features are available
    feature_list = ['Open', 'Volume', 'returns', 'volatility', 'momentum', 'rsi', 'macd']
    available = []
    missing = []
    
    for feat in feature_list:
        if feat.lower() in df.columns:
            available.append(feat)
        else:
            missing.append(feat)
    
    print(f"✅ Available features (can be mapped): {available}")
    print(f"❌ Missing features: {missing}")
else:
    print(f"❌ Data file not found: {data_path}")

print("🎉 Quick diagnosis completed!")
