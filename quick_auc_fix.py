#!/usr/bin/env python3
"""
Quick AUC Fix Script
รันสคริปต์นี้เพื่อแก้ไข AUC ต่ำใน Production system

Usage: python quick_auc_fix.py
"""

import sys
import os
sys.path.append('.')

from fixes.target_variable_fix import create_improved_target
from fixes.feature_engineering_fix import create_high_predictive_features
from fixes.model_hyperparameters_fix import OPTIMIZED_CATBOOST_PARAMS, ENSEMBLE_CONFIG
from fixes.class_imbalance_fix import handle_class_imbalance

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import json

def quick_fix_pipeline():
    """รัน quick fix pipeline"""
    print("🚀 Starting Quick AUC Fix...")
    
    # 1. Load data
    print("📊 Loading data...")
    try:
        df = pd.read_csv("XAUUSD_M1.csv", nrows=10000)
    except FileNotFoundError:
        print("❌ XAUUSD_M1.csv not found")
        return
    
    # 2. Create improved features
    print("🔧 Creating improved features...")
    df_enhanced = create_high_predictive_features(df)
    
    # 3. Create improved target
    print("🎯 Creating improved target...")
    df_enhanced['target'] = create_improved_target(df_enhanced, method="multi_horizon_return")
    
    # 4. Prepare data
    feature_cols = [col for col in df_enhanced.columns 
                   if col not in ['target'] and df_enhanced[col].dtype in ['float64', 'int64']]
    
    X = df_enhanced[feature_cols].fillna(0)
    y = df_enhanced['target'].fillna(0)
    
    # Remove rows where target is NaN
    valid_idx = ~y.isna()
    X, y = X[valid_idx], y[valid_idx]
    
    print(f"📈 Features: {len(feature_cols)}, Samples: {len(X)}")
    print(f"📊 Target distribution: {y.value_counts().to_dict()}")
    
    # 5. Handle class imbalance
    print("⚖️ Handling class imbalance...")
    class_weights = handle_class_imbalance(X, y, method="class_weights")
    
    # 6. Train improved model
    print("🤖 Training improved model...")
    try:
        from catboost import CatBoostClassifier
        
        model = CatBoostClassifier(
            **OPTIMIZED_CATBOOST_PARAMS,
            class_weights=list(class_weights.values())
        )
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        auc_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            auc = roc_auc_score(y_val, y_pred_proba)
            auc_scores.append(auc)
            print(f"   Fold AUC: {auc:.3f}")
        
        final_auc = np.mean(auc_scores)
        print(f"\n🎉 Final Average AUC: {final_auc:.3f}")
        print(f"📈 Improvement: {((final_auc - 0.516) / 0.516 * 100):.1f}%")
        
        if final_auc > 0.65:
            print("✅ SUCCESS: AUC significantly improved!")
        else:
            print("⚠️ Partial improvement. Consider advanced techniques.")
            
        # Save results
        results = {
            "original_auc": 0.516,
            "improved_auc": final_auc,
            "improvement_pct": ((final_auc - 0.516) / 0.516 * 100),
            "cv_scores": auc_scores,
            "status": "success" if final_auc > 0.65 else "partial"
        }
        
        with open("fixes/quick_fix_results.json", "w") as f:
            json.dump(results, f, indent=2)
            
        return final_auc
        
    except ImportError:
        print("❌ CatBoost not installed. Please install: pip install catboost")
        return None
    
if __name__ == "__main__":
    final_auc = quick_fix_pipeline()
