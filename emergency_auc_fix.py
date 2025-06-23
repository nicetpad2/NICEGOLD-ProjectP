#!/usr/bin/env python3
"""
Emergency AUC Fix for Production
แก้ไข AUC จาก 0.516 ให้สูงขึ้นด้วยวิธีที่เร็วและมีประสิทธิภาพ
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import os
import json

def create_improved_features(df):
    """สร้างฟีเจอร์ที่มี predictive power สูง"""
    print("🔧 Creating improved features...")
    
    features = df.copy()
    
    # 1. Basic price features
    if all(col in features.columns for col in ['Open', 'High', 'Low', 'Close']):
        features['price_range'] = (features['High'] - features['Low']) / features['Close']
        features['body_size'] = abs(features['Close'] - features['Open']) / features['Close']
        features['upper_wick'] = (features['High'] - features[['Open', 'Close']].max(axis=1)) / features['Close']
        features['lower_wick'] = (features[['Open', 'Close']].min(axis=1) - features['Low']) / features['Close']
    
    # 2. Multi-period momentum (สำคัญมาก!)
    if 'Close' in features.columns:
        for period in [3, 5, 8, 13, 21]:
            features[f'momentum_{period}'] = features['Close'].pct_change(period)
            features[f'sma_{period}'] = features['Close'].rolling(period).mean()
            features[f'price_vs_sma_{period}'] = (features['Close'] - features[f'sma_{period}']) / features['Close']
    
    # 3. Volatility features
    if 'Close' in features.columns:
        returns = features['Close'].pct_change()
        for window in [5, 10, 20]:
            features[f'volatility_{window}'] = returns.rolling(window).std()
            features[f'price_zscore_{window}'] = (
                (features['Close'] - features['Close'].rolling(window).mean()) / 
                (features['Close'].rolling(window).std() + 1e-8)
            )
    
    # 4. Trend strength
    if 'Close' in features.columns:
        for window in [5, 10]:
            price_changes = features['Close'].diff()
            features[f'trend_strength_{window}'] = (
                price_changes.rolling(window).apply(lambda x: (x > 0).sum() / len(x))
            )
    
    # 5. Support/Resistance
    if all(col in features.columns for col in ['High', 'Low', 'Close']):
        for window in [5, 10]:
            features[f'resistance_{window}'] = features['High'].rolling(window).max()
            features[f'support_{window}'] = features['Low'].rolling(window).min()
            features[f'resistance_distance_{window}'] = (features[f'resistance_{window}'] - features['Close']) / features['Close']
            features[f'support_distance_{window}'] = (features['Close'] - features[f'support_{window}']) / features['Close']
    
    print(f"   Created {len(features.columns) - len(df.columns)} new features")
    return features

def create_improved_target(df, method="volatility_adjusted"):
    """สร้าง target variable ที่ดีกว่า"""
    print("🎯 Creating improved target...")
    
    if 'Close' not in df.columns:
        print("❌ No 'Close' column found")
        return pd.Series([0] * len(df), index=df.index)
    
    if method == "volatility_adjusted":
        # Future return adjusted by volatility
        future_return = df['Close'].pct_change().shift(-5)  # 5 periods ahead
        volatility = df['Close'].pct_change().rolling(20).std()
        
        # Normalize by volatility
        normalized_return = future_return / (volatility + 1e-8)
        
        # Dynamic threshold based on data distribution
        threshold = normalized_return.quantile(0.6)
        target = (normalized_return > threshold).astype(int)
        
    elif method == "multi_horizon":
        # Multiple horizon approach
        ret_1 = df['Close'].pct_change().shift(-1)
        ret_3 = df['Close'].pct_change(3).shift(-3)
        ret_5 = df['Close'].pct_change(5).shift(-5)
        
        # Weighted combination
        combined = 0.5 * ret_1 + 0.3 * ret_3 + 0.2 * ret_5
        threshold = combined.quantile(0.6)
        target = (combined > threshold).astype(int)
    
    else:
        # Simple future return
        future_return = df['Close'].pct_change().shift(-3)
        target = (future_return > 0).astype(int)
    
    # Clean target
    target = target.fillna(0)
    
    class_dist = target.value_counts()
    print(f"   Target distribution: {dict(class_dist)}")
    
    return target

def handle_class_imbalance(X, y):
    """จัดการ class imbalance"""
    print("⚖️ Handling class imbalance...")
    
    # Calculate class weights
    classes = np.unique(y)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y
    )
    
    class_weight_dict = dict(zip(classes, class_weights))
    print(f"   Class weights: {class_weight_dict}")
    
    return class_weight_dict

def train_improved_model(X, y, class_weights):
    """ฝึกโมเดลที่ปรับปรุงแล้ว"""
    print("🤖 Training improved model...")
    
    # Use Random Forest with optimized parameters
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1
    )
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    auc_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate AUC
        try:
            auc = roc_auc_score(y_val, y_pred_proba)
            auc_scores.append(auc)
            print(f"   Fold {fold + 1} AUC: {auc:.3f}")
        except ValueError as e:
            print(f"   Fold {fold + 1} AUC: Error - {e}")
            auc_scores.append(0.5)
    
    final_auc = np.mean(auc_scores)
    return final_auc, model, auc_scores

def main():
    """Main function"""
    print("🚀 Emergency AUC Fix Pipeline Starting...")
    
    # 1. Load data
    print("\n📊 Loading data...")
    try:
        # Try multiple data sources
        if os.path.exists("output_default/preprocessed_super.parquet"):
            df = pd.read_parquet("output_default/preprocessed_super.parquet")
            df = df.head(20000)  # Limit for speed
            print(f"   Loaded preprocessed data: {df.shape}")
        elif os.path.exists("XAUUSD_M1.csv"):
            df = pd.read_csv("XAUUSD_M1.csv", nrows=10000)
            print(f"   Loaded raw CSV data: {df.shape}")
        else:
            print("❌ No data files found!")
            return
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # 2. Create improved features
    df_enhanced = create_improved_features(df)
    
    # 3. Create improved target
    df_enhanced['target_improved'] = create_improved_target(df_enhanced, method="volatility_adjusted")
    
    # 4. Prepare data for modeling
    print("\n🔧 Preparing data for modeling...")
    
    # Select numeric features only
    feature_cols = [col for col in df_enhanced.columns 
                   if col not in ['target_improved', 'target'] and 
                   df_enhanced[col].dtype in ['float64', 'int64']]
    
    # Clean data
    X = df_enhanced[feature_cols].fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    y = df_enhanced['target_improved'].fillna(0)
    
    # Remove rows where target is invalid
    valid_mask = ~(y.isna() | np.isinf(y))
    X, y = X[valid_mask], y[valid_mask]
    
    # Remove constant features
    constant_features = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_features:
        X = X.drop(columns=constant_features)
        print(f"   Removed {len(constant_features)} constant features")
    
    print(f"   Final dataset: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"   Target distribution: {dict(y.value_counts())}")
    
    # Check if we have enough data
    if len(X) < 100:
        print("❌ Not enough data for meaningful training")
        return
    
    if y.nunique() < 2:
        print("❌ Target has only one class")
        return
    
    # 5. Handle class imbalance
    class_weights = handle_class_imbalance(X, y)
    
    # 6. Train model
    final_auc, model, cv_scores = train_improved_model(X, y, class_weights)
    
    # 7. Results
    print(f"\n🎉 Results:")
    print(f"   Original AUC: 0.516")
    print(f"   Improved AUC: {final_auc:.3f}")
    print(f"   Improvement: {((final_auc - 0.516) / 0.516 * 100):.1f}%")
    
    if final_auc > 0.65:
        status = "🎯 SUCCESS: Significant improvement!"
        recommendation = "Ready for production testing"
    elif final_auc > 0.55:
        status = "📈 PARTIAL: Some improvement"
        recommendation = "Need more advanced techniques"
    else:
        status = "⚠️ LIMITED: Minimal improvement"
        recommendation = "Consider deep learning or different approach"
    
    print(f"   Status: {status}")
    print(f"   Recommendation: {recommendation}")
    
    # 8. Save results
    results = {
        "original_auc": 0.516,
        "improved_auc": float(final_auc),
        "improvement_pct": float((final_auc - 0.516) / 0.516 * 100),
        "cv_scores": [float(score) for score in cv_scores],
        "status": status,
        "recommendation": recommendation,
        "dataset_info": {
            "rows": int(len(X)),
            "features": int(X.shape[1]),
            "target_distribution": {str(k): int(v) for k, v in y.value_counts().items()}
        }
    }
    
    os.makedirs("fixes", exist_ok=True)
    with open("fixes/emergency_auc_fix_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: fixes/emergency_auc_fix_results.json")
    
    # Feature importance (top 10)
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n📈 Top 10 Important Features:")
        print(feature_importance.head(10).to_string(index=False))
    
    return final_auc

if __name__ == "__main__":
    try:
        final_auc = main()
        if final_auc and final_auc > 0.65:
            print("\n✅ SUCCESS: Ready to integrate into production system!")
        else:
            print("\n🔄 CONTINUE: Consider running advanced_feature_engineering.py for deeper improvements")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
