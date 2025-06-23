#!/usr/bin/env python3
"""
🚨 CRITICAL NaN AUC FIX 
Advanced fix for extreme class imbalance causing NaN AUC scores
Addresses: 201.7:1 imbalance, low feature correlation, model convergence issues
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import json
from datetime import datetime

warnings.filterwarnings('ignore')

def critical_data_fixes(df, target_col='target'):
    """แก้ไขปัญหา data แบบรุนแรง"""
    print("🛠️ Applying critical data fixes...")
    
    df_fixed = df.copy()
    
    # 1. ทำความสะอาด data อย่างรุนแรง
    feature_cols = [col for col in df_fixed.columns if col != target_col]
    
    # แทนที่ทุก NaN และ infinite values
    for col in feature_cols:
        if df_fixed[col].dtype in ['int64', 'float64']:
            median_val = df_fixed[col].median()
            df_fixed[col] = df_fixed[col].fillna(median_val)
            df_fixed[col] = df_fixed[col].replace([np.inf, -np.inf], median_val)
            
            # ลบ outliers รุนแรง (นอก 3 sigma)
            mean_val = df_fixed[col].mean()
            std_val = df_fixed[col].std()
            if std_val > 0:
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val
                df_fixed[col] = df_fixed[col].clip(lower_bound, upper_bound)
    
    print(f"✅ Data cleaning completed: {df_fixed.shape}")
    
    # 2. สร้าง synthetic minority class samples (AGGRESSIVE)
    target_counts = df_fixed[target_col].value_counts()
    print(f"📊 Original distribution: {target_counts.to_dict()}")
    
    # หาคลาสที่มี samples น้อยที่สุด
    minority_classes = target_counts[target_counts < target_counts.max() / 50]  # < 2% ของ majority
    
    if len(minority_classes) > 0:
        print(f"🎯 Creating synthetic samples for classes: {list(minority_classes.index)}")
        
        balanced_dfs = []
        target_samples_per_class = max(500, target_counts.max() // 100)  # อย่างน้อย 500 หรือ 1% ของ majority
        
        for class_val in target_counts.index:
            class_df = df_fixed[df_fixed[target_col] == class_val]
            
            if len(class_df) < target_samples_per_class:
                # สร้าง synthetic samples แบบรุนแรง
                n_needed = target_samples_per_class - len(class_df)
                print(f"  Creating {n_needed} samples for class {class_val}")
                
                synthetic_samples = []
                base_samples = class_df[feature_cols].values
                
                for _ in range(n_needed):
                    if len(base_samples) > 1:
                        # เลือก 2 samples แบบสุ่ม และทำ interpolation
                        idx1, idx2 = np.random.choice(len(base_samples), 2, replace=True)
                        alpha = np.random.uniform(0.2, 0.8)  # interpolation factor
                        
                        new_sample = alpha * base_samples[idx1] + (1 - alpha) * base_samples[idx2]
                        
                        # เพิ่ม noise เล็กน้อย
                        noise = np.random.normal(0, 0.01, size=new_sample.shape)
                        new_sample += noise
                        
                    else:
                        # ถ้ามี sample เดียว ให้เพิ่ม noise มากขึ้น
                        new_sample = base_samples[0] + np.random.normal(0, 0.05, size=base_samples[0].shape)
                    
                    # สร้าง DataFrame สำหรับ sample ใหม่
                    new_row = {}
                    for i, col in enumerate(feature_cols):
                        new_row[col] = new_sample[i]
                    new_row[target_col] = class_val
                    
                    synthetic_samples.append(new_row)
                
                # รวม synthetic samples เข้าไป
                if synthetic_samples:
                    synthetic_df = pd.DataFrame(synthetic_samples)
                    class_df = pd.concat([class_df, synthetic_df], ignore_index=True)
            
            balanced_dfs.append(class_df)
        
        # รวม balanced data
        df_balanced = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        
        final_counts = df_balanced[target_col].value_counts()
        print(f"✅ Balanced distribution: {final_counts.to_dict()}")
        
        return df_balanced
    
    return df_fixed

def create_powerful_features(df, target_col='target'):
    """สร้าง features ที่มี predictive power สูง"""
    print("🔧 Creating powerful engineered features...")
    
    df_enhanced = df.copy()
    feature_cols = [col for col in df.columns if col != target_col and df[col].dtype in ['int64', 'float64']]
    
    if len(feature_cols) < 2:
        print("⚠️ Not enough numeric features for engineering")
        return df_enhanced
    
    # 1. Moving statistics (มากขึ้น)
    for col in feature_cols[:5]:  # จำกัดไว้ 5 features แรกเพื่อประสิทธิภาพ
        # Rolling statistics
        df_enhanced[f'{col}_ma5'] = df_enhanced[col].rolling(5, min_periods=1).mean()
        df_enhanced[f'{col}_ma10'] = df_enhanced[col].rolling(10, min_periods=1).mean() 
        df_enhanced[f'{col}_std5'] = df_enhanced[col].rolling(5, min_periods=1).std().fillna(0)
        
        # Exponential moving average
        df_enhanced[f'{col}_ema'] = df_enhanced[col].ewm(span=5, adjust=False).mean()
        
        # Lag features
        df_enhanced[f'{col}_lag1'] = df_enhanced[col].shift(1).fillna(df_enhanced[col].mean())
        df_enhanced[f'{col}_lag2'] = df_enhanced[col].shift(2).fillna(df_enhanced[col].mean())
        
        # Change features
        df_enhanced[f'{col}_change'] = df_enhanced[col].diff().fillna(0)
        df_enhanced[f'{col}_pct_change'] = df_enhanced[col].pct_change().fillna(0).replace([np.inf, -np.inf], 0)
    
    # 2. Cross features (combinations)
    if len(feature_cols) >= 2:
        for i in range(min(3, len(feature_cols))):
            for j in range(i+1, min(4, len(feature_cols))):
                col1, col2 = feature_cols[i], feature_cols[j]
                
                # Mathematical combinations
                df_enhanced[f'{col1}_{col2}_sum'] = df_enhanced[col1] + df_enhanced[col2]
                df_enhanced[f'{col1}_{col2}_diff'] = df_enhanced[col1] - df_enhanced[col2]
                df_enhanced[f'{col1}_{col2}_product'] = df_enhanced[col1] * df_enhanced[col2]
                
                # Safe division
                df_enhanced[f'{col1}_{col2}_ratio'] = df_enhanced[col1] / (df_enhanced[col2] + 1e-8)
                
                # Distance features
                df_enhanced[f'{col1}_{col2}_euclidean'] = np.sqrt(df_enhanced[col1]**2 + df_enhanced[col2]**2)
    
    # 3. Statistical features
    numeric_cols = [col for col in df_enhanced.columns if col != target_col and df_enhanced[col].dtype in ['int64', 'float64']]
    
    # Row-wise statistics
    df_enhanced['row_mean'] = df_enhanced[numeric_cols].mean(axis=1)
    df_enhanced['row_std'] = df_enhanced[numeric_cols].std(axis=1).fillna(0)
    df_enhanced['row_min'] = df_enhanced[numeric_cols].min(axis=1)
    df_enhanced['row_max'] = df_enhanced[numeric_cols].max(axis=1)
    df_enhanced['row_range'] = df_enhanced['row_max'] - df_enhanced['row_min']
    
    # Ranking features
    for col in feature_cols[:3]:
        df_enhanced[f'{col}_rank'] = df_enhanced[col].rank(pct=True)
        df_enhanced[f'{col}_quantile'] = pd.qcut(df_enhanced[col], q=10, labels=False, duplicates='drop')
    
    print(f"✅ Feature engineering completed: {df.shape} → {df_enhanced.shape}")
    
    return df_enhanced

def robust_model_testing(df, target_col='target'):
    """ทดสอบ models แบบ robust"""
    print("🧪 Running robust model testing...")
    
    try:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.preprocessing import StandardScaler, RobustScaler
        from sklearn.metrics import classification_report, roc_auc_score
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        print(f"❌ Missing sklearn modules: {e}")
        return {}
    
    # เตรียม data
    feature_cols = [col for col in df.columns if col != target_col and df[col].dtype in ['int64', 'float64']]
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"📊 Final data shape: X={X.shape}, y unique classes={len(y.unique())}")
    print(f"📊 Class distribution: {y.value_counts().to_dict()}")
    
    # Multiple scalers
    scalers = {
        'Standard': StandardScaler(),
        'Robust': RobustScaler(),
    }
    
    # Robust models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=50,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=2000,
            solver='liblinear',
            C=0.1
        )
    }
    
    results = {}
    
    # ทดสอบแต่ละ combination
    for scaler_name, scaler in scalers.items():
        print(f"\n🔄 Testing with {scaler_name} scaler...")
        
        try:
            X_scaled = scaler.fit_transform(X)
            
            # แบ่ง train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42, stratify=y
            )
            
            for model_name, model in models.items():
                key = f"{model_name} ({scaler_name})"
                print(f"  Testing {key}...")
                
                try:
                    # Fit model
                    model.fit(X_train, y_train)
                    
                    # Predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)
                    
                    # AUC calculation
                    if len(np.unique(y)) > 2:  # Multi-class
                        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
                    else:  # Binary
                        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                    
                    # Cross-validation
                    cv_folds = min(3, min(y.value_counts()))  # Safe CV
                    if cv_folds >= 2:
                        cv_scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='roc_auc')
                        cv_mean = cv_scores.mean()
                        cv_std = cv_scores.std()
                    else:
                        cv_mean = auc
                        cv_std = 0
                    
                    results[key] = {
                        'auc': auc,
                        'cv_mean': cv_mean,
                        'cv_std': cv_std,
                        'train_samples': len(y_train),
                        'test_samples': len(y_test),
                        'features': X.shape[1]
                    }
                    
                    print(f"    ✅ {key}: AUC = {auc:.4f}, CV = {cv_mean:.4f} ± {cv_std:.4f}")
                    
                except Exception as e:
                    print(f"    ❌ {key} failed: {e}")
                    results[key] = {'auc': np.nan, 'error': str(e)}
        
        except Exception as e:
            print(f"❌ Scaler {scaler_name} failed: {e}")
    
    return results

def main():
    """Main execution"""
    print("🚨 CRITICAL NaN AUC FIX - STARTING")
    print("=" * 60)
    
    output_dir = Path("output_default")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # 1. ลองโหลด data
        print("📁 Loading data...")
        
        data_files = ['dummy_m1.csv', 'dummy_m15.csv']
        df = None
        
        for file_path in data_files:
            if Path(file_path).exists():
                try:
                    df = pd.read_csv(file_path)
                    print(f"✅ Loaded {file_path}: {df.shape}")
                    break
                except Exception as e:
                    print(f"❌ Failed to load {file_path}: {e}")
        
        if df is None:
            print("🔧 Creating robust synthetic data...")
            # สร้าง synthetic data ที่ robust
            np.random.seed(42)
            n = 5000
            
            # สร้าง features ที่มี pattern
            features = {}
            
            # Base trend
            time_trend = np.linspace(0, 10, n)
            noise_level = 0.1
            
            # Technical indicators simulation
            features['close'] = 100 + np.cumsum(np.random.randn(n) * 0.5) + np.sin(time_trend) * 5
            features['volume'] = np.random.exponential(1000, n)
            features['rsi'] = 50 + 30 * np.sin(time_trend * 2) + np.random.randn(n) * 5
            features['macd'] = np.sin(time_trend * 0.5) + np.random.randn(n) * noise_level
            features['momentum'] = np.diff(features['close'], prepend=features['close'][0]) + np.random.randn(n) * noise_level
            features['volatility'] = np.abs(np.random.randn(n)) + 0.5
            
            df = pd.DataFrame(features)
            
            # สร้าง target ที่มี relationship แต่ imbalanced
            signal = (0.3 * (features['rsi'] - 50) / 50 + 
                     0.2 * features['macd'] + 
                     0.1 * features['momentum'] / features['momentum'].std() +
                     np.random.randn(n) * 0.5)
            
            # สร้าง extreme imbalance
            percentiles = np.percentile(signal, [95, 99])  # top 5% และ 1%
            df['target'] = np.where(signal > percentiles[1], 1,      # 1% positive
                          np.where(signal > percentiles[0], 0,       # 4% neutral  
                                  -1))                              # 95% negative
            
            print(f"✅ Created synthetic data: {df.shape}")
            print(f"📊 Target distribution: {df['target'].value_counts().to_dict()}")
        
        # 2. แก้ไขปัญหาข้อมูล
        df_fixed = critical_data_fixes(df)
        
        # 3. สร้าง powerful features
        df_enhanced = create_powerful_features(df_fixed)
        
        # 4. ทดสอบ models
        results = robust_model_testing(df_enhanced)
        
        # 5. บันทึกผลลัพธ์
        report = {
            'timestamp': datetime.now().isoformat(),
            'original_shape': df.shape,
            'fixed_shape': df_fixed.shape,
            'enhanced_shape': df_enhanced.shape,
            'model_results': results,
            'status': 'completed',
            'summary': {
                'successful_models': len([r for r in results.values() if not np.isnan(r.get('auc', np.nan))]),
                'total_models': len(results),
                'best_auc': max([r.get('auc', 0) for r in results.values() if not np.isnan(r.get('auc', np.nan))], default=0)
            }
        }
        
        # บันทึกไฟล์
        report_file = output_dir / 'critical_nan_auc_fix_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        data_file = output_dir / 'critical_fixed_data.csv'
        df_enhanced.to_csv(data_file, index=False)
        
        # สรุปผลลัพธ์
        print("\n" + "="*60)
        print("🎯 CRITICAL FIX RESULTS")
        print("="*60)
        
        successful_models = [name for name, result in results.items() 
                           if isinstance(result.get('auc'), (int, float)) and not np.isnan(result['auc'])]
        
        if successful_models:
            print("✅ SUCCESS: Fixed critical NaN AUC issue!")
            
            best_model = max(results.items(), key=lambda x: x[1].get('auc', 0) if not np.isnan(x[1].get('auc', np.nan)) else 0)
            best_auc = best_model[1]['auc']
            
            print(f"🏆 Best model: {best_model[0]} with AUC = {best_auc:.4f}")
            print(f"✅ {len(successful_models)}/{len(results)} models working")
            
            for name in successful_models[:3]:  # แสดง top 3
                auc = results[name]['auc']
                cv = results[name].get('cv_mean', 0)
                print(f"  🎯 {name}: AUC = {auc:.4f}, CV = {cv:.4f}")
        else:
            print("❌ CRITICAL: All models still failing!")
            print("🆘 This indicates fundamental issues:")
            print("  1. Environment/package problems")
            print("  2. Data corruption beyond repair")
            print("  3. Hardware/memory issues")
        
        print(f"\n📁 Results saved:")
        print(f"  📋 Report: {report_file}")
        print(f"  📊 Data: {data_file}")
        print(f"📈 Data pipeline: {df.shape} → {df_enhanced.shape}")
        
    except Exception as e:
        print(f"💥 CRITICAL FIX FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
