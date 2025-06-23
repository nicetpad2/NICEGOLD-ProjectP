#!/usr/bin/env python3
"""
🚀 Quick Production Fix Runner
รันการแก้ไขแบบด่วนสำหรับปัญหาที่พบใน pipeline

Usage:
    python quick_production_fix.py
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    print("🚀 Quick Production Fix for Trading System")
    print("=" * 50)
    
    try:
        # Step 1: Fix target values in preprocessed data
        logger.info("Step 1: Fixing target values...")
        
        data_path = "output_default/preprocessed_super.parquet"
        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            print("❌ Data file not found!")
            return
        
        # Load data
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded data: {df.shape}")
        
        # Fix target values - BINARY ONLY (0, 1)
        if 'target' in df.columns:
            original_dist = df['target'].value_counts().to_dict()
            logger.info(f"Original target distribution: {original_dist}")
            
            # Convert ALL values to binary (0 or 1)
            def to_binary(val):
                try:
                    val = float(val)
                    return 1 if val > 0 else 0
                except:
                    return 0
            
            df['target'] = df['target'].apply(to_binary)
            
            final_dist = df['target'].value_counts().to_dict()
            logger.info(f"Fixed target distribution: {final_dist}")
            print(f"✅ Target values fixed: {final_dist}")
            
            # Check for extreme imbalance
            if len(final_dist) >= 2:
                class_0 = final_dist.get(0, 0)
                class_1 = final_dist.get(1, 0)
                if class_1 > 0:
                    ratio = class_0 / class_1
                    print(f"Class imbalance ratio: {ratio:.2f}:1")
                    
                    if ratio > 100:
                        print("⚠️ EXTREME IMBALANCE DETECTED!")
                        # Create more balanced samples
                        n_minority = min(class_1 * 10, class_0 // 10)  # 10:1 max ratio
                        if n_minority > class_1:
                            # Oversample minority class
                            minority_indices = df[df['target'] == 1].index
                            oversample_indices = np.random.choice(
                                minority_indices, 
                                size=n_minority - class_1, 
                                replace=True
                            )
                            oversample_df = df.loc[oversample_indices].copy()
                            df = pd.concat([df, oversample_df], ignore_index=True)
                            
                            new_dist = df['target'].value_counts().to_dict()
                            new_ratio = new_dist[0] / new_dist[1]
                            print(f"✅ Balanced to {new_ratio:.2f}:1 ratio")
        
        # Step 2: Fix datetime columns
        logger.info("Step 2: Fixing datetime columns...")
        
        datetime_columns = []
        for col in df.columns:
            if col.lower() == 'target':
                continue
                
            if df[col].dtype == 'object':
                # Check if it's datetime-like
                sample = df[col].dropna().head(5)
                if len(sample) > 0:
                    sample_str = str(sample.iloc[0])
                    if any(char in sample_str for char in ['-', ':', '/', ' ']) and len(sample_str) > 8:
                        logger.info(f"Converting datetime column: {col}")
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                            df[col] = df[col].astype('int64', errors='ignore') // 10**9
                            datetime_columns.append(col)
                        except:
                            logger.warning(f"Failed to convert {col}, removing it")
                            df = df.drop(columns=[col])
                    else:
                        # Try numeric conversion
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            df[col] = df[col].fillna(0)
                        except:
                            logger.warning(f"Removing problematic column: {col}")
                            df = df.drop(columns=[col])
        
        print(f"✅ Converted {len(datetime_columns)} datetime columns")
        
        # Step 3: Save fixed data
        logger.info("Step 3: Saving fixed data...")
        
        os.makedirs("fixes", exist_ok=True)
        fixed_path = "fixes/preprocessed_super_fixed.parquet"
        df.to_parquet(fixed_path)
        
        print(f"✅ Fixed data saved: {fixed_path}")
        
        # Step 4: Test model training
        logger.info("Step 4: Testing model training...")
        
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import roc_auc_score, classification_report
            
            # Select features
            features = [col for col in df.columns if col != 'target']
            numeric_features = df[features].select_dtypes(include=[np.number]).columns.tolist()
            
            X = df[numeric_features].fillna(0)
            y = df['target']
            
            if len(X) > 100 and len(numeric_features) > 0:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Train model with class weights
                model = RandomForestClassifier(
                    n_estimators=50, 
                    random_state=42,
                    class_weight='balanced',
                    max_depth=10
                )
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                auc_score = roc_auc_score(y_test, y_pred_proba)
                
                y_pred = model.predict(X_test)
                
                print(f"✅ Model Test Results:")
                print(f"   Features used: {len(numeric_features)}")
                print(f"   Training samples: {len(X_train)}")
                print(f"   Test AUC: {auc_score:.4f}")
                
                # Classification report
                report = classification_report(y_test, y_pred)
                print("   Classification Report:")
                for line in report.split('\n'):
                    if line.strip():
                        print(f"   {line}")
                
                if auc_score > 0.6:
                    print("🚀 SUCCESS: Model training works! AUC > 0.6")
                else:
                    print("⚠️ WARNING: Low AUC score, needs more tuning")
                
            else:
                print("⚠️ Insufficient data for model testing")
        
        except Exception as e:
            logger.error(f"Model testing failed: {e}")
            print(f"❌ Model test failed: {e}")
        
        # Step 5: Create production config
        logger.info("Step 5: Creating production config...")
        
        config = {
            "timestamp": datetime.now().isoformat(),
            "data_fixes": {
                "target_conversion": "binary_0_1",
                "datetime_columns_converted": len(datetime_columns),
                "final_shape": list(df.shape),
                "final_target_distribution": df['target'].value_counts().to_dict()
            },
            "next_steps": [
                "python ProjectP.py (select mode 7)",
                "python run_ultimate_pipeline.py",
                "Monitor results in logs/"
            ]
        }
        
        import json
        config_path = "fixes/production_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Production config saved: {config_path}")
        
        print("\n🎉 PRODUCTION FIX COMPLETED!")
        print("\nNext steps:")
        print("1. python ProjectP.py (select mode 7 for Ultimate Pipeline)")
        print("2. python run_ultimate_pipeline.py (direct execution)")
        print("3. Monitor results in logs/ and fixes/ directories")
        
        return True
        
    except Exception as e:
        logger.error(f"Production fix failed: {e}")
        print(f"❌ Fix failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
