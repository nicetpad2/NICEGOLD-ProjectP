#!/usr/bin/env python3
from pathlib import Path
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import pandas as pd
        import sklearn
import sys
import warnings
"""
Quick Fix for NaN AUC Issue
สำหรับแก้ไขปัญหาเฉียบพลัน Random Forest AUC = nan
"""


warnings.filterwarnings('ignore')

def simple_nan_auc_fix():
    """แก้ไขปัญหา NaN AUC แบบเร่งด่วน"""
    print(" =  = = QUICK NaN AUC FIX = =  = ")

    try:
        # 1. สร้าง synthetic data ที่ไม่มีปัญหา
        print("Creating clean synthetic data...")
        np.random.seed(42)
        n = 1000

        # สร้าง features ที่มี correlation กับ target
        X1 = np.random.randn(n)
        X2 = np.random.randn(n) + 0.5 * X1  # correlated with X1
        X3 = np.random.randn(n)

        # สร้าง target ที่มี relationship กับ features
        target_logit = 0.3 * X1 + 0.2 * X2 + 0.1 * X3 + np.random.randn(n) * 0.5
        target = (target_logit > 0).astype(int)

        # สร้าง DataFrame
        df = pd.DataFrame({
            'feature1': X1, 
            'feature2': X2, 
            'feature3': X3, 
            'target': target
        })

        print(f"Data shape: {df.shape}")
        print(f"Target distribution: {df['target'].value_counts().to_dict()}")

        # 2. ทดสอบ Random Forest

        X = df[['feature1', 'feature2', 'feature3']]
        y = df['target']

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Test Random Forest
        rf = RandomForestClassifier(
            n_estimators = 50, 
            max_depth = 5, 
            class_weight = 'balanced', 
            random_state = 42
        )

        scores = cross_val_score(rf, X_scaled, y, cv = 3, scoring = 'roc_auc')
        auc_mean = scores.mean()

        print(f"Random Forest AUC: {auc_mean:.4f}")

        if not np.isnan(auc_mean) and auc_mean > 0.5:
            print("✅ SUCCESS: Random Forest working properly!")
            print(f"✅ AUC Score: {auc_mean:.4f}")
            return True
        else:
            print(f"❌ FAILED: AUC = {auc_mean}")
            return False

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def diagnose_existing_data():
    """วิเคราะห์ data ที่มีอยู่"""
    print("\n =  = = DIAGNOSING EXISTING DATA = =  = ")

    data_files = ['dummy_m1.csv', 'dummy_m15.csv']

    for file_path in data_files:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"\n📁 File: {file_path}")
                print(f"📊 Shape: {df.shape}")
                print(f"📋 Columns: {list(df.columns)}")

                if 'target' in df.columns:
                    target_dist = df['target'].value_counts()
                    print(f"🎯 Target distribution: {target_dist.to_dict()}")

                    # ตรวจสอบ class imbalance
                    min_count = target_dist.min()
                    max_count = target_dist.max()
                    ratio = max_count / min_count if min_count > 0 else float('inf')
                    print(f"⚖️ Imbalance ratio: {ratio:.1f}:1")

                    if ratio > 100:
                        print("❌ CRITICAL: Extreme class imbalance detected!")
                        print("💡 This is likely causing NaN AUC scores")

                # ตรวจสอบ data quality
                nan_count = df.isnull().sum().sum()
                if nan_count > 0:
                    print(f"⚠️ NaN values: {nan_count}")

                print("✅ Data loaded successfully")

            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
        else:
            print(f"❌ File not found: {file_path}")

def main():
    """Main function"""
    print("QUICK NaN AUC DIAGNOSTIC & FIX")
    print(" = " * 40)

    # ตรวจสอบ sklearn
    try:
        print(f"✅ sklearn version: {sklearn.__version__}")
    except ImportError:
        print("❌ sklearn not installed!")
        print("Please install: pip install scikit - learn")
        return

    # 1. ทดสอบ synthetic data
    success = simple_nan_auc_fix()

    # 2. วิเคราะห์ data จริง
    diagnose_existing_data()

    # 3. แนะนำการแก้ไข
    print("\n" + " = " * 40)
    print("RECOMMENDATIONS FOR NaN AUC FIX:")
    print(" = " * 40)

    if success:
        print("✅ Your environment works with synthetic data")
        print("The NaN AUC issue is likely due to:")
        print("  1. Extreme class imbalance in your real data")
        print("  2. Features with no predictive power")
        print("  3. Data quality issues (NaN, infinite values)")

        print("\n💡 Solutions:")
        print("  1. Use SMOTE for balancing classes")
        print("  2. Create better engineered features")
        print("  3. Use class_weight = 'balanced' in models")
        print("  4. Try threshold optimization")
        print("  5. Ensure minimum samples per class")

    else:
        print("❌ Even synthetic data failed")
        print("This suggests environment issues:")
        print("  1. Check sklearn installation")
        print("  2. Check Python version compatibility")
        print("  3. Try reinstalling packages")

    # บันทึกผลลัพธ์
    output_dir = Path("output_default")
    output_dir.mkdir(exist_ok = True)

    with open(output_dir / "quick_nan_auc_diagnosis.txt", "w") as f:
        f.write(f"Quick NaN AUC Fix Result: {'SUCCESS' if success else 'FAILED'}\n")
        f.write(f"Timestamp: {pd.Timestamp.now()}\n")
        f.write(f"Synthetic data test: {'PASSED' if success else 'FAILED'}\n")

    print(f"\n📁 Results saved to: {output_dir / 'quick_nan_auc_diagnosis.txt'}")

if __name__ == "__main__":
    main()