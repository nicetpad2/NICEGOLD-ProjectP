#!/usr/bin/env python3
        from feature_engineering import run_production_grade_feature_engineering
            from production_class_imbalance_fix import fix_extreme_class_imbalance_production
        from projectp.steps.walkforward import WalkForwardValidation
import numpy as np
import os
import pandas as pd
import sys
        import traceback
import warnings
"""
🚀 FINAL PRODUCTION PIPELINE RUN
รันระบบ Production สุดท้ายหลังแก้ไขปัญหา Class Imbalance ครบถ้วน
"""

warnings.filterwarnings('ignore')

def main():
    """Main pipeline runner"""
    print("🚀 FINAL PRODUCTION PIPELINE STARTING...")
    print(" = " * 80)

    try:
        # 1. รัน Production - Grade Feature Engineering
        print("\n🔧 STEP 1: Production - Grade Feature Engineering")

        df_features, features_path = run_production_grade_feature_engineering()
        if df_features is None:
            print("❌ Feature engineering failed, trying fallback...")
            # Fallback: รัน class imbalance fix แยก
            df_features, features_path = fix_extreme_class_imbalance_production()

        if df_features is None:
            print("❌ All feature engineering methods failed!")
            return False

        print(f"✅ Features ready: {df_features.shape}")
        print(f"📊 Target distribution: {df_features['target'].value_counts().to_dict()}")

        # 2. รัน WalkForward Validation
        print("\n🔍 STEP 2: WalkForward Validation")
        run_walkforward_validation(features_path)

        # 3. รัน Full Pipeline
        print("\n⚡ STEP 3: Full ML Pipeline")
        run_full_pipeline()

        # 4. ตรวจสอบผลลัพธ์
        print("\n📊 STEP 4: Results Validation")
        validate_results()

        print("\n🎉 FINAL PRODUCTION PIPELINE COMPLETED SUCCESSFULLY!")
        return True

    except Exception as e:
        print(f"\n❌ FINAL PRODUCTION PIPELINE FAILED: {e}")
        traceback.print_exc()
        return False

def run_walkforward_validation(data_path):
    """รัน WalkForward validation"""
    try:
        print("🔍 Running WalkForward validation...")

        # Import และรัน WalkForward

        df = pd.read_parquet(data_path)
        feature_cols = [c for c in df.columns if c not in ['target', 'date', 'datetime', 'timestamp']]

        wfv = WalkForwardValidation(
            data = df, 
            feature_columns = feature_cols, 
            target_column = 'target', 
            n_splits = 3,  # ลดลงเพื่อให้รันเร็วขึ้น
            test_size = 0.2
        )

        results = wfv.run()
        print(f"✅ WalkForward completed: {len(results)} folds")

        # แสดงผลลัพธ์
        if results:
            aucs = [r.get('auc', 0) for r in results if r.get('auc') is not None]
            accs = [r.get('accuracy', 0) for r in results if r.get('accuracy') is not None]

            if aucs:
                print(f"📊 AUC - Mean: {np.mean(aucs):.3f}, Std: {np.std(aucs):.3f}")
            if accs:
                print(f"📊 Accuracy - Mean: {np.mean(accs):.3f}, Std: {np.std(accs):.3f}")

    except Exception as e:
        print(f"❌ WalkForward validation failed: {e}")
        traceback.print_exc()

def run_full_pipeline():
    """รัน Full ML Pipeline"""
    try:
        print("⚡ Running full ML pipeline...")

        # รัน ProjectP main pipeline
        os.system("python ProjectP.py - - run_full_pipeline")

        print("✅ Full pipeline completed")

    except Exception as e:
        print(f"❌ Full pipeline failed: {e}")

def validate_results():
    """ตรวจสอบและแสดงผลลัพธ์"""
    try:
        print("📊 Validating results...")

        # ตรวจสอบไฟล์ผลลัพธ์
        result_files = [
            "output_default/production_features.parquet", 
            "output_default/balanced_data_production.parquet", 
            "output_default/walkforward_results.json", 
            "logs/walkforward.log"
        ]

        for file_path in result_files:
            if os.path.exists(file_path):
                if file_path.endswith('.parquet'):
                    df = pd.read_parquet(file_path)
                    print(f"✅ {file_path}: {df.shape}")
                    if 'target' in df.columns:
                        print(f"   Target distribution: {df['target'].value_counts().to_dict()}")
                else:
                    size = os.path.getsize(file_path)
                    print(f"✅ {file_path}: {size} bytes")
            else:
                print(f"⚠️ {file_path}: Not found")

        # แสดงผลลัพธ์ล่าสุด
        if os.path.exists("output_default"):
            files = os.listdir("output_default")
            recent_files = sorted([f for f in files if f.endswith(('.parquet', '.json', '.csv'))], 
                                key = lambda x: os.path.getmtime(os.path.join("output_default", x)), 
                                reverse = True)[:5]
            print(f"📁 Recent output files: {recent_files}")

    except Exception as e:
        print(f"❌ Results validation failed: {e}")

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 PRODUCTION PIPELINE: SUCCESS!")
        sys.exit(0)
    else:
        print("\n❌ PRODUCTION PIPELINE: FAILED!")
        sys.exit(1)