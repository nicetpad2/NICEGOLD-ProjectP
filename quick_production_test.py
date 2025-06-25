#!/usr/bin/env python3
from pathlib import Path
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
import json
import numpy as np
import os
import pandas as pd
"""
Quick Production Test
ทดสอบระบบอย่างรวดเร็วและแสดงผลลัพธ์
"""


def quick_test():
    """ทดสอบระบบอย่างรวดเร็ว"""
    print("🚀 Quick Production Test")
    print(" = " * 50)

    # 1. ตรวจสอบข้อมูล
    print("\n📊 Checking Data...")

    data_found = False
    df = None

    # ลองหาไฟล์ข้อมูล
    if Path("output_default/preprocessed_super.parquet").exists():
        df = pd.read_parquet("output_default/preprocessed_super.parquet")
        df = df.head(5000)  # ใช้ข้อมูลเพียง 5000 rows เพื่อความเร็ว
        print(f"   ✅ Found preprocessed data: {df.shape}")
        data_found = True

    elif Path("XAUUSD_M1.csv").exists():
        df = pd.read_csv("XAUUSD_M1.csv", nrows = 5000)
        print(f"   ✅ Found CSV data: {df.shape}")
        data_found = True

    if not data_found:
        print("   ❌ No data found")
        return False

    # 2. ตรวจสอบ features
    print(f"\n🔧 Features Analysis:")
    numeric_cols = df.select_dtypes(include = [np.number]).columns
    print(f"   Numeric columns: {len(numeric_cols)}")
    print(f"   Total columns: {len(df.columns)}")

    if len(numeric_cols) < 3:
        print("   ⚠️ Too few numeric features")
        return False

    # 3. สร้าง target basิc
    print(f"\n🎯 Creating Target...")
    if 'Close' in df.columns:
        # Simple target: future return > 0
        future_return = df['Close'].pct_change().shift( - 3)
        target = (future_return > 0).astype(int).fillna(0)

        target_dist = target.value_counts()
        print(f"   Target distribution: {dict(target_dist)}")

        if len(target_dist) < 2:
            print("   ❌ Only one class in target")
            return False

        imbalance_ratio = target_dist.max() / target_dist.min()
        print(f"   Class imbalance ratio: {imbalance_ratio:.1f}:1")

    else:
        print("   ❌ No 'Close' column found")
        return False

    # 4. Quick model test
    print(f"\n🤖 Quick Model Test...")
    try:

        # เตรียมข้อมูล
        X = df[numeric_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = target

        # แยกข้อมูล
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = 0.3, random_state = 42
        )

        # ฝึกโมเดล
        model = RandomForestClassifier(
            n_estimators = 50,  # ลดลงเพื่อความเร็ว
            random_state = 42, 
            class_weight = 'balanced'
        )

        model.fit(X_train, y_train)

        # ทำนาย
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)

        print(f"   🎉 Quick Test AUC: {auc:.3f}")

        if auc > 0.60:
            status = "✅ GOOD"
            recommendation = "Ready for full pipeline"
        elif auc > 0.55:
            status = "📈 FAIR"
            recommendation = "Needs feature engineering"
        else:
            status = "⚠️ POOR"
            recommendation = "Major improvements needed"

        print(f"   Status: {status}")
        print(f"   Recommendation: {recommendation}")

        # บันทึกผลลัพธ์
        results = {
            "quick_test_auc": float(auc), 
            "data_shape": list(df.shape), 
            "features_count": len(numeric_cols), 
            "target_distribution": {str(k): int(v) for k, v in target_dist.items()}, 
            "imbalance_ratio": float(imbalance_ratio), 
            "status": status, 
            "recommendation": recommendation
        }

        os.makedirs("fixes", exist_ok = True)
        with open("fixes/quick_test_results.json", "w") as f:
            json.dump(results, f, indent = 2)

        print(f"\n💾 Results saved to: fixes/quick_test_results.json")

        # Top features
        feature_importance = pd.DataFrame({
            'feature': X.columns, 
            'importance': model.feature_importances_
        }).sort_values('importance', ascending = False)

        print(f"\n📈 Top 5 Features:")
        print(feature_importance.head(5).to_string(index = False))

        return auc > 0.55

    except Exception as e:
        print(f"   ❌ Model test failed: {e}")
        return False

def main():
    """Main function"""
    success = quick_test()

    print(f"\n{' = '*50}")
    if success:
        print("🎉 Quick Test PASSED!")
        print("💡 Next steps:")
        print("   1. Run full emergency_auc_fix.py")
        print("   2. Apply advanced feature engineering")
        print("   3. Test with ProjectP.py full pipeline")
    else:
        print("⚠️ Quick Test FAILED!")
        print("💡 Troubleshooting needed:")
        print("   1. Check data quality")
        print("   2. Review feature engineering")
        print("   3. Investigate target creation")

    return success

if __name__ == "__main__":
    main()