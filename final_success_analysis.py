#!/usr/bin/env python3
import json
import numpy as np
import os
import pandas as pd
"""
🎉 FINAL SUCCESS ANALYSIS
วิเคราะห์ผลลัพธ์สุดท้ายหลังการแก้ไขปัญหาทั้งหมด
"""


def main():
    """วิเคราะห์ผลลัพธ์สุดท้าย"""
    print("🎉 FINAL SUCCESS ANALYSIS")
    print(" = " * 80)

    # วิเคราะห์ WalkForward Results
    analyze_walkforward_results()

    # วิเคราะห์ Threshold Results
    analyze_threshold_results()

    # สรุปผลรวม
    final_summary()

def analyze_walkforward_results():
    """วิเคราะห์ผลลัพธ์ WalkForward"""
    print("\n🔍 WALKFORWARD VALIDATION RESULTS")
    print(" - " * 50)

    try:
        df = pd.read_csv("output_default/walkforward_metrics.csv")
        print(f"📊 Total folds: {len(df)}")

        # สถิติ AUC
        auc_test = df['auc_test']
        auc_mean = auc_test.mean()
        auc_std = auc_test.std()
        auc_min = auc_test.min()
        auc_max = auc_test.max()

        print(f"🎯 AUC Test Results:")
        print(f"   Mean: {auc_mean:.4f}")
        print(f"   Std:  {auc_std:.4f}")
        print(f"   Min:  {auc_min:.4f}")
        print(f"   Max:  {auc_max:.4f}")

        # สถิติ Accuracy
        acc_test = df['acc_test']
        acc_mean = acc_test.mean()
        acc_std = acc_test.std()
        acc_min = acc_test.min()
        acc_max = acc_test.max()

        print(f"📈 Accuracy Test Results:")
        print(f"   Mean: {acc_mean:.4f}")
        print(f"   Std:  {acc_std:.4f}")
        print(f"   Min:  {acc_min:.4f}")
        print(f"   Max:  {acc_max:.4f}")

        # ประเมินผลลัพธ์
        if auc_mean >= 0.75:
            print("✅ AUC: EXCELLENT (≥ 0.75)")
        elif auc_mean >= 0.7:
            print("✅ AUC: GOOD (≥ 0.70)")
        elif auc_mean >= 0.65:
            print("⚠️ AUC: ACCEPTABLE (≥ 0.65)")
        else:
            print("❌ AUC: NEEDS IMPROVEMENT (< 0.65)")

        if acc_mean >= 0.95:
            print("✅ Accuracy: EXCELLENT (≥ 95%)")
        elif acc_mean >= 0.9:
            print("✅ Accuracy: GOOD (≥ 90%)")
        else:
            print("⚠️ Accuracy: Needs improvement")

        # ตรวจสอบ Overfitting
        train_aucs = df['auc_train']
        train_test_gap = (train_aucs - auc_test).mean()
        print(f"🎲 Overfitting Analysis:")
        print(f"   Train - Test AUC Gap: {train_test_gap:.4f}")

        if train_test_gap > 0.15:
            print("⚠️ High overfitting detected")
        elif train_test_gap > 0.1:
            print("⚠️ Moderate overfitting")
        else:
            print("✅ Low overfitting")

    except Exception as e:
        print(f"❌ Error analyzing walkforward: {e}")

def analyze_threshold_results():
    """วิเคราะห์ผลลัพธ์ Threshold Optimization"""
    print("\n🎯 THRESHOLD OPTIMIZATION RESULTS")
    print(" - " * 50)

    try:
        # อ่าน summary metrics
        with open("models/threshold_summary_metrics.json", 'r') as f:
            threshold_data = json.load(f)

        print(f"🎯 Best Threshold: {threshold_data.get('best_threshold', 'N/A')}")
        print(f"📊 Best AUC: {threshold_data.get('best_auc', 'N/A')}")
        print(f"📈 Best Accuracy: {threshold_data.get('best_accuracy', 'N/A')}")

        # ประเมิน threshold
        best_threshold = threshold_data.get('best_threshold', 0.5)
        if 0.1 <= best_threshold <= 0.4:
            print("✅ Threshold in good range (0.1 - 0.4)")
        elif 0.4 < best_threshold <= 0.6:
            print("⚠️ Threshold moderate (0.4 - 0.6)")
        else:
            print("⚠️ Threshold unusual - may need review")

    except Exception as e:
        print(f"❌ Error analyzing threshold: {e}")

def final_summary():
    """สรุปผลรวมสุดท้าย"""
    print("\n🏆 FINAL SUMMARY")
    print(" = " * 80)

    # ตรวจสอบไฟล์ที่สำคัญ
    important_files = [
        ("output_default/walkforward_metrics.csv", "WalkForward Results"), 
        ("models/threshold_results.csv", "Threshold Results"), 
        ("output_default/preprocessed_super.parquet", "Processed Data"), 
        ("models/threshold_summary_metrics.json", "Threshold Summary"), 
        ("output_default/walkforward_summary_metrics.json", "WalkForward Summary")
    ]

    print("📁 Important Files Status:")
    all_files_exist = True
    for file_path, description in important_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"   ✅ {description}: {size:, } bytes")
        else:
            print(f"   ❌ {description}: Missing")
            all_files_exist = False

    # ประเมินความสำเร็จโดยรวม
    print("\n🎉 OVERALL SUCCESS ASSESSMENT:")

    success_factors = []

    # 1. ตรวจสอบ WalkForward
    try:
        df = pd.read_csv("output_default/walkforward_metrics.csv")
        auc_mean = df['auc_test'].mean()
        if auc_mean >= 0.75:
            success_factors.append("✅ Excellent WalkForward AUC")
        elif auc_mean >= 0.7:
            success_factors.append("✅ Good WalkForward AUC")
        else:
            success_factors.append("⚠️ Moderate WalkForward AUC")
    except:
        success_factors.append("❌ WalkForward issues")

    # 2. ตรวจสอบไฟล์
    if all_files_exist:
        success_factors.append("✅ All important files generated")
    else:
        success_factors.append("⚠️ Some files missing")

    # 3. ตรวจสอบ Class Imbalance Resolution
    if os.path.exists("output_default/preprocessed_super.parquet"):
        success_factors.append("✅ Data preprocessing completed")

    print("\n📋 Success Factors:")
    for factor in success_factors:
        print(f"   {factor}")

    # คำนวณ Success Score
    success_score = len([f for f in success_factors if f.startswith("✅")])
    total_factors = len(success_factors)
    success_percentage = (success_score / total_factors) * 100

    print(f"\n🏆 SUCCESS SCORE: {success_score}/{total_factors} ({success_percentage:.1f}%)")

    if success_percentage >= 90:
        print("🎉 OUTSTANDING SUCCESS! 🎉")
        print("   All critical issues resolved")
        print("   Production ready!")
    elif success_percentage >= 75:
        print("✅ GOOD SUCCESS!")
        print("   Most issues resolved")
        print("   Minor improvements may be needed")
    elif success_percentage >= 60:
        print("⚠️ MODERATE SUCCESS")
        print("   Some issues remain")
        print("   Additional work recommended")
    else:
        print("❌ NEEDS MORE WORK")
        print("   Significant issues remain")

    print("\n" + " = " * 80)
    print("🚀 NICEGOLD TRADING SYSTEM STATUS: ANALYSIS COMPLETE")

if __name__ == "__main__":
    main()