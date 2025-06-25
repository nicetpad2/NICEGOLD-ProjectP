#!/usr/bin/env python3
from datetime import datetime
import json
import os
import pandas as pd
    import sys
import time
"""
Production Results Viewer
ดูผลลัพธ์การรันระบบโปรดักชั่นแบบสด ๆ
"""


def view_production_results():
    """ดูผลลัพธ์การรันระบบโปรดักชั่น"""
    print("🎯 PRODUCTION RESULTS VIEWER")
    print(" = " * 50)
    print(f"⏰ Timestamp: {datetime.now().strftime('%Y - %m - %d %H:%M:%S')}")
    print()

    # 1. Check fixes directory
    print("📁 CHECKING FIXES STATUS:")
    fixes_dir = "fixes"
    if os.path.exists(fixes_dir):
        files = os.listdir(fixes_dir)
        for file in sorted(files):
            if file.endswith('.json'):
                print(f"   ✅ {file}")
            elif file.endswith('.py'):
                print(f"   📄 {file}")
            elif file.endswith('.parquet'):
                print(f"   📊 {file}")

    # 2. Check if quick fix results exist
    print("\n🚀 QUICK FIX RESULTS:")
    quick_results_path = "fixes/quick_fix_results.json"
    if os.path.exists(quick_results_path):
        try:
            with open(quick_results_path, 'r') as f:
                results = json.load(f)

            print(f"   Original AUC: {results.get('original_auc', 'N/A')}")
            print(f"   Improved AUC: {results.get('improved_auc', 'N/A'):.4f}")
            print(f"   Improvement: {results.get('improvement_pct', 'N/A'):.1f}%")
            print(f"   Status: {results.get('status', 'N/A')}")

            if 'cv_scores' in results:
                cv_scores = results['cv_scores']
                print(f"   CV Scores: {[f'{score:.3f}' for score in cv_scores]}")

        except Exception as e:
            print(f"   ❌ Error reading results: {e}")
    else:
        print("   ⏳ Quick fix still running or not started...")

    # 3. Check main pipeline logs
    print("\n📊 PIPELINE STATUS:")
    log_files = [
        "production_auc_fix.log", 
        "projectp_full.log", 
        "logs/pipeline.log"
    ]

    for log_file in log_files:
        if os.path.exists(log_file):
            print(f"   ✅ Found: {log_file}")
            # Read last few lines
            try:
                with open(log_file, 'r', encoding = 'utf - 8') as f:
                    lines = f.readlines()
                    if lines:
                        last_lines = lines[ - 3:]
                        print("   Last entries:")
                        for line in last_lines:
                            line = line.strip()
                            if line:
                                print(f"     {line}")
                print()
            except:
                pass
        else:
            print(f"   ❌ Not found: {log_file}")

    # 4. Check data status
    print("\n💾 DATA STATUS:")
    data_files = {
        "fixes/preprocessed_super_fixed.parquet": "Fixed Data", 
        "output_default/preprocessed_super.parquet": "Original Data", 
        "XAUUSD_M1.csv": "Raw Data"
    }

    for file_path, description in data_files.items():
        if os.path.exists(file_path):
            try:
                if file_path.endswith('.parquet'):
                    df = pd.read_parquet(file_path)
                elif file_path.endswith('.csv'):
                    df = pd.read_csv(file_path, nrows = 1000)

                shape = df.shape
                print(f"   ✅ {description}: {shape[0]:, } rows × {shape[1]} cols")

                # Check target if exists
                if 'target' in df.columns:
                    target_dist = df['target'].value_counts().to_dict()
                    print(f"      Target: {target_dist}")

            except Exception as e:
                print(f"   ⚠️ {description}: Error reading - {e}")
        else:
            print(f"   ❌ {description}: Not found")

    # 5. Check model files
    print("\n🤖 MODEL STATUS:")
    model_dirs = ["models", "output_default", "mlruns"]

    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            files = os.listdir(model_dir)
            model_files = [f for f in files if any(ext in f for ext in ['.pkl', '.joblib', '.cbm', '.json'])]

            if model_files:
                print(f"   ✅ {model_dir}/: {len(model_files)} model files")
                for model_file in model_files[:3]:  # Show first 3
                    print(f"      - {model_file}")
                if len(model_files) > 3:
                    print(f"      ... and {len(model_files) - 3} more")
            else:
                print(f"   📁 {model_dir}/: No model files yet")
        else:
            print(f"   ❌ {model_dir}/: Directory not found")

    # 6. Performance summary
    print("\n📈 PERFORMANCE SUMMARY:")

    # Try to get latest validation results
    validation_path = "fixes/final_validation_report.json"
    if os.path.exists(validation_path):
        try:
            with open(validation_path, 'r') as f:
                validation = json.load(f)

            checks = validation.get('checks', {})
            passed = sum(1 for v in checks.values() if v)
            total = len(checks)

            print(f"   Validation Score: {passed}/{total}")
            print(f"   Production Ready: {'✅ YES' if validation.get('production_ready', False) else '⚠️ NO'}")

            if 'test_auc' in validation:
                print(f"   Test AUC: {validation['test_auc']:.4f}")

        except Exception as e:
            print(f"   ❌ Error reading validation: {e}")

    # 7. Next steps
    print("\n🚀 NEXT STEPS:")
    if os.path.exists(quick_results_path):
        try:
            with open(quick_results_path, 'r') as f:
                results = json.load(f)

            auc = results.get('improved_auc', 0)
            if auc > 0.65:
                print("   🎉 SUCCESS! Ready for production deployment")
                print("   ➤ Run full pipeline: python ProjectP.py - - mode 7")
                print("   ➤ Monitor results in real - time")
            elif auc > 0.55:
                print("   ⚠️ Good progress! Consider additional tuning")
                print("   ➤ Try ensemble methods or feature engineering")
            else:
                print("   🔧 Needs more work - consider advanced techniques")
                print("   ➤ Deep learning models or external data")

        except:
            pass
    else:
        print("   ⏳ Wait for quick fix to complete...")
        print("   ➤ Monitor: python quick_auc_fix.py")

    return True

def monitor_live_results():
    """Monitor results in real - time"""
    print("\n👁️ LIVE MONITORING MODE")
    print("Press Ctrl + C to stop monitoring...")

    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            view_production_results()
            print(f"\n⏰ Refreshing in 30 seconds...")
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n\n✋ Monitoring stopped by user")

if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == " -  - monitor":
        monitor_live_results()
    else:
        view_production_results()

        # Ask if user wants live monitoring
        try:
            response = input("\n🔄 Start live monitoring? (y/n): ").lower()
            if response in ['y', 'yes']:
                monitor_live_results()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")