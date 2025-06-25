#!/usr/bin/env python3
from datetime import datetime
import json
import os
import time
"""
Monitor full pipeline progress
"""


def monitor_pipeline():
    """Monitor the full pipeline execution"""
    print("🔍 Monitoring Full Pipeline Execution...")
    print(f"⏰ Started at: {datetime.now().strftime('%Y - %m - %d %H:%M:%S')}")

    output_dir = "output_default"

    # Key files to monitor
    key_files = [
        "preprocessed_super.parquet", 
        "train_features.txt", 
        "catboost_model_best_cv.pkl", 
        "catboost_model_best.pkl", 
        "predictions.csv", 
        "walkforward_metrics.csv"
    ]

    print("\n📋 Pipeline Steps Status:")
    print(" = "*50)

    for i, file in enumerate(key_files, 1):
        file_path = os.path.join(output_dir, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024*1024)  # MB
            mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            print(f"✅ Step {i}: {file} ({size:.1f}MB, {mtime.strftime('%H:%M:%S')})")
        else:
            print(f"⏳ Step {i}: {file} (waiting...)")

    # Check for error logs
    error_files = [f for f in os.listdir(output_dir) if 'error' in f.lower() or 'fail' in f.lower()]
    if error_files:
        print(f"\n⚠️  Error files found: {error_files}")

    # Check latest log files
    log_files = [f for f in os.listdir(output_dir) if f.endswith('.log')]
    if log_files:
        latest_log = max(log_files, key = lambda x: os.path.getmtime(os.path.join(output_dir, x)))
        print(f"\n📄 Latest log: {latest_log}")

        # Show last few lines of latest log
        try:
            with open(os.path.join(output_dir, latest_log), 'r', encoding = 'utf - 8') as f:
                lines = f.readlines()
                if lines:
                    print("   Last few entries:")
                    for line in lines[ - 3:]:
                        print(f"   {line.strip()}")
        except Exception as e:
            print(f"   Could not read log: {e}")

    print("\n" + " = "*50)
    print("💡 To check progress: Re - run this script")
    print("🎯 Pipeline completion indicators:")
    print("   - predictions.csv exists and is recent")
    print("   - walkforward_metrics.csv shows final results")
    print("   - No recent error files")

if __name__ == "__main__":
    monitor_pipeline()