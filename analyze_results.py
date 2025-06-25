#!/usr/bin/env python3
from datetime import datetime
                    import ast
import json
import os
import pandas as pd
"""
ProjectP Results Analyzer - วิเคราะห์ผลลัพธ์ ProjectP อย่างครอบคลุม
"""


def analyze_results():
    """วิเคราะห์ผลลัพธ์ ProjectP"""
    print(" = " * 60)
    print("📊 ProjectP Results Analysis Report")
    print(" = " * 60)
    print(f"🕐 Analysis Time: {datetime.now()}")
    print()

    # 1. Check Classification Report
    print("1️⃣ Model Performance Analysis")
    print(" - " * 40)

    if os.path.exists('classification_report.json'):
        try:
            with open('classification_report.json', 'r') as f:
                data = json.load(f)

            accuracy = data.get('accuracy', 0)
            macro_avg = data.get('macro avg', {})
            weighted_avg = data.get('weighted avg', {})

            print(f"🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

            # Performance rating
            if accuracy >= 0.95:
                rating = "🟢 EXCELLENT"
                auc_estimate = min(0.99, accuracy + 0.02)
            elif accuracy >= 0.85:
                rating = "🟡 GOOD"
                auc_estimate = min(0.95, accuracy + 0.015)
            elif accuracy >= 0.70:
                rating = "🟠 FAIR"
                auc_estimate = min(0.85, accuracy + 0.01)
            else:
                rating = "🔴 NEEDS IMPROVEMENT"
                auc_estimate = max(0.5, accuracy)

            print(f"📈 Performance Rating: {rating}")
            print(f"🎲 Estimated AUC: {auc_estimate:.4f}")
            print()

            # Detailed metrics
            if macro_avg:
                print("📋 Detailed Metrics:")
                print(f"   Precision (Macro): {macro_avg.get('precision', 0):.4f}")
                print(f"   Recall (Macro): {macro_avg.get('recall', 0):.4f}")
                print(f"   F1 - Score (Macro): {macro_avg.get('f1 - score', 0):.4f}")
                print()

            # Class - wise performance
            print("🏷️ Class - wise Performance:")
            for key, value in data.items():
                if isinstance(value, dict) and 'precision' in value:
                    if key not in ['macro avg', 'weighted avg']:
                        precision = value.get('precision', 0)
                        recall = value.get('recall', 0)
                        f1 = value.get('f1 - score', 0)
                        support = value.get('support', 0)
                        print(f"   Class {key}: P = {precision:.3f}, R = {recall:.3f}, F1 = {f1:.3f} (n = {int(support)})")

        except Exception as e:
            print(f"❌ Error reading classification report: {e}")
    else:
        print("❌ Classification report not found")

    print()

    # 2. Feature Analysis
    print("2️⃣ Feature Engineering Analysis")
    print(" - " * 40)

    if os.path.exists('features_main.json'):
        try:
            with open('features_main.json', 'r') as f:
                content = f.read()
                # Parse JSON - like content
                if content.strip().startswith('['):
                    features = ast.literal_eval(content.split('\n')[2:])  # Skip comments
                else:
                    features = json.loads(content)

            print(f"🔧 Total Features: {len(features)}")
            print("📝 Feature Categories:")

            # Categorize features
            categories = {
                'Technical Indicators': [], 
                'Price Features': [], 
                'Lag Features': [], 
                'Other': []
            }

            for feature in features:
                if any(indicator in feature.upper() for indicator in ['RSI', 'MACD', 'ATR', 'ADX']):
                    categories['Technical Indicators'].append(feature)
                elif any(price in feature.upper() for price in ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']):
                    categories['Price Features'].append(feature)
                elif 'lag' in feature.lower():
                    categories['Lag Features'].append(feature)
                else:
                    categories['Other'].append(feature)

            for category, feats in categories.items():
                if feats:
                    print(f"   {category}: {len(feats)} features")
                    if len(feats) <= 5:
                        print(f"     {', '.join(feats)}")
                    else:
                        print(f"     {', '.join(feats[:3])}... ( + {len(feats) - 3} more)")

        except Exception as e:
            print(f"❌ Error reading features: {e}")
    else:
        print("❌ Features file not found")

    print()

    # 3. System Performance
    print("3️⃣ System Performance Analysis")
    print(" - " * 40)

    if os.path.exists('system_info.json'):
        try:
            with open('system_info.json', 'r') as f:
                sys_data = json.load(f)

            print("💻 System Resources:")
            if 'memory' in sys_data:
                memory = sys_data['memory']
                print(f"   RAM: {memory.get('used_gb', 0):.1f}GB / {memory.get('total_gb', 0):.1f}GB")
                print(f"   RAM Usage: {memory.get('percent', 0):.1f}%")

            if 'cpu' in sys_data:
                print(f"   CPU Usage: {sys_data['cpu'].get('percent', 0):.1f}%")

            if 'execution_time' in sys_data:
                exec_time = sys_data['execution_time']
                print(f"   Execution Time: {exec_time:.1f} seconds")

        except Exception as e:
            print(f"❌ Error reading system info: {e}")
    else:
        print("⚠️ System info not available")

    # 4. File Status
    print("\n4️⃣ Output Files Status")
    print(" - " * 40)

    important_files = [
        'classification_report.json', 
        'features_main.json', 
        'system_info.json', 
        'predictions.csv', 
        'preprocessed_super.parquet'
    ]

    total_size = 0
    files_found = 0

    for filename in important_files:
        if os.path.exists(filename):
            stat = os.stat(filename)
            size_mb = stat.st_size / (1024 * 1024)
            modified = datetime.fromtimestamp(stat.st_mtime)
            total_size += size_mb
            files_found += 1

            status = "✅" if size_mb > 0 else "⚠️"
            print(f"   {status} {filename}: {size_mb:.1f}MB ({modified:%H:%M:%S})")
        else:
            print(f"   ❌ {filename}: Not found")

    print(f"\n📊 Summary: {files_found}/{len(important_files)} files found, {total_size:.1f}MB total")

    # 5. Recommendations
    print("\n5️⃣ Recommendations & Next Steps")
    print(" - " * 40)

    if os.path.exists('classification_report.json'):
        try:
            with open('classification_report.json', 'r') as f:
                data = json.load(f)
            accuracy = data.get('accuracy', 0)

            if accuracy >= 0.95:
                print("🎉 EXCELLENT Results!")
                print("   ✓ Model is performing very well")
                print("   ✓ Ready for production deployment")
                print("   ✓ Consider A/B testing with real data")
            elif accuracy >= 0.85:
                print("👍 GOOD Results!")
                print("   ✓ Model shows strong performance")
                print("   ✓ Consider additional feature engineering")
                print("   ✓ Test with more validation data")
            else:
                print("🔧 Improvement Opportunities:")
                print("   • Review feature selection")
                print("   • Try different algorithms")
                print("   • Increase training data")
                print("   • Check for data quality issues")

        except:
            pass

    print("\n📈 General Recommendations:")
    print("   • Monitor model performance over time")
    print("   • Set up automated retraining pipeline")
    print("   • Implement proper model versioning")
    print("   • Create monitoring dashboards")

    print("\n" + " = " * 60)
    print("✅ Analysis Complete!")
    print(" = " * 60)

def main():
    """Main function"""
    analyze_results()

if __name__ == "__main__":
    main()