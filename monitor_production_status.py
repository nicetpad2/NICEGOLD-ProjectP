#!/usr/bin/env python3
"""
Real-time Production Status Monitor
ตรวจสอบสถานะการทำงานและผลลัพธ์แบบ real-time
"""

import os
import json
import time
import pandas as pd
from pathlib import Path
from datetime import datetime

def monitor_production_status():
    """Monitor production pipeline status"""
    print("🖥️ Real-time Production Status Monitor")
    print("=" * 50)
    
    status_info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline_status": "Unknown",
        "auc_results": {},
        "data_status": {},
        "errors": []
    }
    
    # 1. Check pipeline files
    print("\n📊 Checking Pipeline Status...")
    
    pipeline_files = [
        "projectp_full.log",
        "fixes/emergency_auc_fix_results.json",
        "output_default/preprocessed_super.parquet",
        "models/best_model.pkl"
    ]
    
    for file_path in pipeline_files:
        if Path(file_path).exists():
            try:
                stat = os.stat(file_path)
                size_mb = stat.st_size / (1024 * 1024)
                mod_time = datetime.fromtimestamp(stat.st_mtime)
                print(f"   ✅ {file_path}: {size_mb:.1f}MB (modified: {mod_time.strftime('%H:%M:%S')})")
                
                # Check specific files
                if file_path.endswith('.json'):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if 'auc' in str(data).lower():
                            status_info["auc_results"][file_path] = data
                            
            except Exception as e:
                print(f"   ⚠️ {file_path}: Error - {e}")
                status_info["errors"].append(f"{file_path}: {e}")
        else:
            print(f"   ❌ {file_path}: Not found")
    
    # 2. Check latest AUC results
    print(f"\n🎯 Latest AUC Results:")
    
    latest_auc = None
    latest_file = None
    
    for file_path, data in status_info["auc_results"].items():
        for key, value in data.items():
            if 'auc' in key.lower() and isinstance(value, (int, float)):
                print(f"   📈 {file_path}: {key} = {value:.3f}")
                if latest_auc is None or value > latest_auc:
                    latest_auc = value
                    latest_file = file_path
    
    if latest_auc:
        improvement = ((latest_auc - 0.516) / 0.516 * 100) if latest_auc > 0.516 else 0
        print(f"\n   🏆 Best AUC: {latest_auc:.3f} (+{improvement:.1f}% vs baseline 0.516)")
        
        if latest_auc > 0.70:
            status = "🎉 EXCELLENT - Production Ready!"
        elif latest_auc > 0.60:
            status = "✅ GOOD - Ready for testing"
        elif latest_auc > 0.55:
            status = "📈 FAIR - Needs optimization"
        else:
            status = "⚠️ POOR - Major improvements needed"
            
        print(f"   Status: {status}")
        status_info["pipeline_status"] = status
    else:
        print(f"   ❌ No AUC results found")
        status_info["pipeline_status"] = "No Results"
    
    # 3. Check data quality
    print(f"\n📋 Data Quality Check:")
    
    try:
        if Path("output_default/preprocessed_super.parquet").exists():
            df = pd.read_parquet("output_default/preprocessed_super.parquet")
            
            missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            print(f"   📊 Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
            print(f"   🔢 Numeric columns: {len(numeric_cols)}")
            print(f"   💧 Missing data: {missing_pct:.1f}%")
            
            status_info["data_status"] = {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "numeric_columns": len(numeric_cols),
                "missing_pct": float(missing_pct)
            }
            
            # Check for target distribution if exists
            if 'target' in df.columns:
                target_dist = df['target'].value_counts()
                print(f"   🎯 Target distribution: {dict(target_dist)}")
                status_info["data_status"]["target_distribution"] = dict(target_dist)
        else:
            print(f"   ❌ Preprocessed data not found")
            
    except Exception as e:
        print(f"   ⚠️ Data check error: {e}")
        status_info["errors"].append(f"Data check: {e}")
    
    # 4. Check system performance
    print(f"\n💻 System Performance:")
    
    try:
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        print(f"   🖥️ CPU: {cpu_percent:.1f}%")
        print(f"   💾 RAM: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")
        print(f"   💿 Disk: {disk.percent:.1f}% ({disk.free/1024**3:.1f}GB free)")
        
    except ImportError:
        print(f"   ℹ️ psutil not available for system monitoring")
    except Exception as e:
        print(f"   ⚠️ System check error: {e}")
    
    # 5. Save status report
    os.makedirs("fixes", exist_ok=True)
    with open("fixes/production_status_report.json", "w") as f:
        json.dump(status_info, f, indent=2)
    
    # 6. Summary and recommendations
    print(f"\n🚀 Summary & Recommendations:")
    
    if latest_auc and latest_auc > 0.65:
        print(f"   ✅ SUCCESS: System is performing well!")
        print(f"   📋 Next steps:")
        print(f"     1. Deploy to staging environment")
        print(f"     2. Set up production monitoring")
        print(f"     3. Configure automated alerts")
        
    elif latest_auc and latest_auc > 0.55:
        print(f"   📈 PROGRESS: System shows improvement")
        print(f"   📋 Recommendations:")
        print(f"     1. Apply advanced feature engineering")
        print(f"     2. Test ensemble methods")
        print(f"     3. Optimize hyperparameters further")
        
    else:
        print(f"   ⚠️ NEEDS WORK: More improvements required")
        print(f"   📋 Actions needed:")
        print(f"     1. Review data quality and preprocessing")
        print(f"     2. Investigate feature relevance")
        print(f"     3. Consider different modeling approaches")
    
    print(f"\n💾 Status report saved: fixes/production_status_report.json")
    print(f"⏰ Monitor completed at: {datetime.now().strftime('%H:%M:%S')}")
    
    return status_info

def main():
    """Main monitoring function"""
    try:
        return monitor_production_status()
    except Exception as e:
        print(f"❌ Monitoring error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
