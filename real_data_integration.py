"""
Integration module to replace dummy data with real data in ProjectP pipeline
=========================================================================

This module integrates the real data loader into the existing ProjectP pipeline
"""

import os
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from real_data_loader import load_real_trading_data

def integrate_real_data_to_pipeline():
    """
    แทนที่ข้อมูลตัวอย่างด้วยข้อมูลจริงในระบบ pipeline
    """
    print("🚀 INTEGRATING REAL DATA INTO PIPELINE")
    print("=" * 60)
    
    try:
        # 1. โหลดข้อมูลจริง M15 (แนะนำสำหรับ production)
        print("📊 Loading real M15 data...")
        df_m15, info_m15 = load_real_trading_data(
            timeframe="M15",
            max_rows=50000,  # ใช้ 50k แถวสำหรับการเทรน
            profit_threshold=0.12  # เกณฑ์กำไร 0.12%
        )
        
        # 2. บันทึกข้อมูลที่ประมวลผลแล้วลงไฟล์
        output_dir = "output_default"
        os.makedirs(output_dir, exist_ok=True)
        
        # บันทึกข้อมูลหลัก
        real_data_path = os.path.join(output_dir, "real_trading_data.csv")
        df_m15.to_csv(real_data_path, index=False)
        print(f"💾 Saved real data to: {real_data_path}")
        
        # บันทึกข้อมูลสำหรับ training/testing
        features_path = os.path.join(output_dir, "features.csv")
        targets_path = os.path.join(output_dir, "targets.csv")
        
        feature_cols = [col for col in df_m15.columns if col not in 
                       ['Time', 'target', 'target_binary', 'target_return']]
        target_cols = ['target', 'target_binary', 'target_return']
        
        df_m15[feature_cols].to_csv(features_path, index=False)
        df_m15[target_cols].to_csv(targets_path, index=False)
        
        print(f"💾 Saved features to: {features_path}")
        print(f"💾 Saved targets to: {targets_path}")
        
        # 3. สร้างไฟล์สถิติ
        stats = {
            'data_source': 'XAUUSD_M15.csv (Real Market Data)',
            'total_samples': len(df_m15),
            'feature_count': len(feature_cols),
            'time_range': {
                'start': df_m15['Time'].min().isoformat(),
                'end': df_m15['Time'].max().isoformat()
            },
            'price_range': {
                'min': float(df_m15['Close'].min()),
                'max': float(df_m15['Close'].max())
            },
            'target_distribution': info_m15['target_distribution'],
            'class_balance': {
                'hold_pct': info_m15['target_distribution'].get(0, 0) / len(df_m15) * 100,
                'buy_pct': info_m15['target_distribution'].get(1, 0) / len(df_m15) * 100,
                'sell_pct': info_m15['target_distribution'].get(2, 0) / len(df_m15) * 100
            }
        }
        
        import json
        stats_path = os.path.join(output_dir, "real_data_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"📊 Saved statistics to: {stats_path}")
        
        # 4. สร้างฟังก์ชันสำหรับโหลดข้อมูลจริงใน pipeline
        create_pipeline_data_loader()
        
        print("\n✅ REAL DATA INTEGRATION COMPLETED!")
        print(f"✅ Real data samples: {len(df_m15):,}")
        print(f"✅ Features: {len(feature_cols)}")
        print(f"✅ Time range: {df_m15['Time'].min()} to {df_m15['Time'].max()}")
        print(f"✅ Target balance: Buy {stats['class_balance']['buy_pct']:.1f}% | Sell {stats['class_balance']['sell_pct']:.1f}% | Hold {stats['class_balance']['hold_pct']:.1f}%")
        
        return True, stats
        
    except Exception as e:
        print(f"❌ Failed to integrate real data: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def create_pipeline_data_loader():
    """สร้างไฟล์ data loader ที่ใช้ข้อมูลจริง"""
    
    loader_code = '''"""
Pipeline Data Loader - ใช้ข้อมูลจริงแทนข้อมูลตัวอย่าง
=================================================
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, Any, Optional

def load_pipeline_data(output_dir: str = "output_default") -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    โหลดข้อมูลจริงสำหรับ pipeline
    
    Returns:
        features_df: DataFrame ของ features
        targets_df: DataFrame ของ targets  
        stats: สถิติข้อมูล
    """
    try:
        # โหลดข้อมูล
        features_path = os.path.join(output_dir, "features.csv")
        targets_path = os.path.join(output_dir, "targets.csv")
        stats_path = os.path.join(output_dir, "real_data_stats.json")
        
        if not all(os.path.exists(p) for p in [features_path, targets_path, stats_path]):
            print("🔄 Real data not found, generating...")
            from real_data_integration import integrate_real_data_to_pipeline
            success, stats = integrate_real_data_to_pipeline()
            if not success:
                raise FileNotFoundError("Failed to generate real data")
        
        # โหลดไฟล์
        features_df = pd.read_csv(features_path)
        targets_df = pd.read_csv(targets_path)
        
        import json
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        print(f"✅ Loaded real trading data: {len(features_df):,} samples, {len(features_df.columns)} features")
        
        return features_df, targets_df, stats
        
    except Exception as e:
        print(f"❌ Error loading pipeline data: {e}")
        # Fallback ไปใช้ข้อมูลตัวอย่างชั่วคราว
        return create_fallback_data()

def create_fallback_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """สร้างข้อมูลสำรองกรณีโหลดข้อมูลจริงไม่ได้"""
    print("⚠️ Using fallback dummy data")
    
    np.random.seed(42)
    n_samples = 1000
    
    # สร้าง features
    features_df = pd.DataFrame({
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randn(n_samples),
        'price': 1800 + np.random.randn(n_samples) * 50,
        'volume': np.random.uniform(0.1, 1.0, n_samples)
    })
    
    # สร้าง targets
    targets_df = pd.DataFrame({
        'target': np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.25, 0.25]),
        'target_binary': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'target_return': np.random.randn(n_samples) * 0.1
    })
    
    stats = {
        'data_source': 'Fallback dummy data',
        'total_samples': n_samples,
        'feature_count': len(features_df.columns)
    }
    
    return features_df, targets_df, stats

# สำหรับใช้แทนฟังก์ชันเดิมที่สร้างข้อมูลตัวอย่าง
def get_base_data(*args, **kwargs):
    """แทนที่ฟังก์ชันเดิมที่ใช้ข้อมูลตัวอย่าง"""
    features_df, targets_df, stats = load_pipeline_data()
    
    # รวมข้อมูลเป็น DataFrame เดียว
    df = pd.concat([features_df, targets_df], axis=1)
    df['target'] = targets_df['target']
    
    return df

def load_main_training_data(*args, **kwargs):
    """แทนที่ฟังก์ชันโหลดข้อมูลเทรนนิ่งเดิม"""
    return get_base_data(*args, **kwargs)

# Monkey patch สำหรับ compatibility
import sys
if 'projectp.utils_data' in sys.modules:
    projectp_utils_data = sys.modules['projectp.utils_data']
    projectp_utils_data.load_main_training_data = load_main_training_data
    projectp_utils_data.get_base_data = get_base_data

if __name__ == "__main__":
    # ทดสอบการโหลดข้อมูล
    features_df, targets_df, stats = load_pipeline_data()
    print(f"Features shape: {features_df.shape}")
    print(f"Targets shape: {targets_df.shape}")
    print(f"Stats: {stats}")
'''
    
    # บันทึกไฟล์
    with open("pipeline_data_loader.py", "w", encoding="utf-8") as f:
        f.write(loader_code)
    
    print("📁 Created pipeline_data_loader.py")

def patch_projectp_for_real_data():
    """ปรับปรุง ProjectP.py เพื่อใช้ข้อมูลจริง"""
    
    patch_code = '''
# ===== REAL DATA INTEGRATION PATCH =====
# แทนที่การใช้ข้อมูลตัวอย่างด้วยข้อมูลจริง

print("🔄 Patching ProjectP to use real data...")
try:
    from pipeline_data_loader import load_pipeline_data, get_base_data, load_main_training_data
    
    # Monkey patch เพื่อใช้ข้อมูลจริง
    import sys
    
    # แทนที่ใน projectp.utils_data หากมี
    if 'projectp.utils_data' in sys.modules:
        utils_data = sys.modules['projectp.utils_data']
        utils_data.load_main_training_data = load_main_training_data
        utils_data.get_base_data = get_base_data
        print("✅ Patched projectp.utils_data")
    
    # แทนที่ใน projectp.utils_data_csv หากมี  
    if 'projectp.utils_data_csv' in sys.modules:
        utils_data_csv = sys.modules['projectp.utils_data_csv']
        utils_data_csv.load_and_prepare_main_csv = load_main_training_data
        print("✅ Patched projectp.utils_data_csv")
    
    print("🎯 Real data integration patch applied successfully!")
    
except ImportError as e:
    print(f"⚠️ Could not patch for real data: {e}")
    print("📋 Continuing with existing data source...")

# ===== END REAL DATA INTEGRATION PATCH =====
'''
    
    return patch_code

if __name__ == "__main__":
    print("🚀 Starting Real Data Integration")
    success, stats = integrate_real_data_to_pipeline()
    
    if success:
        print("\n🎉 INTEGRATION SUCCESSFUL!")
        print("📋 Next Steps:")
        print("1. ข้อมูลจริงถูกบันทึกใน output_default/")
        print("2. สามารถรัน ProjectP.py ได้ตามปกติ")
        print("3. ระบบจะใช้ข้อมูลจริงแทนข้อมูลตัวอย่างโดยอัตโนมัติ")
        
        # แสดงสถิติ
        if stats:
            print(f"\n📊 Data Statistics:")
            print(f"   - Samples: {stats['total_samples']:,}")
            print(f"   - Features: {stats['feature_count']}")
            print(f"   - Time range: {stats['time_range']['start']} to {stats['time_range']['end']}")
            print(f"   - Price range: ${stats['price_range']['min']:.2f} - ${stats['price_range']['max']:.2f}")
    else:
        print("\n❌ INTEGRATION FAILED!")
        print("จะใช้ข้อมูลตัวอย่างแทน")
