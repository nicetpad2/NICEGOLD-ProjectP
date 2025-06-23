"""
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
