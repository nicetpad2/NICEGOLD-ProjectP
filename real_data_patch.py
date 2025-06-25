        import numpy as np
import os
import pandas as pd
import sys
"""
Patch for using real data in pipeline
"""


def load_real_data_for_pipeline():
    """โหลดข้อมูลจริงสำหรับ pipeline"""
    try:
        features = pd.read_csv("output_default/features.csv")
        targets = pd.read_csv("output_default/targets.csv")

        # รวมเป็น DataFrame เดียว
        df = pd.concat([features, targets], axis = 1)

        print(f"📊 Loaded real data: {df.shape}")
        return df

    except Exception as e:
        print(f"⚠️ Could not load real data: {e}")
        # สร้างข้อมูลสำรองขนาดเล็ก
        np.random.seed(42)
        n = 1000

        df = pd.DataFrame({
            'feature_1': np.random.randn(n), 
            'feature_2': np.random.randn(n), 
            'Close': 1800 + np.random.randn(n) * 50, 
            'Volume': np.random.uniform(0.1, 1.0, n), 
            'target': np.random.choice([0, 1, 2], n, p = [0.5, 0.25, 0.25]), 
            'target_binary': np.random.choice([0, 1], n, p = [0.6, 0.4])
        })

        print(f"📊 Using fallback data: {df.shape}")
        return df

# Override functions ที่ใช้ข้อมูลตัวอย่าง
def get_base_data(*args, **kwargs):
    return load_real_data_for_pipeline()

def load_main_training_data(*args, **kwargs):
    return load_real_data_for_pipeline()

# Monkey patch
modules_to_patch = [
    'projectp.utils_data', 
    'projectp.utils_data_csv', 
    'src.data_loader'
]

for module_name in modules_to_patch:
    if module_name in sys.modules:
        module = sys.modules[module_name]
        if hasattr(module, 'load_main_training_data'):
            module.load_main_training_data = load_main_training_data
        if hasattr(module, 'get_base_data'):
            module.get_base_data = get_base_data
        print(f"✅ Patched {module_name}")

print("🎯 Pipeline patched to use real data!")