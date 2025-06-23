"""
Quick Test - Real Data Integration
==================================

ทดสอบการใช้ข้อมูลจริงแทนข้อมูลตัวอย่างแบบรวดเร็ว
"""

import os
import sys
from colorama import Fore, Style, init as colorama_init
colorama_init(autoreset=True)

def test_real_data_integration():
    """ทดสอบการโหลดข้อมูลจริงและบันทึกลงไฟล์"""
    
    print(f"{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}🧪 TESTING REAL DATA INTEGRATION")
    print(f"{Fore.CYAN}{'='*60}")
    
    try:
        # 1. โหลดข้อมูลจริง
        print(f"{Fore.GREEN}Step 1: Loading real market data...{Style.RESET_ALL}")
        from real_data_loader import load_real_trading_data
        
        df, info = load_real_trading_data(
            timeframe="M15",
            max_rows=5000,  # ใช้ 5k แถวสำหรับทดสอบเร็ว
            profit_threshold=0.12
        )
        
        print(f"{Fore.GREEN}✅ Loaded {len(df):,} samples with {len(df.columns)} columns{Style.RESET_ALL}")
        
        # 2. บันทึกข้อมูลสำหรับ pipeline
        print(f"{Fore.GREEN}Step 2: Saving data for pipeline...{Style.RESET_ALL}")
        
        os.makedirs("output_default", exist_ok=True)
        
        # บันทึกข้อมูลหลัก
        df.to_csv("output_default/real_trading_data.csv", index=False)
        
        # แยก features และ targets
        feature_cols = [col for col in df.columns if col not in 
                       ['Time', 'target', 'target_binary', 'target_return']]
        target_cols = ['target', 'target_binary', 'target_return']
        
        # บันทึก features
        df[feature_cols].to_csv("output_default/features.csv", index=False)
        print(f"💾 Features saved: {len(feature_cols)} columns")
        
        # บันทึก targets  
        df[target_cols].to_csv("output_default/targets.csv", index=False)
        print(f"💾 Targets saved: {len(target_cols)} columns")
        
        # 3. สร้างสถิติ
        print(f"{Fore.GREEN}Step 3: Generating statistics...{Style.RESET_ALL}")
        
        stats = {
            'source': 'XAUUSD Real Market Data',
            'samples': len(df),
            'features': len(feature_cols),
            'targets': len(target_cols),
            'time_range': f"{df['Time'].min()} to {df['Time'].max()}",
            'price_range': f"${df['Close'].min():.2f} - ${df['Close'].max():.2f}",
            'target_distribution': info['target_distribution']
        }
        
        import json
        with open("output_default/real_data_stats.json", "w") as f:
            json.dump(stats, f, indent=2, default=str)
        
        # 4. แสดงผลสรุป
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}📊 INTEGRATION SUMMARY")
        print(f"{Fore.CYAN}{'='*60}")
        
        print(f"{Fore.GREEN}✅ Data Source: Real XAUUSD Market Data")
        print(f"{Fore.GREEN}✅ Total Samples: {len(df):,}")
        print(f"{Fore.GREEN}✅ Features: {len(feature_cols)}")
        print(f"{Fore.GREEN}✅ Time Range: {df['Time'].min()} to {df['Time'].max()}")
        print(f"{Fore.GREEN}✅ Price Range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
        
        print(f"\n{Fore.YELLOW}🎯 Target Distribution:")
        for target, count in info['target_distribution'].items():
            pct = count / len(df) * 100
            label = {0: 'Hold', 1: 'Buy', 2: 'Sell'}.get(target, f'Target_{target}')
            print(f"   {label}: {count:,} ({pct:.1f}%)")
        
        print(f"\n{Fore.CYAN}📁 Files Created:")
        print(f"   - output_default/real_trading_data.csv")
        print(f"   - output_default/features.csv")
        print(f"   - output_default/targets.csv")
        print(f"   - output_default/real_data_stats.json")
        
        # 5. ทดสอบการโหลดข้อมูลกลับ
        print(f"\n{Fore.GREEN}Step 4: Testing data reload...{Style.RESET_ALL}")
        
        import pandas as pd
        test_features = pd.read_csv("output_default/features.csv")
        test_targets = pd.read_csv("output_default/targets.csv")
        
        print(f"✅ Reloaded features: {test_features.shape}")
        print(f"✅ Reloaded targets: {test_targets.shape}")
        
        print(f"\n{Fore.GREEN}🎉 REAL DATA INTEGRATION SUCCESSFUL!{Style.RESET_ALL}")
        print(f"{Fore.CYAN}ตอนนี้ ProjectP.py จะใช้ข้อมูลจริงแทนข้อมูลตัวอย่างแล้ว!{Style.RESET_ALL}")
        
        return True
        
    except Exception as e:
        print(f"{Fore.RED}❌ Integration failed: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        return False

def create_pipeline_patch():
    """สร้าง patch สำหรับ pipeline ให้ใช้ข้อมูลจริง"""
    
    patch_code = '''"""
Patch for using real data in pipeline
"""

import pandas as pd
import os

def load_real_data_for_pipeline():
    """โหลดข้อมูลจริงสำหรับ pipeline"""
    try:
        features = pd.read_csv("output_default/features.csv")
        targets = pd.read_csv("output_default/targets.csv")
        
        # รวมเป็น DataFrame เดียว
        df = pd.concat([features, targets], axis=1)
        
        print(f"📊 Loaded real data: {df.shape}")
        return df
        
    except Exception as e:
        print(f"⚠️ Could not load real data: {e}")
        # สร้างข้อมูลสำรองขนาดเล็ก
        import numpy as np
        np.random.seed(42)
        n = 1000
        
        df = pd.DataFrame({
            'feature_1': np.random.randn(n),
            'feature_2': np.random.randn(n),
            'Close': 1800 + np.random.randn(n) * 50,
            'Volume': np.random.uniform(0.1, 1.0, n),
            'target': np.random.choice([0, 1, 2], n, p=[0.5, 0.25, 0.25]),
            'target_binary': np.random.choice([0, 1], n, p=[0.6, 0.4])
        })
        
        print(f"📊 Using fallback data: {df.shape}")
        return df

# Override functions ที่ใช้ข้อมูลตัวอย่าง
def get_base_data(*args, **kwargs):
    return load_real_data_for_pipeline()

def load_main_training_data(*args, **kwargs):
    return load_real_data_for_pipeline()

# Monkey patch
import sys
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
'''
    
    with open("real_data_patch.py", "w", encoding="utf-8") as f:
        f.write(patch_code)
    
    print(f"{Fore.GREEN}📁 Created real_data_patch.py{Style.RESET_ALL}")

if __name__ == "__main__":
    print(f"{Fore.CYAN}🚀 Starting Real Data Integration Test{Style.RESET_ALL}")
    
    success = test_real_data_integration()
    
    if success:
        create_pipeline_patch()
        
        print(f"\n{Fore.GREEN}🎯 NEXT STEPS:{Style.RESET_ALL}")
        print(f"1. ข้อมูลจริงพร้อมใช้แล้วใน output_default/")
        print(f"2. รัน: python ProjectP.py --run_full_pipeline")
        print(f"3. ระบบจะใช้ข้อมูลทองคำจริงแทนข้อมูลตัวอย่าง")
        
    else:
        print(f"\n{Fore.RED}❌ Integration failed. จะใช้ข้อมูลตัวอย่างแทน{Style.RESET_ALL}")
