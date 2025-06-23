#!/usr/bin/env python3
"""
Simple Production Pipeline Test
==============================
"""

import os
import sys

def safe_print(msg):
    try:
        print(f"[TEST] {msg}")
    except UnicodeEncodeError:
        safe_msg = msg.encode('ascii', errors='ignore').decode('ascii')
        print(f"[TEST] {safe_msg}")

def test_pipeline():
    safe_print("Testing NICEGOLD Production Pipeline...")
    
    try:
        # Test data loading
        import pandas as pd
        import numpy as np
        
        # Try to find data file
        data_files = ["XAUUSD_M1.parquet", "XAUUSD_M1.csv"]
        df = None
        
        for file in data_files:
            if os.path.exists(file):
                try:
                    if file.endswith('.parquet'):
                        df = pd.read_parquet(file)
                    else:
                        df = pd.read_csv(file)
                    safe_print(f"Data loaded: {file}, shape: {df.shape}")
                    break
                except Exception as e:
                    safe_print(f"Failed to load {file}: {e}")
        
        if df is None:
            safe_print("No data files found!")
            return False
        
        # Test feature engineering
        try:
            from production_class_imbalance_fix_clean import run_production_class_imbalance_fix
            
            safe_print("Testing feature engineering...")
            df_processed = run_production_class_imbalance_fix(df)
            
            if df_processed is not None:
                safe_print(f"Feature engineering successful: {df_processed.shape}")
                
                # Save test result
                os.makedirs("output_default", exist_ok=True)
                df_processed.to_parquet("output_default/test_processed.parquet")
                safe_print("Test data saved successfully")
                
                return True
            else:
                safe_print("Feature engineering returned None")
                return False
                
        except Exception as e:
            safe_print(f"Feature engineering failed: {e}")
            import traceback
            safe_print(f"Error details: {traceback.format_exc()}")
            return False
            
    except Exception as e:
        safe_print(f"Critical test failure: {e}")
        return False

if __name__ == "__main__":
    success = test_pipeline()
    if success:
        safe_print("PIPELINE TEST: SUCCESS")
        sys.exit(0)
    else:
        safe_print("PIPELINE TEST: FAILED")
        sys.exit(1)
