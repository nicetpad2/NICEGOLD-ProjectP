#!/usr/bin/env python3
"""
Test script for Ultimate Pipeline
ทดสอบการรัน Ultimate Pipeline แบบอัตโนมัติ
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_ultimate_pipeline():
    """ทดสอบการรัน Ultimate Pipeline"""
    print("🔥 Testing Ultimate Pipeline Integration...")
    
    try:
        # Import และทดสอบ run_ultimate_pipeline
        from projectp.pipeline import run_ultimate_pipeline
        print("✅ Successfully imported run_ultimate_pipeline")
        
        # ทดสอบ AUC Improvement functions
        try:
            from auc_improvement_pipeline import (
                run_auc_emergency_fix,
                run_advanced_feature_engineering, 
                run_model_ensemble_boost,
                run_threshold_optimization_v2
            )
            print("✅ Successfully imported AUC Improvement functions")
        except ImportError as e:
            print(f"⚠️ AUC Improvement functions not available: {e}")
        
        # ทดสอบ PIPELINE_STEPS_ULTIMATE
        from projectp.pipeline import PIPELINE_STEPS_ULTIMATE
        print(f"✅ PIPELINE_STEPS_ULTIMATE has {len(PIPELINE_STEPS_ULTIMATE)} steps")
        
        for i, (step_name, step_func) in enumerate(PIPELINE_STEPS_ULTIMATE):
            print(f"   {i+1}. {step_name}")
        
        print("\n🎯 Ready to run Ultimate Pipeline!")
        print("   Command: python ProjectP.py (then select mode 7)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing ultimate pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ultimate_pipeline()
    if success:
        print("\n🏆 Ultimate Pipeline integration test PASSED!")
    else:
        print("\n💥 Ultimate Pipeline integration test FAILED!")
        sys.exit(1)
