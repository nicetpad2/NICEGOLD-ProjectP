#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test AUC Improvement Functions
"""

import sys
import os
sys.path.append(os.getcwd())

try:
    from feature_engineering import (
        run_auc_emergency_fix, 
        run_advanced_feature_engineering, 
        run_model_ensemble_boost, 
        run_threshold_optimization_v2
    )
    
    print("🚨 Running AUC Emergency Fix...")
    result1 = run_auc_emergency_fix()
    print(f"Emergency Fix Result: {result1}")
    
    print("\n🧠 Running Advanced Feature Engineering...")
    result2 = run_advanced_feature_engineering()
    print(f"Advanced Features Result: {result2}")
    
    print("\n🚀 Running Model Ensemble Boost...")
    result3 = run_model_ensemble_boost()
    print(f"Ensemble Boost Result: {result3}")
    
    print("\n🎯 Running Threshold Optimization V2...")
    result4 = run_threshold_optimization_v2()
    print(f"Threshold Optimization Result: {result4}")
    
    print(f"\n📊 Overall Results:")
    print(f"Emergency Fix: {result1}")
    print(f"Advanced Features: {result2}")
    print(f"Ensemble Boost: {result3}")
    print(f"Threshold Optimization: {result4}")
    
    if all([result1, result2, result3, result4]):
        print("\n✅ ALL AUC FIXES SUCCESSFUL!")
    else:
        print(f"\n⚠️ Some fixes need attention")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
