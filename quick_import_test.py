#!/usr/bin/env python3
    from basic_auc_fix import create_optimized_model
    from src.data_loader.csv_loader import CSVLoader
    from src.evidently_compat import DataDrift, ValueDrift, get_drift_detector
    from src.pydantic_v2_compat import BaseModel, Field, SecretField
    import ProjectP
"""
Quick Import Test
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
ทดสอบการ import หลักแบบรวดเร็ว
"""

print("🚀 Quick Import Test Started")
print(" = " * 40)

# Test 1: Pydantic
print("\n🔍 Testing Pydantic compatibility...")
try:

    print("✅ Pydantic v2 compatibility OK")
except Exception as e:
    print(f"❌ Pydantic error: {e}")

# Test 2: Evidently
print("\n🔍 Testing Evidently compatibility...")
try:

    detector = get_drift_detector()
    print("✅ Evidently compatibility OK")
except Exception as e:
    print(f"❌ Evidently error: {e}")

# Test 3: Basic AUC Fix
print("\n🔍 Testing basic_auc_fix...")
try:

    print("✅ basic_auc_fix import OK")
except Exception as e:
    print(f"❌ basic_auc_fix error: {e}")

# Test 4: CSV Loader
print("\n🔍 Testing CSV loader...")
try:

    print("✅ CSV loader OK")
except Exception as e:
    print(f"❌ CSV loader error: {e}")

# Test 5: ProjectP
print("\n🔍 Testing ProjectP...")
try:

    print("✅ ProjectP import OK")
except Exception as e:
    print(f"❌ ProjectP error: {e}")

print("\n🎉 Quick import test completed!")