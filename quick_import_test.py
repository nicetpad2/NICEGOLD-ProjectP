#!/usr/bin/env python3
"""
Quick Import Test
================
ทดสอบการ import หลักแบบรวดเร็ว
"""

print("🚀 Quick Import Test Started")
print("=" * 40)

# Test 1: Pydantic
print("\n🔍 Testing Pydantic compatibility...")
try:
    from src.pydantic_v2_compat import BaseModel, Field, SecretField

    print("✅ Pydantic v2 compatibility OK")
except Exception as e:
    print(f"❌ Pydantic error: {e}")

# Test 2: Evidently
print("\n🔍 Testing Evidently compatibility...")
try:
    from src.evidently_compat import DataDrift, ValueDrift, get_drift_detector

    detector = get_drift_detector()
    print("✅ Evidently compatibility OK")
except Exception as e:
    print(f"❌ Evidently error: {e}")

# Test 3: Basic AUC Fix
print("\n🔍 Testing basic_auc_fix...")
try:
    from basic_auc_fix import create_optimized_model

    print("✅ basic_auc_fix import OK")
except Exception as e:
    print(f"❌ basic_auc_fix error: {e}")

# Test 4: CSV Loader
print("\n🔍 Testing CSV loader...")
try:
    from src.data_loader.csv_loader import CSVLoader

    print("✅ CSV loader OK")
except Exception as e:
    print(f"❌ CSV loader error: {e}")

# Test 5: ProjectP
print("\n🔍 Testing ProjectP...")
try:
    import ProjectP

    print("✅ ProjectP import OK")
except Exception as e:
    print(f"❌ ProjectP error: {e}")

print("\n🎉 Quick import test completed!")
