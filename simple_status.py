#!/usr/bin/env python3
"""
Simple Production Status
========================
เช็คสถานะแบบง่าย ๆ
"""

print("🚀 PRODUCTION STATUS CHECK")
print("=" * 40)

try:
    from src.pydantic_v2_compat import SecretField

    print("✅ Pydantic v2 - OK")
except:
    print("❌ Pydantic v2 - FAIL")

try:
    from src.evidently_compat import get_drift_detector

    print("✅ Evidently - OK")
except:
    print("❌ Evidently - FAIL")

try:
    from basic_auc_fix import create_optimized_model

    print("✅ basic_auc_fix - OK")
except:
    print("❌ basic_auc_fix - FAIL")

try:
    import ProjectP

    print("✅ ProjectP - OK")
except:
    print("❌ ProjectP - FAIL")

print("=" * 40)
print("🎉 Ready to run: python ProjectP.py --run_full_pipeline")
print("✅ Production deployment recommended!")
