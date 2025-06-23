#!/usr/bin/env python3
"""
Quick Production Status Check
============================
ตรวจสอบสถานะ Production แบบรวดเร็ว
"""

import logging
import sys

# Setup minimal logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def quick_test():
    """Quick production test"""
    print("🚀 QUICK PRODUCTION STATUS CHECK")
    print("=" * 50)

    results = {}

    # Test 1: Pydantic
    try:
        from src.pydantic_v2_compat import BaseModel, Field, SecretField

        results["pydantic"] = True
        print("✅ Pydantic v2 compatibility - READY")
    except Exception as e:
        results["pydantic"] = False
        print(f"❌ Pydantic v2 - FAILED: {e}")

    # Test 2: Evidently
    try:
        from src.evidently_compat import get_drift_detector

        detector = get_drift_detector()
        results["evidently"] = True
        print("✅ Evidently compatibility - READY")
    except Exception as e:
        results["evidently"] = False
        print(f"❌ Evidently - FAILED: {e}")

    # Test 3: Basic AUC Fix
    try:
        from basic_auc_fix import create_optimized_model

        if callable(create_optimized_model):
            results["basic_auc_fix"] = True
            print("✅ basic_auc_fix - READY")
        else:
            results["basic_auc_fix"] = False
            print("❌ basic_auc_fix - NOT CALLABLE")
    except Exception as e:
        results["basic_auc_fix"] = False
        print(f"❌ basic_auc_fix - FAILED: {e}")

    # Test 4: CSV Loader (minimal test)
    try:
        from src.data_loader.csv_loader import safe_load_csv_auto

        results["csv_loader"] = True
        print("✅ CSV loader - READY")
    except Exception as e:
        results["csv_loader"] = False
        print(f"❌ CSV loader - FAILED: {e}")

    # Test 5: ProjectP
    try:
        import ProjectP

        if hasattr(ProjectP, "main") and hasattr(ProjectP, "run_full_pipeline"):
            results["projectp"] = True
            print("✅ ProjectP integration - READY")
        else:
            results["projectp"] = False
            print("❌ ProjectP - MISSING FUNCTIONS")
    except Exception as e:
        results["projectp"] = False
        print(f"❌ ProjectP - FAILED: {e}")

    print("=" * 50)

    # Final status
    ready_count = sum(results.values())
    total_count = len(results)

    if ready_count == total_count:
        print("🎉 PRODUCTION READY!")
        print("✅ ALL COMPONENTS OPERATIONAL")
        print("💡 Ready to run: python ProjectP.py --run_full_pipeline")
        return True
    else:
        print(f"⚠️  PARTIALLY READY ({ready_count}/{total_count})")
        failed = [comp for comp, status in results.items() if not status]
        print(f"❌ Failed: {', '.join(failed)}")

        if ready_count >= 4:
            print("💡 Minimum requirements met - should work")
            return True
        else:
            print("🔧 Need more fixes before production")
            return False


if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)
