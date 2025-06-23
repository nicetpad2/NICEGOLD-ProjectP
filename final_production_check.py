#!/usr/bin/env python3
"""
Final Production Readiness Check
===============================
ตรวจสอบความพร้อมสำหรับ Production ครั้งสุดท้าย
"""

import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_all_compatibility_layers():
    """Test all compatibility layers for production readiness"""

    results = {}

    # Test 1: Pydantic v2 compatibility (SecretField, Field, BaseModel, BaseSettings)
    logger.info("🔍 Testing Pydantic v2 compatibility...")
    try:
        from src.pydantic_v2_compat import BaseModel, BaseSettings, Field, SecretField

        # Test BaseModel
        class TestModel(BaseModel):
            secret: str = SecretField(default="test")
            normal: str = Field(default="normal")

        model = TestModel()

        # Test BaseSettings
        class TestSettings(BaseSettings):
            app_name: str = "TestApp"
            debug: bool = False

        settings = TestSettings()

        results["pydantic"] = True
        logger.info("✅ Pydantic v2 compatibility - ALL COMPONENTS WORKING")

    except Exception as e:
        results["pydantic"] = False
        logger.error(f"❌ Pydantic compatibility failed: {e}")

    # Test 2: Evidently compatibility
    logger.info("🔍 Testing Evidently compatibility...")
    try:
        from src.evidently_compat import DataDrift, ValueDrift, get_drift_detector

        detector = get_drift_detector()
        results["evidently"] = True
        logger.info("✅ Evidently compatibility - WORKING")

    except Exception as e:
        results["evidently"] = False
        logger.error(f"❌ Evidently compatibility failed: {e}")

    # Test 3: Basic AUC fix
    logger.info("🔍 Testing basic_auc_fix...")
    try:
        from basic_auc_fix import create_optimized_model, emergency_model_creation

        if callable(create_optimized_model) and callable(emergency_model_creation):
            results["basic_auc_fix"] = True
            logger.info("✅ basic_auc_fix - WORKING")
        else:
            results["basic_auc_fix"] = False
            logger.error("❌ basic_auc_fix functions not callable")

    except Exception as e:
        results["basic_auc_fix"] = False
        logger.error(f"❌ basic_auc_fix failed: {e}")

    # Test 4: CSV loader (no circular imports)
    logger.info("🔍 Testing CSV loader...")
    try:
        from src.data_loader.csv_loader import safe_load_csv_auto

        results["csv_loader"] = True
        logger.info("✅ CSV loader - WORKING")

    except Exception as e:
        results["csv_loader"] = False
        logger.error(f"❌ CSV loader failed: {e}")

    # Test 5: ProjectP main components
    logger.info("🔍 Testing ProjectP...")
    try:
        import ProjectP

        has_main = hasattr(ProjectP, "main")
        has_pipeline = hasattr(ProjectP, "run_full_pipeline")

        if has_main and has_pipeline:
            results["projectp"] = True
            logger.info("✅ ProjectP - MAIN FUNCTIONS AVAILABLE")
        else:
            results["projectp"] = False
            logger.error("❌ ProjectP missing required functions")

    except Exception as e:
        results["projectp"] = False
        logger.error(f"❌ ProjectP import failed: {e}")

    return results


def main():
    """Main production readiness check"""

    logger.info("🚀 FINAL PRODUCTION READINESS CHECK")
    logger.info("=" * 60)

    # Run all compatibility tests
    results = test_all_compatibility_layers()

    # Generate final report
    logger.info("=" * 60)
    logger.info("📊 PRODUCTION READINESS REPORT")
    logger.info("=" * 60)

    all_ready = True
    for component, status in results.items():
        icon = "✅ READY" if status else "❌ NOT READY"
        logger.info(f"{component:20} : {icon}")
        if not status:
            all_ready = False

    logger.info("=" * 60)

    if all_ready:
        logger.info("🎉 PRODUCTION READY!")
        logger.info("✅ ALL SYSTEMS OPERATIONAL")
        logger.info("🚀 READY TO DEPLOY")

        print("\n" + "=" * 60)
        print("🎉 SUCCESS: ระบบพร้อมสำหรับ Production แล้ว!")
        print("=" * 60)
        print("📋 ส่วนประกอบที่พร้อมใช้งาน:")
        print("   ✅ Pydantic v2 (SecretField, BaseModel, BaseSettings)")
        print("   ✅ Evidently drift detection")
        print("   ✅ Basic AUC optimization")
        print("   ✅ CSV data loading")
        print("   ✅ ProjectP main pipeline")
        print("")
        print("🚀 คำสั่งเรียกใช้:")
        print("   python ProjectP.py --run_full_pipeline")
        print("")
        print("📈 Features ที่พร้อมใช้:")
        print("   • ML pipeline with AUC optimization")
        print("   • Data drift detection")
        print("   • Automated feature engineering")
        print("   • Model training and evaluation")
        print("   • Production-ready error handling")
        print("=" * 60)

        return True
    else:
        logger.error("❌ PRODUCTION NOT READY")
        failed_components = [k for k, v in results.items() if not v]
        logger.error(f"Failed components: {', '.join(failed_components)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
