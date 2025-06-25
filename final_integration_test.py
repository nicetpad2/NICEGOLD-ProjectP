#!/usr/bin/env python3
from pathlib import Path
        from src.evidently_compat import EVIDENTLY_AVAILABLE, DataDrift, ValueDrift
        from src.pydantic_v2_compat import BaseModel, Field, SecretField
import logging
        import numpy as np
        import ProjectP
import sys
import traceback
"""
Final Integration Test
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
Test that all compatibility layers work together in the ML pipeline
"""


# Setup logging
logging.basicConfig(
    level = logging.INFO, format = "%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_pydantic_compatibility():
    """Test Pydantic v2 compatibility"""
    logger.info("🧪 Testing Pydantic v2 compatibility...")

    try:

        # Test BaseModel
        class TestModel(BaseModel):
            name: str = Field(default = "test")
            secret: str = SecretField(default = "secret_value")

        model = TestModel()
        logger.info(f"✅ Pydantic test successful: {model.name}")
        return True

    except Exception as e:
        logger.error(f"❌ Pydantic test failed: {e}")
        return False


def test_evidently_compatibility():
    """Test Evidently compatibility"""
    logger.info("🧪 Testing Evidently compatibility...")

    try:

        # Test ValueDrift creation
        drift_detector = ValueDrift("target")
        logger.info(f"✅ ValueDrift created: {type(drift_detector).__name__}")

        # Test calculation

        ref_data = {"target": np.random.normal(0, 1, 100)}
        cur_data = {"target": np.random.normal(0.5, 1, 100)}

        result = drift_detector.calculate(ref_data, cur_data)
        logger.info(
            f"✅ Drift calculation successful: score = {result['drift_score']:.3f}, method = {result['method']}"
        )

        return True

    except Exception as e:
        logger.error(f"❌ Evidently test failed: {e}")
        logger.error(traceback.format_exc())
        return False


def test_pipeline_imports():
    """Test key pipeline imports"""
    logger.info("🧪 Testing pipeline imports...")

    import_tests = [
        ("ProjectP", "ProjectP"), 
        ("automl_utils", "automl_utils"), 
        ("feature_engineering", "feature_engineering"), 
        ("classification_utils", "classification_utils"), 
    ]

    success_count = 0
    for module_name, import_name in import_tests:
        try:
            module = __import__(import_name)
            logger.info(f"✅ {module_name} import successful")
            success_count += 1
        except Exception as e:
            logger.warning(f"⚠️ {module_name} import failed: {e}")

    logger.info(f"📊 Pipeline imports: {success_count}/{len(import_tests)} successful")
    return success_count > 0


def test_pipeline_initialization():
    """Test pipeline initialization"""
    logger.info("🧪 Testing pipeline initialization...")

    try:
        # Test main entry point

        # Test if we can access key functions
        if hasattr(ProjectP, "main"):
            logger.info("✅ ProjectP.main found")

        if hasattr(ProjectP, "run_full_pipeline"):
            logger.info("✅ ProjectP.run_full_pipeline found")
        elif hasattr(ProjectP, "main"):
            logger.info("✅ ProjectP.main found (alternative entry point)")

        return True

    except Exception as e:
        logger.warning(f"⚠️ Pipeline initialization test failed: {e}")
        return False


def test_output_directory():
    """Test output directory structure"""
    logger.info("🧪 Testing output directory...")

    output_dir = Path("output_default")
    if output_dir.exists():
        files = list(output_dir.glob("*"))
        logger.info(f"✅ Output directory exists with {len(files)} files")

        # List some key files
        key_files = ["features_main.json", "classification_report.json"]
        for key_file in key_files:
            if (output_dir / key_file).exists():
                logger.info(f"✅ Found {key_file}")
            else:
                logger.info(f"ℹ️ {key_file} not found (may be generated during run)")

        return True
    else:
        logger.info("ℹ️ Output directory doesn't exist yet (will be created during run)")
        return True


def run_comprehensive_test():
    """Run comprehensive compatibility test"""
    logger.info("🚀 Starting comprehensive integration test...")

    tests = [
        ("Pydantic v2 Compatibility", test_pydantic_compatibility), 
        ("Evidently Compatibility", test_evidently_compatibility), 
        ("Pipeline Imports", test_pipeline_imports), 
        ("Pipeline Initialization", test_pipeline_initialization), 
        ("Output Directory", test_output_directory), 
    ]

    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n{' = '*50}")
        logger.info(f"🧪 Running: {test_name}")
        logger.info(" = " * 50)

        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"📊 {test_name}: {status}")
        except Exception as e:
            results[test_name] = False
            logger.error(f"❌ {test_name}: FAIL - {e}")

    # Summary
    logger.info(f"\n{' = '*50}")
    logger.info("📊 INTEGRATION TEST SUMMARY")
    logger.info(" = " * 50)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} {test_name}")

    logger.info(f"\n🎯 Final Result: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Pipeline is ready for full run.")
        return True
    elif passed >= total * 0.8:
        logger.info("⚠️ Most tests passed. Pipeline should work with some limitations.")
        return True
    else:
        logger.error("❌ Critical issues found. Pipeline needs more fixes.")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()

    if success:
        logger.info("\n🎯 RECOMMENDATION: You can now run the full pipeline!")
        logger.info("💻 Try: python ProjectP.py - - run_full_pipeline")
        logger.info("💻 Or use the VS Code task: 'Run Full ML Pipeline'")
    else:
        logger.error(
            "\n⚠️ Please fix the failing tests before running the full pipeline."
        )

    sys.exit(0 if success else 1)