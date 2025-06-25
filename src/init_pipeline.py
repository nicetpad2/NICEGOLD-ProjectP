from pathlib import Path
            from sklearn.feature_selection import mutual_info_regression
            from sklearn.metrics import mutual_info_regression
        from src.evidently_compat import DataDrift, ValueDrift
        from src.evidently_compat import ValueDrift
        from src.pydantic_secretfield import BaseModel, Field, SecretField
        from src.pydantic_secretfield import SecretField
        import builtins
import logging
import sys
import warnings
"""
Pipeline Import Initializer
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
Professional fix for all pipeline import issues including Pydantic v2
Import this module at the start of any script that needs compatibility
"""


# Suppress warnings
warnings.filterwarnings("ignore", category = UserWarning)
warnings.filterwarnings("ignore", category = DeprecationWarning)

# Setup logging
logging.basicConfig(level = logging.INFO, format = "%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def initialize_pipeline_compatibility():
    """Initialize all compatibility fixes for the pipeline"""

    logger.info("🔧 Initializing pipeline compatibility...")

    # Add current directory to path
    current_dir = Path(__file__).parent
    project_root = current_dir.parent

    for path in [str(current_dir), str(project_root)]:
        if path not in sys.path:
            sys.path.insert(0, path)

    success_count = 0
    total_fixes = 3

    # Fix 1: Pydantic SecretField
    try:
        # Make them available globally


        builtins.SecretField = SecretField
        builtins.PydanticField = Field
        builtins.PydanticBaseModel = BaseModel

        # Test that they work
        test_field = SecretField(default = "test")

        logger.info("✅ Pydantic SecretField compatibility established")
        success_count += 1

    except Exception as e:
        logger.warning(f"⚠️ Pydantic fix failed: {e}")

    # Fix 2: Evidently compatibility
    try:


        builtins.ValueDrift = ValueDrift
        builtins.DataDrift = DataDrift

        logger.info("✅ Evidently compatibility established")
        success_count += 1

    except Exception as e:
        logger.warning(f"⚠️ Evidently fix failed: {e}")

    # Fix 3: Other common imports
    try:
        # sklearn mutual_info_regression
        try:
        except ImportError:


        builtins.mutual_info_regression = mutual_info_regression

        logger.info("✅ Common imports compatibility established")
        success_count += 1

    except Exception as e:
        logger.warning(f"⚠️ Common imports fix failed: {e}")

    # Report results
    success_rate = success_count / total_fixes
    if success_rate >= 0.8:
        logger.info(
            f"🎉 Pipeline compatibility initialized successfully ({success_count}/{total_fixes})"
        )
        return True
    else:
        logger.warning(
            f"⚠️ Pipeline compatibility partial success ({success_count}/{total_fixes})"
        )
        return False


def test_pipeline_imports():
    """Test that key pipeline imports work"""

    logger.info("🧪 Testing pipeline imports...")

    test_results = []

    # Test 1: Pydantic SecretField
    try:

        field = SecretField(default = "test")
        test_results.append(("Pydantic SecretField", True, None))
    except Exception as e:
        test_results.append(("Pydantic SecretField", False, str(e)))

    # Test 2: Alternative Pydantic import
    try:
        SecretField = getattr(__builtins__, "SecretField", None)
        if SecretField:
            field = SecretField(default = "test")
            test_results.append(("Global SecretField", True, None))
        else:
            test_results.append(("Global SecretField", False, "Not available"))
    except Exception as e:
        test_results.append(("Global SecretField", False, str(e)))

    # Test 3: Evidently
    try:

        drift = ValueDrift(column_name = "test")
        test_results.append(("Evidently ValueDrift", True, None))
    except Exception as e:
        test_results.append(("Evidently ValueDrift", False, str(e)))

    # Report results
    logger.info("📊 Import Test Results:")
    for test_name, success, error in test_results:
        if success:
            logger.info(f"   ✅ {test_name}")
        else:
            logger.warning(f"   ⚠️ {test_name}: {error}")

    success_count = sum(1 for _, success, _ in test_results if success)
    total_tests = len(test_results)

    return success_count, total_tests


# Auto - initialize on import
try:
    initialization_success = initialize_pipeline_compatibility()
    test_success_count, test_total = test_pipeline_imports()

    if initialization_success and test_success_count >= test_total * 0.7:
        logger.info("🎉 Pipeline imports ready! You can now run your pipeline.")
    else:
        logger.warning("⚠️ Pipeline initialization completed with some issues.")
        logger.info(
            "💡 Try importing specific modules from src.pydantic_secretfield if needed"
        )

except Exception as e:
    logger.error(f"❌ Pipeline initialization failed: {e}")

# Export the key functions for manual use
__all__ = ["initialize_pipeline_compatibility", "test_pipeline_imports"]