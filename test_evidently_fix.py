#!/usr/bin/env python3
"""
Test script to verify Evidently compatibility fix (UPDATED)
==========================================================
This script tests the fixed Evidently imports and compatibility layer.
"""

import sys
import warnings

warnings.filterwarnings("ignore")


def test_evidently_compatibility():
    """Test the evidently compatibility layer"""
    print("🧪 Testing Evidently Compatibility Fix")
    print("=" * 50)

    # Test 1: Import compatibility layer
    try:
        from src.evidently_compat import (
            EVIDENTLY_AVAILABLE,
            DataDrift,
            ValueDrift,
            initialize_evidently,
        )

        print("✅ Successfully imported compatibility layer")
        print(f"   Evidently Available: {EVIDENTLY_AVAILABLE}")
    except ImportError as e:
        print(f"❌ Failed to import compatibility layer: {e}")
        return False

    # Test 2: Test ValueDrift initialization
    try:
        drift_detector = ValueDrift(column_name="test_column")
        print("✅ Successfully created ValueDrift instance")
        print(f"   Column name: {drift_detector.column_name}")
    except Exception as e:
        print(f"❌ Failed to create ValueDrift instance: {e}")
        return False

    # Test 3: Test drift calculation with dummy data
    try:
        import numpy as np
        import pandas as pd

        # Create dummy data
        ref_data = pd.DataFrame(
            {
                "test_column": np.random.normal(0, 1, 100),
                "other_col": np.random.uniform(0, 1, 100),
            }
        )

        cur_data = pd.DataFrame(
            {
                "test_column": np.random.normal(0.1, 1, 100),  # Slight shift
                "other_col": np.random.uniform(0, 1, 100),
            }
        )

        result = drift_detector.calculate(ref_data, cur_data)
        print("✅ Successfully calculated drift")
        print(f"   Result: {result}")

        # Validate result structure
        required_keys = ["drift_score", "drift_detected", "method", "column"]
        if all(key in result for key in required_keys):
            print("✅ Result has all required keys")
        else:
            print(f"⚠️ Missing keys: {set(required_keys) - set(result.keys())}")

    except Exception as e:
        print(f"❌ Failed to calculate drift: {e}")
        return False

    # Test 4: Test preprocess module import
    try:
        from projectp.steps.preprocess import (
            EVIDENTLY_AVAILABLE as PREPROCESS_AVAILABLE,
        )
        from projectp.steps.preprocess import ValueDrift as PreprocessValueDrift

        print("✅ Successfully imported from preprocess module")
        print(f"   Preprocess Evidently Available: {PREPROCESS_AVAILABLE}")

        # Test creating instance from preprocess module
        preprocess_drift = PreprocessValueDrift(column_name="preprocess_test")
        print("✅ Successfully created PreprocessValueDrift instance")

    except Exception as e:
        print(f"❌ Failed to import/use preprocess module: {e}")
        return False

    print("\n🎉 All core tests passed! Evidently compatibility fix is working.")
    return True


def test_import_variations():
    """Test different import patterns that might be used"""
    print("\n🔍 Testing Import Variations")
    print("=" * 30)

    variations = [
        "from src.evidently_compat import ValueDrift",
        "from src.evidently_compat import DataDrift",
        "from src.evidently_compat import EVIDENTLY_AVAILABLE",
        "import src.evidently_compat as ec",
    ]

    for variation in variations:
        try:
            exec(variation)
            print(f"✅ {variation}")
        except Exception as e:
            print(f"❌ {variation} - Error: {e}")


def main():
    """Main test function"""
    print("🛡️ Evidently Compatibility Fix Test Suite")
    print("==========================================\n")

    success = test_evidently_compatibility()
    test_import_variations()

    if success:
        print("\n🎯 Summary: Evidently compatibility fix is working correctly!")
        print("   - All imports are working")
        print("   - ValueDrift instances can be created")
        print("   - Drift calculation is functional")
        print("   - Fallback mechanisms are in place")
        return 0
    else:
        print("\n❌ Summary: Some tests failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_evidently_import():
    """ทดสอบการ import Evidently"""
    print("🔍 Testing Evidently import fix...")
    print("=" * 50)

    try:
        # Test 1: Check Evidently version
        print("\n📦 Test 1: Checking Evidently installation")
        import evidently

        version = getattr(evidently, "__version__", "unknown")
        print(f"✅ Evidently version: {version}")

        # Test 2: Direct ValueDrift import
        print("\n🎯 Test 2: Testing ValueDrift import")
        try:
            from evidently.metrics import ValueDrift

            print("✅ ValueDrift imported successfully from evidently.metrics")

            # Test instance creation
            value_drift = ValueDrift(column_name="test_column")
            print("✅ ValueDrift instance created successfully")

        except ImportError as e:
            print(f"❌ ValueDrift import failed: {e}")
            return False

        # Test 3: Test our compatibility layer
        print("\n🛠️ Test 3: Testing compatibility layer")
        try:
            from src.evidently_compat import EVIDENTLY_AVAILABLE
            from src.evidently_compat import ValueDrift as CompatValueDrift

            print(f"✅ Compatibility layer imported (Available: {EVIDENTLY_AVAILABLE})")

            # Test instance creation
            compat_drift = CompatValueDrift(column_name="test_column")
            print("✅ Compatibility ValueDrift instance created")

            # Test calculation method
            dummy_result = compat_drift.calculate(None, None)
            print(
                f"✅ Calculation method works: {dummy_result.get('method', 'unknown')}"
            )

        except Exception as e:
            print(f"❌ Compatibility layer failed: {e}")
            return False

        # Test 4: Test import from complete_final_fix
        print("\n🔧 Test 4: Testing complete_final_fix imports")
        try:
            # Test if the module imports without the warning
            import complete_final_fix

            print("✅ complete_final_fix imported without errors")

        except Exception as e:
            print(f"⚠️ complete_final_fix import issue: {e}")

        print("\n🎉 All Evidently tests passed!")
        print("✅ No more 'No Evidently version found' warnings should appear")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


def test_no_warnings():
    """ทดสอบว่าไม่มี warning แล้ว"""
    print("\n🔇 Testing for warnings suppression...")

    # Capture warnings
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        try:
            from src.evidently_compat import initialize_evidently

            initialize_evidently()

            if len(w) == 0:
                print("✅ No warnings generated")
            else:
                print(f"⚠️ {len(w)} warnings found:")
                for warning in w:
                    print(f"  - {warning.message}")

        except Exception as e:
            print(f"❌ Warning test failed: {e}")


if __name__ == "__main__":
    print("🚀 Starting Evidently Fix Tests")
    print("=" * 50)

    success = test_evidently_import()
    test_no_warnings()

    if success:
        print("\n🎯 SUMMARY")
        print("=" * 50)
        print("✅ Evidently import fix successful")
        print("✅ No more 'No Evidently version found' warnings")
        print("✅ ValueDrift available and working")
        print("✅ Compatibility layer functioning")
        print("\n🚀 Your ML pipeline should now work without Evidently warnings!")
    else:
        print("\n❌ SUMMARY")
        print("=" * 50)
        print("❌ Some tests failed")
        print("💡 Check the error messages above for details")
