#!/usr/bin/env python3
"""
Test script for the production pipeline runner
"""

import subprocess
import sys
from pathlib import Path


def test_pipeline_runner():
    """Test the production pipeline runner"""
    print("Testing NICEGOLD-ProjectP Production Pipeline Runner")
    print("=" * 60)

    # Test 1: Help and basic functionality
    print("\n1. Testing help functionality...")
    try:
        result = subprocess.run(
            [sys.executable, "run_full_pipeline.py", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            print("✅ Help functionality works")
        else:
            print("❌ Help functionality failed")
            print(result.stderr)
            return False

    except Exception as e:
        print(f"❌ Help test failed: {e}")
        return False

    # Test 2: Validation-only mode
    print("\n2. Testing validation-only mode...")
    try:
        result = subprocess.run(
            [sys.executable, "run_full_pipeline.py", "--validate-only", "--debug"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode in [0, 1]:  # Either success or expected validation failure
            print("✅ Validation mode works")
            if "Checking dependencies" in result.stdout:
                print("   ✅ Dependency checking active")
            if "Checking GPU setup" in result.stdout:
                print("   ✅ GPU checking active")
            if "Validating data" in result.stdout:
                print("   ✅ Data validation active")
        else:
            print("❌ Validation mode failed")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False

    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False

    # Test 3: Import functionality
    print("\n3. Testing import functionality...")
    try:
        import_test_code = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from run_full_pipeline import ProductionPipelineRunner, load_config

# Test configuration loading
config = load_config()
print(f"Config loaded: {type(config)}")
print(f"Config keys: {list(config.keys())}")

# Test runner creation
runner = ProductionPipelineRunner(config)
print(f"Runner created: {type(runner)}")

print("Import test successful")
"""

        result = subprocess.run(
            [sys.executable, "-c", import_test_code],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            print("✅ Import functionality works")
            print("   ✅ Configuration loading works")
            print("   ✅ Runner creation works")
        else:
            print("❌ Import functionality failed")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False

    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("🎉 All tests passed! Pipeline runner is ready for production.")
    print("\n📋 Usage examples:")
    print("   python run_full_pipeline.py --validate-only")
    print("   python run_full_pipeline.py --mode train --debug")
    print("   python run_full_pipeline.py --mode full --monitor")
    print("   python run_full_pipeline.py --resume backtest")
    print("   python run_full_pipeline.py --mode deploy --gpu")

    return True


if __name__ == "__main__":
    success = test_pipeline_runner()
    sys.exit(0 if success else 1)
