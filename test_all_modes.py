from pathlib import Path
import os
import subprocess
import sys
import time
"""
Ultimate Pipeline Test Suite
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
Tests all pipeline modes to ensure production readiness.
"""


def run_command_safely(cmd, description):
    """Run command with proper encoding handling."""
    print(f"\n🔄 {description}")
    print(f"📝 Command: {' '.join(cmd)}")

    try:
        # Use proper encoding for Windows
        result = subprocess.run(
            cmd, 
            capture_output = True, 
            text = True, 
            encoding = 'utf - 8', 
            errors = 'replace', 
            timeout = 300  # 5 minutes timeout
        )

        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            if result.stdout:
                print("📤 Output:")
                # Only show last 10 lines to avoid spam
                lines = result.stdout.strip().split('\n')
                for line in lines[ - 10:]:
                    print(f"  {line}")
        else:
            print(f"❌ {description} - FAILED")
            if result.stderr:
                print("📤 Error:")
                lines = result.stderr.strip().split('\n')
                for line in lines[ - 5:]:
                    print(f"  {line}")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"💥 {description} - EXCEPTION: {e}")
        return False

def main():
    """Run comprehensive test suite."""
    print("🚀 NICEGOLD Ultimate Pipeline Test Suite")
    print(" = " * 80)

    # Test commands
    tests = [
        {
            "cmd": [sys.executable, "ProjectP.py", " -  - check_resources"], 
            "desc": "Resource Check Test"
        }, 
        {
            "cmd": [sys.executable, "ProjectP.py", " -  - run_full_pipeline"], 
            "desc": "Full Pipeline Test"
        }, 
        {
            "cmd": [sys.executable, "ProjectP.py", " -  - debug_full_pipeline"], 
            "desc": "Debug Pipeline Test"
        }, 
        {
            "cmd": [sys.executable, "ProjectP.py", " -  - preprocess"], 
            "desc": "Preprocessing Test"
        }, 
        {
            "cmd": [sys.executable, "ProjectP.py", " -  - realistic_backtest"], 
            "desc": "Realistic Backtest Test"
        }, 
        {
            "cmd": [sys.executable, "ProjectP.py", " -  - robust_backtest"], 
            "desc": "Robust Backtest Test"
        }, 
        {
            "cmd": [sys.executable, "ProjectP.py", " -  - realistic_backtest_live"], 
            "desc": "Live Backtest Test"
        }, 
        {
            "cmd": [sys.executable, "production_pipeline_runner.py", " -  - run_full_pipeline"], 
            "desc": "Production Runner Test"
        }
    ]

    # Run tests
    results = []

    for test in tests:
        success = run_command_safely(test["cmd"], test["desc"])
        results.append((test["desc"], success))
        time.sleep(2)  # Brief pause between tests

    # Final summary
    print("\n" + " = " * 80)
    print("📊 TEST RESULTS SUMMARY")
    print(" = " * 80)

    success_count = sum(1 for _, success in results if success)
    total_count = len(results)

    print(f"✅ Successful tests: {success_count}/{total_count}")
    print(f"❌ Failed tests: {total_count - success_count}/{total_count}")

    print("\nDetailed Results:")
    for desc, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} - {desc}")

    # Check output directory
    output_dir = Path("output_default")
    if output_dir.exists():
        files = list(output_dir.glob("**/*"))
        print(f"\n📁 Output files created: {len(files)}")
        for file_path in files[:10]:  # Show first 10 files
            print(f"  📄 {file_path}")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more files")

    print("\n" + " = " * 80)

    if success_count == total_count:
        print("🎉 ALL TESTS PASSED! System is production ready!")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)