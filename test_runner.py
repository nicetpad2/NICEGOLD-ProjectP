import os
import subprocess
import sys

def run_test(test_path):
    print(f"Running test: {test_path}")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_path, "-v"],
            capture_output=True,
            text=True
        )
        print("--- STDOUT ---")
        print(result.stdout)
        print("--- STDERR ---")
        print(result.stderr)
        print(f"Exit code: {result.returncode}")
        return result.returncode == 0
    except Exception as e:
        print(f"Error running test: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    else:
        test_path = "tests/test_strategy_import_safe_load.py"
    
    success = run_test(test_path)
    print(f"Test {'passed' if success else 'failed'}")
    sys.exit(0 if success else 1)
