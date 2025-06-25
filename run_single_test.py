
import os
import pytest
import sys
def run_test(test_path):
    """Run a specific test file"""
    print(f"Running test {test_path}")
    result = pytest.main([' - v', test_path])
    return result

if __name__ == "__main__":
    # If no argument is provided, default to testing a file that exists
    test_path = sys.argv[1] if len(sys.argv) > 1 else "tests/test_backtest_engine.py"

    if not os.path.exists(test_path):
        print(f"Error: Test file {test_path} does not exist")
        sys.exit(1)

    exit_code = run_test(test_path)
    print(f"Test run completed with exit code {exit_code}")
    sys.exit(exit_code)