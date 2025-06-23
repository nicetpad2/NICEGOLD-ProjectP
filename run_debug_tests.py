import argparse
import logging
import os
import sys
import pytest
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Debug version of run_tests.py with more verbose output

class _SummaryPlugin:
    """Plugin เก็บสถิติผลการทดสอบ"""
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.failures = []

    def pytest_runtest_logreport(self, report):
        if report.when == 'call':
            self.total += 1
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1
                self.failures.append(report)
            elif report.skipped:
                self.skipped += 1


def main() -> None:
    parser = argparse.ArgumentParser(description='รัน test suite พร้อม verbose debugging')
    parser.add_argument('--test-file', type=str, help='ทดสอบเฉพาะไฟล์นี้')
    parser.add_argument('--test-dir', type=str, default='tests', help='ไดเรกทอรีของไฟล์ทดสอบ')
    args = parser.parse_args()

    os.environ.setdefault('COMPACT_LOG', '1')
    logging.basicConfig(level=logging.INFO)
    
    print("[DEBUG] Starting test run...")
    
    pytest_args = ['-v']  # Verbose output
    
    if args.test_file:
        pytest_args.append(os.path.join(args.test_dir, args.test_file))
    else:
        pytest_args.append(args.test_dir)

    print(f"[DEBUG] Running pytest with args: {pytest_args}")
    
    summary = _SummaryPlugin()
    exit_code = pytest.main(pytest_args, plugins=[summary])
    
    if exit_code != 0:
        print("[DEBUG] Tests failed")
        print(f"[DEBUG] Failed tests: {len(summary.failures)}")
        for i, failure in enumerate(summary.failures):
            print(f"--- Failure {i+1} ---")
            print(f"Test: {failure.nodeid}")
            print(f"Reason: {failure.longreprtext}")
    else:
        print("[DEBUG] All tests passed")
    
    print(f"[SUMMARY] Total tests: {summary.total}, Passed: {summary.passed}, Failed: {summary.failed}, Skipped: {summary.skipped}")
    
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
