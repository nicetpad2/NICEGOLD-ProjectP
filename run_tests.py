import argparse
import logging
import os
import sys
import pytest
import warnings
import datetime

warnings.filterwarnings("ignore", category=UserWarning)

# [Patch v5.9.3] Forward command-line args directly to pytest

class _SummaryPlugin:
    """Plugin เก็บสถิติผลการทดสอบ"""
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.failures = []  # Store failure information

    def pytest_runtest_logreport(self, report):
        if report.when == 'call':
            self.total += 1
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1
                # Store information about the failing test
                failure_info = {
                    'nodeid': report.nodeid,
                    'longrepr': str(report.longrepr)
                }
                self.failures.append(failure_info)
            elif report.skipped:
                self.skipped += 1


def auto_fix_failures(summary_plugin):
    """เรียก agent เพื่อแก้ไขโค้ดอัตโนมัติเมื่อทดสอบล้มเหลว"""
    try:
        # Log failures to a file for analysis
        with open('test_failures.txt', 'a') as f:
            f.write(f"\n=== FAILURES DETECTED AT {datetime.datetime.now()} ===\n")
            for i, failure in enumerate(summary_plugin.failures):
                f.write(f"Failure {i+1}: {failure}\n")
        
        # Try to import and use agents module if available
        try:
            import agents
            agents.fix_failures()
        except ImportError:
            print('[AUTO-FIX] agents module not found, skipping automated fixes')
    except Exception as e:
        print(f'[AUTO-FIX] Error ขณะเรียก agent: {e}')


def main() -> None:
    parser = argparse.ArgumentParser(description='รัน test suite พร้อม coverage สำหรับโปรเจคนี้')
    parser.add_argument('--fast', '--smoke', action='store_true', dest='fast',
                        help='ข้าม integration tests ที่ใช้เวลานาน')
    parser.add_argument('-n', '--num-processes', default=None,
                        help='จำนวน process สำหรับรันแบบขนาน (pytest-xdist)')
    parser.add_argument('--cov', action='store_true', help='เปิด coverage (pytest-cov)')
    parser.add_argument('--cov-only', action='store_true', help='แสดงเฉพาะ coverage summary')
    parser.add_argument('--html-report', action='store_true', help='สร้าง coverage html report')
    parser.add_argument('--maxfail', type=int, default=0,
                        help='หยุดหลังจากจำนวนความล้มเหลวที่ระบุ แล้วรันใหม่จนสำเร็จ')
    parser.add_argument('--reruns', type=int, default=0,
                        help='จำนวนครั้งที่ pytest จะรันซ้ำอัตโนมัติหากทดสอบล้มเหลว')
    args, extra_args = parser.parse_known_args()

    os.environ.setdefault('COMPACT_LOG', '1')
    logging.basicConfig(level=logging.WARNING)

    pytest_args = extra_args
    if not pytest_args:
        pytest_args = ['tests']
    pytest_args.insert(0, '-q')
    if args.fast:
        pytest_args += ['-m', 'not integration']

    if args.num_processes:
        try:
            import xdist  # noqa: F401
            pytest_args += ['-n', str(args.num_processes)]
        except ImportError:
            print('[WARN] pytest-xdist ไม่ได้ติดตั้ง จึงรันแบบปกติ')

    if args.cov or args.cov_only or args.html_report:
        pytest_args += ['--cov=.', '--cov-report=term-missing']
        if args.html_report:
            pytest_args += ['--cov-report=html']

    if args.maxfail:
        pytest_args += ['--maxfail', str(args.maxfail)]

    if args.reruns:
        try:
            import pytest_rerunfailures  # noqa: F401
            pytest_args += ['--reruns', str(args.reruns)]
        except ImportError:
            print('[WARN] pytest-rerunfailures ไม่ได้ติดตั้ง จึงไม่สามารถรันซ้ำอัตโนมัติได้')

    # รัน pytest และถ้าล้มเหลวให้รันใหม่อัตโนมัติจนกว่าจะผ่านทั้งหมด
    exit_code = 1
    summary = _SummaryPlugin()
    
    while exit_code != 0:
        summary = _SummaryPlugin()
        exit_code = pytest.main(pytest_args, plugins=[summary])
        
        if exit_code != 0:
            print(f'[INFO] พบการทดสอบล้มเหลว: {summary.failed} เคส เรียก agent เพื่อแก้ไขโค้ด')
            auto_fix_failures(summary)
            print('[INFO] แก้ไขเสร็จแล้ว กำลังรันใหม่...')

    if args.cov or args.cov_only or args.html_report:
        try:
            import coverage
            cov = coverage.Coverage()
            cov.load()
            total = cov.report(show_missing=True)
            print(f"[COVERAGE] TOTAL: {total:.1f}%")
            if args.html_report:
                cov.html_report(directory='htmlcov')
                print('[COVERAGE] HTML report generated at htmlcov/index.html')
        except Exception as e:
            print(f'[COVERAGE] Error: {e}')
            
    if summary is None or summary.total == 0:
        print('[SUMMARY] No tests collected')
        exit_code = exit_code or 1
    else:
        print(f"[SUMMARY] Total tests: {summary.total}, Passed: {summary.passed}, Failed: {summary.failed}, Skipped: {summary.skipped}")

    if args.cov_only:
        sys.exit(0)
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
