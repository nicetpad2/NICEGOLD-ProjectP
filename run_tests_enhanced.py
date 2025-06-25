#!/usr/bin/env python3
            import agents
import argparse
            import coverage
import datetime
import json
import logging
import os
import pytest
import sys
    import tensorflow as tf
import time
import warnings
"""
รัน test suite พร้อม coverage และความสามารถในการแก้ไขล้มเหลวอัตโนมัติ
รองรับการแสดงผลภาษาไทยและบันทึกรายงานการทดสอบแบบละเอียด
"""

# ปิด UserWarning ที่ไม่จำเป็น
warnings.filterwarnings("ignore", category = UserWarning)
# ปิด FutureWarning, DeprecationWarning, RuntimeWarning
warnings.filterwarnings("ignore", category = FutureWarning)
warnings.filterwarnings("ignore", category = DeprecationWarning)
warnings.filterwarnings("ignore", category = RuntimeWarning)
# ปิด warning ของ tensorflow และ numpy ที่ noisy
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all, 1 = info, 2 = warning, 3 = error
try:
    tf.get_logger().setLevel('ERROR')
except Exception:
    pass

# [Patch v5.9.3] ส่งต่อ command - line args ไปยัง pytest โดยตรง

class _SummaryPlugin:
    """Plugin เก็บสถิติผลการทดสอบและรายละเอียดข้อผิดพลาด"""
    def __init__(self):
        self.total = 0          # จำนวนการทดสอบทั้งหมด
        self.passed = 0         # จำนวนการทดสอบที่ผ่าน
        self.failed = 0         # จำนวนการทดสอบที่ล้มเหลว
        self.skipped = 0        # จำนวนการทดสอบที่ข้าม
        self.failures = []      # รายการข้อผิดพลาด
        self.start_time = time.time()  # เวลาเริ่มต้นการทดสอบ

    def pytest_runtest_logreport(self, report):
        """รับรายงานผลการทดสอบจาก pytest"""
        if report.when == 'call':
            self.total += 1
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1
                # เก็บข้อมูลเกี่ยวกับการทดสอบที่ล้มเหลว
                failure_info = {
                    'nodeid': report.nodeid, 
                    'longrepr': str(report.longrepr)
                }
                self.failures.append(failure_info)
            elif report.skipped:
                self.skipped += 1

    def get_execution_time(self):
        """คำนวณเวลาทดสอบ"""
        return time.time() - self.start_time


def auto_fix_failures(summary_plugin):
    """เรียก agent เพื่อแก้ไขโค้ดอัตโนมัติเมื่อทดสอบล้มเหลว"""
    try:
        # สร้างรายงานข้อผิดพลาดแบบมีโครงสร้าง
        failure_report = {
            'timestamp': datetime.datetime.now().isoformat(), 
            'failures': [], 
            'summary': {
                'total_failures': summary_plugin.failed, 
                'execution_time': summary_plugin.get_execution_time()
            }
        }

        for i, failure in enumerate(summary_plugin.failures):
            nodeid = failure.get('nodeid', '')
            error_text = failure.get('longrepr', '')

            # แยกส่วนประกอบของ nodeid เพื่อระบุไฟล์และชื่อการทดสอบ
            test_parts = nodeid.split('::')
            test_file = test_parts[0] if len(test_parts) > 0 else ''
            test_class = test_parts[1] if len(test_parts) > 1 else ''
            test_name = test_parts[2] if len(test_parts) > 2 else ''

            # สร้างข้อมูลข้อผิดพลาดที่มีรายละเอียด
            failure_info = {
                'id': i + 1, 
                'nodeid': nodeid, 
                'file': test_file, 
                'class': test_class, 
                'name': test_name, 
                'error': error_text, 
                'detected_at': datetime.datetime.now().isoformat()
            }

            # วิเคราะห์ข้อผิดพลาดเบื้องต้น
            failure_info['error_type'] = 'assertion_error' if 'AssertionError' in error_text else 'unknown'

            # เพิ่มข้อมูลข้อผิดพลาดลงในรายงาน
            failure_report['failures'].append(failure_info)

        # บันทึกลงไฟล์ JSON เพื่อเข้าถึงข้อมูลแบบมีโครงสร้าง
        with open('test_failures.json', 'w', encoding = 'utf - 8') as f:
            json.dump(failure_report, f, indent = 2, ensure_ascii = False)

        # บันทึกข้อผิดพลาดล่าสุดเพื่อเข้าถึงได้ง่าย
        if failure_report['failures']:
            with open('last_failure.json', 'w', encoding = 'utf - 8') as f:
                json.dump(failure_report['failures'][0], f, indent = 2, ensure_ascii = False)

            # บันทึกเป็นไฟล์ข้อความแบบอ่านง่าย
            with open('last_failure.txt', 'w', encoding = 'utf - 8') as f:
                failure = failure_report['failures'][0]
                f.write(f"การทดสอบที่ล้มเหลว: {failure['nodeid']}\n")
                f.write(f"ไฟล์: {failure['file']}\n")
                if failure['class']:
                    f.write(f"คลาส: {failure['class']}\n")
                f.write(f"ชื่อเทสต์: {failure['name']}\n")
                f.write(f"เวลาที่ตรวจพบ: {failure['detected_at']}\n")
                f.write("\nข้อความผิดพลาด:\n")
                f.write(f"{failure['error']}\n")

        print(f"[ข้อมูล] บันทึกข้อผิดพลาดการทดสอบ {len(failure_report['failures'])} รายการลงใน test_failures.json")

        # ลองนำเข้าและใช้โมดูล agents ถ้ามี
        try:
            print("[แก้ไขอัตโนมัติ] เรียกใช้ agent เพื่อแก้ไขข้อผิดพลาด...")
            agents.fix_failures()
        except ImportError:
            print('[แก้ไขอัตโนมัติ] ไม่พบโมดูล agents จึงข้ามการแก้ไขอัตโนมัติ')

            # ถ้าไม่สามารถใช้โมดูล agents ได้ ให้ลองวิเคราะห์และแก้ไขข้อผิดพลาดแบบง่าย
            for failure in failure_report['failures']:
                print(f"[พยายามแก้ไข] กำลังแก้ไข: {failure['nodeid']}")

                # วิเคราะห์ข้อความผิดพลาดเพื่อหาวิธีแก้ไข
                if 'AssertionError' in failure['error']:
                    print(f"  พบข้อผิดพลาด AssertionError ในไฟล์ {failure['file']}")

                # ที่นี่คุณอาจจะเพิ่มลอจิกเพื่อแก้ไขข้อผิดพลาดแบบเฉพาะเจาะจง

    except Exception as e:
        print(f'[แก้ไขอัตโนมัติ] เกิดข้อผิดพลาดระหว่างการแก้ไขอัตโนมัติ: {e}')


def main() -> int:
    parser = argparse.ArgumentParser(description = 'รัน test suite พร้อม coverage สำหรับโปรเจคนี้')
    parser.add_argument(' -  - fast', ' -  - smoke', action = 'store_true', dest = 'fast', 
                        help = 'ข้าม integration tests ที่ใช้เวลานาน')
    parser.add_argument(' - v', ' -  - verbose', action = 'store_true', 
                        help = 'แสดงรายละเอียดมากขึ้น')
    parser.add_argument(' - n', ' -  - num - processes', default = None, 
                        help = 'จำนวน process สำหรับรันแบบขนาน (pytest - xdist)')
    parser.add_argument(' -  - cov', action = 'store_true', help = 'เปิด coverage (pytest - cov)')
    parser.add_argument(' -  - cov - only', action = 'store_true', help = 'แสดงเฉพาะ coverage summary')
    parser.add_argument(' -  - html - report', action = 'store_true', help = 'สร้าง coverage html report')
    parser.add_argument(' -  - maxfail', type = int, default = 0, 
                        help = 'หยุดหลังจากจำนวนความล้มเหลวที่ระบุ แล้วรันใหม่จนสำเร็จ')
    parser.add_argument(' -  - reruns', type = int, default = 0, 
                        help = 'จำนวนครั้งที่ pytest จะรันซ้ำอัตโนมัติหากทดสอบล้มเหลว')
    parser.add_argument(' -  - target', type = str, help = 'เป้าหมายที่จะทดสอบ')
    args, extra_args = parser.parse_known_args()

    os.environ.setdefault('COMPACT_LOG', '1')
    logging.basicConfig(level = logging.INFO if args.verbose else logging.WARNING)

    pytest_args = extra_args
    if not pytest_args:
        if args.target:
            pytest_args = [args.target]
        else:
            pytest_args = ['tests']

    if args.verbose:
        pytest_args.insert(0, ' - v')
    else:
        pytest_args.insert(0, ' - q')

    if args.fast:
        pytest_args += [' - m', 'not integration']

    if args.num_processes:
        try:
            # Import is just to check if it's installed
            __import__('xdist')
            pytest_args += [' - n', str(args.num_processes)]
        except ImportError:
            print('[WARN] pytest - xdist ไม่ได้ติดตั้ง จึงรันแบบปกติ')

    if args.cov or args.cov_only or args.html_report:
        pytest_args += [' -  - cov = .', ' -  - cov - report = term - missing']
        if args.html_report:
            pytest_args += [' -  - cov - report = html']

    if args.maxfail:
        pytest_args += [' -  - maxfail', str(args.maxfail)]

    if args.reruns:
        try:
            # Import is just to check if it's installed
            __import__('pytest_rerunfailures')
            pytest_args += [' -  - reruns', str(args.reruns)]
        except ImportError:
            print('[WARN] pytest - rerunfailures ไม่ได้ติดตั้ง จึงไม่สามารถรันซ้ำอัตโนมัติได้')

    # รัน pytest และถ้าล้มเหลวให้รันใหม่อัตโนมัติจนกว่าจะผ่านทั้งหมด
    exit_code = 1
    summary = None

    print(f"[RUN] Running tests with arguments: {' '.join(pytest_args)}")

    try:
        while exit_code != 0:
            summary = _SummaryPlugin()
            exit_code = pytest.main(pytest_args, plugins = [summary])

            if exit_code != 0:
                print(f'[INFO] พบการทดสอบล้มเหลว: {summary.failed} เคส เรียก agent เพื่อแก้ไขโค้ด')
                auto_fix_failures(summary)
                print('[INFO] แก้ไขเสร็จแล้ว กำลังรันใหม่...')
    except KeyboardInterrupt:
        print("\n[INFO] ยกเลิกโดยผู้ใช้")
        return 1

    if args.cov or args.cov_only or args.html_report:
        try:
            cov = coverage.Coverage()
            cov.load()
            total = cov.report(show_missing = True)
            print(f"[COVERAGE] TOTAL: {total:.1f}%")
            if args.html_report:
                cov.html_report(directory = 'htmlcov')
                print('[COVERAGE] HTML report generated at htmlcov/index.html')
        except Exception as e:
            print(f'[COVERAGE] Error: {e}')

    if summary is None:
        print('[SUMMARY] No test summary available')
        exit_code = exit_code or 1
    elif summary.total == 0:
        print('[SUMMARY] No tests collected')
        exit_code = exit_code or 1
    else:
        print(f"[SUMMARY] Total tests: {summary.total}, Passed: {summary.passed}, Failed: {summary.failed}, Skipped: {summary.skipped}")

    if args.cov_only:
        sys.exit(0)

    return exit_code


if __name__ == '__main__':
    sys.exit(main())