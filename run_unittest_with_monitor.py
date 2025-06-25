
from pathlib import Path
import importlib
import json
import os
import re
import subprocess
import sys
import traceback
import unittest
def find_test_files():
    """Find all test files in the tests directory"""
    tests_dir = os.path.join(os.path.dirname(__file__), 'tests')
    test_files = []

    for root, _, files in os.walk(tests_dir):
        for file in files:
            if file.startswith('test_') and file.endswith('.py'):
                # Get relative path from project root
                rel_path = os.path.relpath(os.path.join(root, file))
                # Convert to module path
                module_path = rel_path.replace(os.sep, '.').replace('.py', '')
                test_files.append(module_path)

    return test_files

def run_test_file(test_module):
    """Run a single test file and return results"""
    try:
        module = importlib.import_module(test_module)
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)

        # Create a test runner that will capture results
        runner = unittest.TextTestRunner(verbosity = 2)
        result = runner.run(suite)

        return {
            'module': test_module, 
            'run': result.testsRun, 
            'errors': [(str(test), str(err)) for test, err in result.errors], 
            'failures': [(str(test), str(err)) for test, err in result.failures], 
            'skipped': len(result.skipped), 
            'success': result.wasSuccessful()
        }
    except Exception as e:
        print(f"Failed to run test module {test_module}: {e}")
        traceback.print_exc()
        return {
            'module': test_module, 
            'run': 0, 
            'errors': [('module_import', str(e))], 
            'failures': [], 
            'skipped': 0, 
            'success': False
        }

def fix_failure(module_name, test_name, error_msg):
    """Attempt to fix a failing test"""
    print(f"\nTrying to fix: {module_name} - {test_name}")
    print(f"Error: {error_msg}")

    # Extract a cleaner error message
    error_lines = error_msg.split("\n")
    clean_error = "Unknown error"
    for line in error_lines:
        if "AssertionError:" in line:
            clean_error = line.split("AssertionError:", 1)[1].strip()
            break

    print(f"Simplified error: {clean_error}")

    # Try to locate the test file
    module_file = module_name.replace('.', os.sep) + '.py'
    if not os.path.exists(module_file):
        print(f"Cannot find test file: {module_file}")
        return False

    # Read the file content
    with open(module_file, 'r', encoding = 'utf - 8') as f:
        content = f.read()

    # Extract the test method name from the test name
    # Format is usually like "test_something (tests.test_module.TestClass)"
    match = re.search(r'(\w + ) \(.*\)', test_name)
    if match:
        test_method = match.group(1)
        print(f"Looking for test method: {test_method}")

        # Find the test method in the file
        test_pattern = re.compile(f"def {test_method}\\(.*\\):")
        match = test_pattern.search(content)
        if match:
            start_pos = match.start()
            print(f"Found test method at position: {start_pos}")

            # This is where you would implement logic to fix the test
            # For now, we'll just print that we found it
            print(f"Found test method, but automatic fixing is not yet implemented")
            return False

    print("Could not determine how to fix the test")
    return False

def run_coverage():
    """รัน pytest พร้อมกับ coverage และวิเคราะห์ผล"""
    try:
        print("\n" + " = " * 80)
        print("วัด Code Coverage")
        print(" = " * 80)

        # รัน pytest พร้อมกับ coverage
        command = [
            sys.executable, 
            ' - m', 'pytest', 
            'tests',  # รันเทสต์ทั้งหมด
            ' -  - cov = .',  # วัด coverage ของโค้ดทั้งหมด
            ' -  - cov - report = json:coverage.json',  # บันทึกเป็น JSON
            ' - q'  # quiet mode
        ]

        print(f"Running: {' '.join(command)}")
        result = subprocess.run(command, capture_output = True, text = True)

        if result.returncode != 0:
            print(f"Error running coverage: {result.stderr}")
            return None

        # อ่านผลจาก coverage.json
        if os.path.exists('coverage.json'):
            with open('coverage.json', 'r') as f:
                cov_data = json.load(f)

            total_statements = 0
            covered_statements = 0
            files_with_cov = 0
            total_files = 0

            # วิเคราะห์ข้อมูล coverage
            uncovered_files = []
            low_cov_files = []
            high_cov_files = []
            for file_path, data in cov_data['files'].items():
                # ข้ามไฟล์ในไดเรกทอรี .venv หรือไฟล์เทสต์เอง
                if '.venv' in file_path or '/tests/' in file_path or '\\tests\\' in file_path:
                    continue

                # ข้ามไฟล์ที่เป็น site - packages
                if 'site - packages' in file_path:
                    continue

                total_files += 1
                executed = data.get('executed_lines', [])
                missing = data.get('missing_lines', [])

                # นับจำนวน statement ทั้งหมดและที่ถูก cover
                file_total = len(executed) + len(missing)
                file_covered = len(executed)
                total_statements += file_total
                covered_statements += file_covered

                if file_total > 0:
                    cov_percent = (file_covered / file_total) * 100
                    rel_path = file_path.replace('\\', '/')
                    if file_path.startswith('g:'):  # ถ้าเป็น absolute path
                        rel_path = os.path.relpath(file_path)

                    if file_covered > 0:
                        files_with_cov += 1
                        if cov_percent < 50:
                            low_cov_files.append((rel_path, cov_percent))
                        else:
                            high_cov_files.append((rel_path, cov_percent))
                    else:
                        uncovered_files.append(rel_path)

            # แสดงผลสรุป
            total_cov = 0
            if total_statements > 0:
                total_cov = (covered_statements / total_statements) * 100

            print(f"\n =  =  = = Coverage Summary = =  =  = ")
            print(f"จำนวนไฟล์ทั้งหมด: {total_files}")
            print(f"ไฟล์ที่มี coverage: {files_with_cov} ({files_with_cov / total_files * 100:.1f}%)")
            print(f"จำนวน statement ทั้งหมด: {total_statements}")
            print(f"Statement ที่ถูก cover: {covered_statements}")
            print(f"Total Coverage: {total_cov:.1f}%")

            # แสดงไฟล์ที่ควรเพิ่ม coverage
            if total_cov < 30:
                print("\n =  =  = = ไฟล์ที่ควรทำ test เพิ่ม = =  =  = ")

                # เรียงไฟล์ที่มี coverage ต่ำ
                low_cov_files.sort(key = lambda x: x[1])

                # แสดงไฟล์ที่มี coverage ต่ำ
                if low_cov_files:
                    print("\nไฟล์ที่มี coverage ต่ำ:")
                    for file_path, cov in low_cov_files[:10]:  # แสดง 10 ไฟล์แรก
                        print(f"  {file_path}: {cov:.1f}%")

                # แสดงไฟล์ที่ไม่มี coverage
                if uncovered_files:
                    print("\nไฟล์ที่สำคัญที่ไม่มี test:")
                    for file_path in uncovered_files[:10]:  # แสดง 10 ไฟล์แรก
                        if any(keyword in file_path for keyword in ['utils', 'config', 'core', 'base']):
                            print(f"  {file_path}")

                # แนะนำไฟล์ที่ควรทำ test ก่อน
                print("\nแนะนำให้เพิ่ม test กับ utility functions และ core modules ก่อน")

            return {
                'total_coverage': total_cov, 
                'files_covered': files_with_cov, 
                'total_files': total_files, 
                'covered_statements': covered_statements, 
                'total_statements': total_statements
            }
        else:
            print("ไม่พบไฟล์ coverage.json")
            return None
    except Exception as e:
        print(f"Error in run_coverage: {e}")
        traceback.print_exc()
        return None

def recommend_next_tests(coverage_data):
    """แนะนำไฟล์ที่ควรทำ test เพิ่มเติมเพื่อให้ถึง 30% coverage"""
    if not coverage_data or coverage_data['total_coverage'] >= 30:
        return

    # คำนวณว่าต้องทำ test ครอบคลุมเพิ่มอีกกี่ statement
    current_covered = coverage_data['covered_statements']
    total = coverage_data['total_statements']
    target = 0.3 * total  # 30% ของ statement ทั้งหมด
    needed = target - current_covered

    print("\n =  =  = = คำแนะนำในการเพิ่ม Test Coverage = =  =  = ")
    print(f"ปัจจุบัน: {coverage_data['total_coverage']:.1f}%")
    print(f"เป้าหมาย: 30%")
    print(f"ต้องทำ test ครอบคลุมเพิ่มอีก {needed:.0f} statements")

    # แนะนำโมดูลที่ควรทำ test
    print("\nโมดูลที่แนะนำให้ทำ test:")
    print("1. src/utils/ - utility functions มักจะทำ test ง่ายและได้ coverage เยอะ")
    print("2. projectp/ - โมดูลหลักของโปรเจค")
    print("3. src/config.py - configuration functions")

    # สร้างคำสั่งรัน pytest สำหรับแต่ละโมดูล
    print("\nคำสั่งรัน pytest ที่แนะนำ:")
    print(f"{sys.executable} -m pytest tests/test_new_file.py - - cov = src.utils - - cov - report = term")

def main():
    print(" = " * 80)
    print("Running all tests and monitoring for failures")
    print(" = " * 80)

    # รัน coverage analysis ก่อน
    coverage_data = run_coverage()
    if coverage_data:
        recommend_next_tests(coverage_data)

    test_files = find_test_files()
    print(f"Found {len(test_files)} test files")

    total_run = 0
    total_errors = 0
    total_failures = 0
    total_success = 0

    # First, run some basic/core tests to see if the system is functional
    basic_tests = [f for f in test_files if 'test_utils.py' in f or 'test_config' in f]
    for test_module in basic_tests:
        test_files.remove(test_module)
        print(f"\nRunning core test: {test_module}")
        result = run_test_file(test_module)

        total_run += result['run']
        total_errors += len(result['errors'])
        total_failures += len(result['failures'])
        if result['success']:
            total_success += 1

        if not result['success']:
            print(f"Failure in core test {test_module} - needs immediate attention")
            for test, error in result['errors'] + result['failures']:
                fixed = fix_failure(test_module, test, error)
                if fixed:
                    print("Test was automatically fixed. Re - running...")
                    # We would re - run the test here if fixing was implemented
                else:
                    print("Could not automatically fix the test.")
        else:
            print(f"Core test {test_module} passed successfully!")

    # Then run the rest of the tests
    for test_module in test_files:
        print(f"\nRunning test: {test_module}")
        result = run_test_file(test_module)

        total_run += result['run']
        total_errors += len(result['errors'])
        total_failures += len(result['failures'])
        if result['success']:
            total_success += 1

        if not result['success']:
            print(f"Failure in test {test_module}")
            for test, error in result['errors'] + result['failures']:
                fixed = fix_failure(test_module, test, error)
                if fixed:
                    print("Test was automatically fixed. Re - running...")
                    # We would re - run the test here if fixing was implemented
                else:
                    print("Could not automatically fix the test.")

    # Summary
    print("\n" + " = " * 80)
    print(f"SUMMARY: Ran {total_run} tests in {len(test_files) + len(basic_tests)} files")
    print(f"Success: {total_success} files")
    print(f"Files with errors/failures: {len(test_files) + len(basic_tests) - total_success}")
    print(f"Total errors: {total_errors}")
    print(f"Total failures: {total_failures}")
    print(" = " * 80)

    # รัน coverage analysis หลังจากรัน tests
    if coverage_data and coverage_data['total_coverage'] < 30:
        print("\nคุณยังต้องเพิ่ม test cases เพื่อให้ coverage ถึง 30%")
        print(f"ปัจจุบัน coverage: {coverage_data['total_coverage']:.1f}%")

    if total_errors > 0 or total_failures > 0:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())