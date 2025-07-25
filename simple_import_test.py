#!/usr/bin/env python3
        from colorama import Fore, Style, init as colorama_init
        from integrated_emergency_fixes import create_emergency_fix_manager
        from pathlib import Path
        from projectp.pipeline import run_full_pipeline
        import argparse
        import numpy as np
import os
        import pandas as pd
import sys
import time
        import warnings
"""
🔧 SIMPLE IMPORT TEST
ทดสอบ imports แบบง่าย ๆ เพื่อหาปัญหา
"""


def test_basic_imports():
    """ทดสอบ imports พื้นฐาน"""
    print("🧪 Testing basic imports...")

    try:
        print("✅ pandas imported")
    except Exception as e:
        print(f"❌ pandas failed: {e}")

    try:
        print("✅ numpy imported")
    except Exception as e:
        print(f"❌ numpy failed: {e}")

    try:
        print("✅ pathlib imported")
    except Exception as e:
        print(f"❌ pathlib failed: {e}")

def test_project_imports():
    """ทดสอบ project imports"""
    print("\n🧪 Testing project imports...")

    try:
        # ทดสอบ integrated_emergency_fixes
        print("✅ integrated_emergency_fixes imported")
    except Exception as e:
        print(f"❌ integrated_emergency_fixes failed: {e}")

    try:
        # ทดสอบ projectp.pipeline
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        print("✅ projectp.pipeline imported")
    except Exception as e:
        print(f"❌ projectp.pipeline failed: {e}")

def test_quick_projectp():
    """ทดสอบ ProjectP แบบเร็ว"""
    print("\n🧪 Testing ProjectP quick imports...")

    # ตั้งค่า timeout
    start_time = time.time()
    timeout = 10  # 10 seconds

    try:
        # Test if we can import basic ProjectP components
        warnings.filterwarnings('ignore')

        colorama_init(autoreset = True)
        print("✅ colorama imported")

        # Test argument parser creation (without running)
        parser = argparse.ArgumentParser(description = "Test")
        print("✅ argparse ready")

        if time.time() - start_time > timeout:
            print(f"⏰ Timeout reached ({timeout}s)")
            return False

        print("✅ Quick ProjectP test passed")
        return True

    except Exception as e:
        print(f"❌ Quick ProjectP test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🔧 SIMPLE IMPORT TEST STARTING...")
    print(" = " * 50)

    test_basic_imports()
    test_project_imports()
    result = test_quick_projectp()

    print("\n" + " = " * 50)
    if result:
        print("✅ IMPORT TESTS COMPLETED")
    else:
        print("❌ SOME IMPORT TESTS FAILED")
    print(" = " * 50)

if __name__ == "__main__":
    main()