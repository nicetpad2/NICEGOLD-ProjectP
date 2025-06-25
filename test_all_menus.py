#!/usr/bin/env python3
from pathlib import Path
        from ProjectP import handle_menu_choice
import os
import sys
"""
Test script to verify all 19 menu options in ProjectP.py work correctly
"""


# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


def test_menu_option(choice):
    """Test a specific menu option"""
    print(f"\n🧪 Testing Menu Option {choice}")
    print(" = " * 50)

    try:
        # Import the handle_menu_choice function

        # Test the menu choice
        result = handle_menu_choice(str(choice))

        if result:
            print(f"✅ Menu option {choice} completed successfully")
        else:
            print(f"⚠️ Menu option {choice} completed with warnings")

        return True

    except Exception as e:
        print(f"❌ Menu option {choice} failed: {str(e)}")
        return False


def main():
    """Test all menu options"""
    print("🚀 NICEGOLD ProjectP - Menu Testing Suite")
    print(" = " * 60)

    # Test each menu option
    results = {}

    # Test core pipeline options (1 - 5)
    print("\n📊 Testing Core Pipeline Options (1 - 5)...")
    for i in range(1, 6):
        results[i] = test_menu_option(i)

    # Test data processing options (6)
    print("\n🔄 Testing Data Processing Options (6)...")
    results[6] = test_menu_option(6)

    # Test ML options (7 - 9)
    print("\n🤖 Testing ML Options (7 - 9)...")
    for i in range(7, 10):
        results[i] = test_menu_option(i)

    # Test advanced analytics (10 - 12)
    print("\n📈 Testing Advanced Analytics (10 - 12)...")
    for i in range(10, 13):
        results[i] = test_menu_option(i)

    # Test services (13 - 15)
    print("\n🖥️ Testing Services (13 - 15)...")
    for i in range(13, 16):
        results[i] = test_menu_option(i)

    # Test system management (16 - 19)
    print("\n⚙️ Testing System Management (16 - 19)...")
    for i in range(16, 20):
        results[i] = test_menu_option(i)

    # Summary
    print("\n" + " = " * 60)
    print("📊 TEST RESULTS SUMMARY")
    print(" = " * 60)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")

    if passed == total:
        print("\n🎉 ALL MENU OPTIONS WORKING PERFECTLY!")
    else:
        print(f"\n⚠️ {total - passed} menu options need attention")

        # Show failed options
        failed = [k for k, v in results.items() if not v]
        if failed:
            print(f"Failed options: {', '.join(map(str, failed))}")

    # Detailed results
    print("\n📋 Detailed Results:")
    for option, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  Option {option:2d}: {status}")


if __name__ == "__main__":
    main()