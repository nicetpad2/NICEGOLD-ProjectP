#!/usr/bin/env python3
"""
Test script for NICEGOLD ProjectP menu functions
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.menu_operations import MenuOperations


def test_menu_functions():
    """Test all basic menu functions 1-6"""
    menu_ops = MenuOperations()

    print("🧪 Testing NICEGOLD ProjectP Menu Functions")
    print("=" * 50)

    # Test each function
    functions_to_test = [
        ("1. Full Pipeline", menu_ops.full_pipeline),
        ("2. Data Analysis", menu_ops.data_analysis),
        ("3. Quick Test", menu_ops.quick_test),
        ("4. Health Check", menu_ops.health_check),
        ("5. Install Dependencies", menu_ops.install_dependencies),
        ("6. Clean System", menu_ops.clean_system),
    ]

    results = {}

    for name, func in functions_to_test:
        print(f"\n🔍 Testing {name}...")
        print("-" * 30)
        try:
            result = func()
            results[name] = "✅ PASSED" if result else "❌ FAILED"
            print(f"Result: {results[name]}")
        except Exception as e:
            results[name] = f"💥 ERROR: {str(e)}"
            print(f"Error: {e}")
        print("-" * 30)

    # Summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 50)
    for name, result in results.items():
        print(f"{name}: {result}")

    # Count results
    passed = sum(1 for r in results.values() if "✅" in r)
    failed = sum(1 for r in results.values() if "❌" in r)
    errors = sum(1 for r in results.values() if "💥" in r)

    print(f"\n🎯 FINAL RESULTS:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"💥 Errors: {errors}")
    print(f"📈 Success Rate: {(passed/len(functions_to_test)*100):.1f}%")


if __name__ == "__main__":
    test_menu_functions()
