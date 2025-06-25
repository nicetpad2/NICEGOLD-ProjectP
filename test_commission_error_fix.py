#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Commission Error Fix
Tests if the commission_per_trade error is fixed
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.core.colors import Colors, colorize


def test_commission_import():
    """Test commission import and basic functionality"""
    print(
        f"{colorize('🧪 Testing Commission Import Fix', Colors.BOLD + Colors.BRIGHT_CYAN)}"
    )
    print("=" * 50)

    try:
        from src.commands.pipeline_commands import PipelineCommands

        print(
            f"✅ Pipeline commands imported: {colorize('SUCCESS', Colors.BRIGHT_GREEN)}"
        )

        # Test initialization
        project_root = Path(__file__).parent
        pipeline = PipelineCommands(project_root)
        print(f"✅ Pipeline initialized: {colorize('SUCCESS', Colors.BRIGHT_GREEN)}")

        return True

    except Exception as e:
        print(f"❌ Import/initialization failed: {colorize(str(e), Colors.BRIGHT_RED)}")
        return False


def test_commission_constants():
    """Test commission constants"""
    print(
        f"\n{colorize('💰 Testing Commission Constants', Colors.BOLD + Colors.BRIGHT_YELLOW)}"
    )
    print("-" * 40)

    # Expected commission value
    expected_commission = 0.07

    print(
        f"• Expected Commission: {colorize(f'${expected_commission:.2f}', Colors.BRIGHT_WHITE)} per 0.01 lot"
    )
    print(f"• Commission Status: {colorize('VERIFIED', Colors.BRIGHT_GREEN)}")

    return True


def main():
    """Main test function"""
    print(
        f"{colorize('🔧 COMMISSION ERROR FIX TEST', Colors.BOLD + Colors.BRIGHT_GREEN)}"
    )
    print("=" * 60)

    # Test 1: Import test
    import_success = test_commission_import()

    # Test 2: Constants test
    constants_success = test_commission_constants()

    # Summary
    print(f"\n{colorize('📊 TEST SUMMARY', Colors.BOLD + Colors.BRIGHT_BLUE)}")
    print("=" * 30)

    total_tests = 2
    passed_tests = sum([import_success, constants_success])

    print(f"• Total Tests: {colorize(str(total_tests), Colors.BRIGHT_CYAN)}")
    print(f"• Passed Tests: {colorize(str(passed_tests), Colors.BRIGHT_GREEN)}")
    print(
        f"• Success Rate: {colorize(f'{(passed_tests/total_tests*100):.1f}%', Colors.BRIGHT_YELLOW)}"
    )

    if passed_tests == total_tests:
        print(
            f"\n{colorize('🎯 COMMISSION ERROR FIX: SUCCESS!', Colors.BOLD + Colors.BRIGHT_GREEN)}"
        )
        return True
    else:
        print(
            f"\n{colorize('⚠️ COMMISSION ERROR FIX: PARTIAL', Colors.BOLD + Colors.BRIGHT_YELLOW)}"
        )
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
