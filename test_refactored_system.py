#!/usr/bin/env python3
"""
Test script for the refactored ProjectP modular system
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def test_imports():
    """Test all module imports"""
    print("🧪 Testing module imports...")

    try:
        from core.colors import Colors, colorize

        print(
            f"{colorize('✅ Core colors module imported successfully', Colors.BRIGHT_GREEN)}"
        )

        from ui.animations import print_logo, show_loading_animation

        print(
            f"{colorize('✅ UI animations module imported successfully', Colors.BRIGHT_GREEN)}"
        )

        from ui.menu_system import MenuSystem

        print(
            f"{colorize('✅ UI menu system module imported successfully', Colors.BRIGHT_GREEN)}"
        )

        from system.health_monitor import SystemHealthMonitor

        print(
            f"{colorize('✅ System health monitor module imported successfully', Colors.BRIGHT_GREEN)}"
        )

        from commands.pipeline_commands import PipelineCommands

        print(
            f"{colorize('✅ Pipeline commands module imported successfully', Colors.BRIGHT_GREEN)}"
        )

        from commands.analysis_commands import AnalysisCommands

        print(
            f"{colorize('✅ Analysis commands module imported successfully', Colors.BRIGHT_GREEN)}"
        )

        from commands.trading_commands import TradingCommands

        print(
            f"{colorize('✅ Trading commands module imported successfully', Colors.BRIGHT_GREEN)}"
        )

        from commands.ai_commands import AICommands

        print(
            f"{colorize('✅ AI commands module imported successfully', Colors.BRIGHT_GREEN)}"
        )

        return True

    except ImportError as e:
        print(f"{colorize('❌ Import error:', Colors.BRIGHT_RED)} {e}")
        return False


def test_basic_functionality():
    """Test basic functionality of modules"""
    print(f"\n🔧 Testing basic functionality...")

    try:
        from core.colors import Colors, colorize
        from system.health_monitor import SystemHealthMonitor
        from ui.menu_system import MenuSystem

        # Test health monitor
        health_monitor = SystemHealthMonitor()
        health_status = health_monitor.check_system_health()
        print(f"{colorize('✅ Health monitor working', Colors.BRIGHT_GREEN)}")

        # Test menu system
        menu_system = MenuSystem()
        print(f"{colorize('✅ Menu system initialized', Colors.BRIGHT_GREEN)}")

        # Test command validation
        valid_choice = menu_system.validate_choice("1")
        print(f"{colorize('✅ Menu validation working', Colors.BRIGHT_GREEN)}")

        return True

    except Exception as e:
        print(f"{colorize('❌ Functionality error:', Colors.BRIGHT_RED)} {e}")
        return False


def main():
    """Main test function"""
    print(f"🚀 NICEGOLD ProjectP - Modular System Test")
    print(f"=" * 50)

    # Test imports
    imports_ok = test_imports()

    # Test functionality
    functionality_ok = test_basic_functionality()

    # Summary
    print(f"\n📊 Test Summary:")
    print(f"=" * 30)

    if imports_ok and functionality_ok:
        from core.colors import Colors, colorize

        print(
            f"{colorize('🎉 All tests passed! Modular system is working correctly.', Colors.BOLD + Colors.BRIGHT_GREEN)}"
        )
        return 0
    else:
        print(f"❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
