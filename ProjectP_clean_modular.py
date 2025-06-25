#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NICEGOLD ProjectP v2.0 - Main Entry Point
Professional AI Trading System with Modular Architecture

This is the new main entry point that uses the modular architecture.
All functionality has been split into logical modules for better maintainability.
"""

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    """Main entry point for NICEGOLD ProjectP"""
    try:
        # Import the modular menu interface
        from core.menu_interface import menu_interface

        # Run the application
        menu_interface.run()

    except ImportError as e:
        print(f"❌ Failed to import modular system: {e}")
        print("💡 Please ensure all core modules are available:")
        print("   - core/menu_interface.py")
        print("   - core/menu_operations.py")
        print("   - core/config.py")
        print("   - core/system.py")
        print("   - utils/colors.py")
        print("\n🔧 You can try running setup_new.py to install dependencies")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 ขอบคุณที่ใช้งาน NICEGOLD ProjectP!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
