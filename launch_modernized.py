#!/usr/bin/env python3
"""
🚀 NICEGOLD ProjectP - Modernized Launch Script
Experience the beautiful, fast, modernized UI system!
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


def launch_modernized_system():
    """Launch the modernized NICEGOLD ProjectP system"""

    print("\n🚀 Launching NICEGOLD ProjectP Modernized System...")
    print("=" * 60)

    try:
        # Import and launch the modernized menu interface
        from core.menu_interface import MenuInterface

        print("✅ Modernized system loaded successfully!")
        print("🎨 Preparing beautiful UI experience...")
        print("📊 Enhanced progress bars ready...")
        print("⚡ Fast pipeline execution enabled...")
        print("\n🎭 Welcome to the modernized NICEGOLD ProjectP!\n")

        # Launch the main menu interface
        menu = MenuInterface()
        menu.run()

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 Fallback: Running basic system...")

        # Fallback to basic main.py
        try:
            import main

            main.main()
        except Exception as fallback_error:
            print(f"❌ Fallback failed: {fallback_error}")
            print("\n📝 Please check system requirements and try again")

    except KeyboardInterrupt:
        print("\n\n👋 System shutdown by user")

    except Exception as e:
        print(f"\n❌ System error: {e}")
        print("📝 Please check the error logs for details")


if __name__ == "__main__":
    launch_modernized_system()
