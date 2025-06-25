#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NICEGOLD ProjectP v2.0 - Professional AI Trading System
Main entry point with modular architecture

Author: NICEGOLD Enterprise
Version: 2.0
Date: June 24, 2025
"""

import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Initialize modern logger
try:
    from utils.simple_logger import (
        critical,
        error,
        info,
        setup_logger,
        success,
        warning,
    )

    # Setup the global logger with enhanced features
    logger = setup_logger(
        name="NICEGOLD_ProjectP",
        enable_file_logging=True,
        enable_sound=False  # Disabled for terminal environments
    )
    MODERN_LOGGER_AVAILABLE = True
    
    # Welcome message with modern logger
    success("🚀 NICEGOLD ProjectP v2.0 - Modern Logger Initialized")
    info("📊 Advanced terminal logging with progress bars enabled")
    
except ImportError as e:
    MODERN_LOGGER_AVAILABLE = False
    print("⚠️ Modern logger not available, using fallback logging")
    print(f"   Error: {e}")
    
    # Fallback logger functions
    def info(msg, **kwargs):
        print(f"ℹ️ [INFO] {msg}")

    def success(msg, **kwargs):
        print(f"✅ [SUCCESS] {msg}")

    def warning(msg, **kwargs):
        print(f"⚠️ [WARNING] {msg}")

    def error(msg, **kwargs):
        print(f"❌ [ERROR] {msg}")

    def critical(msg, **kwargs):
        print(f"🚨 [CRITICAL] {msg}")
    
    logger = None

# Import safe input handler
try:
    from utils.input_handler import safe_input
except ImportError:
    # Fallback safe_input if utils not available
    def safe_input(prompt="", default="", timeout=None):
        """Fallback safe input function"""
        import sys

        try:
            if not sys.stdin.isatty():
                print(f"{prompt}[Non-interactive mode - using default: {default}]")
                return default
            return input(prompt)
        except EOFError:
            print(f"\n[EOFError - using default: {default}]")
            return default
        except KeyboardInterrupt:
            print("\n[Interrupted by user]")
            raise
        except Exception as e:
            print(f"\n[Input error: {e} - using default: {default}]")
            return default


class ProjectPApplication:
    """Main application class for NICEGOLD ProjectP"""

    def __init__(self):
        """Initialize the application"""
        self.start_time = datetime.now()
        self.core_available = False
        self.advanced_logger_available = False
        self.ai_agents_available = False

        # Initialize color system
        self.init_colors()

        # Try to load modules
        self.load_modules()

        # Setup environment
        self.setup_environment()

    def init_colors(self):
        """Initialize color system"""
        try:
            from utils.colors import Colors, colorize

            self.Colors = Colors
            self.colorize = colorize
        except ImportError:
            # Fallback color system
            class Colors:
                RESET = "\033[0m"
                BRIGHT_CYAN = "\033[96m"
                BRIGHT_MAGENTA = "\033[95m"
                BRIGHT_GREEN = "\033[92m"
                BRIGHT_YELLOW = "\033[93m"
                BRIGHT_BLUE = "\033[94m"
                BRIGHT_RED = "\033[91m"
                WHITE = "\033[97m"
                DIM = "\033[2m"

            self.Colors = Colors
            self.colorize = lambda text, color: f"{color}{text}{Colors.RESET}"

    def load_modules(self):
        """Load available modules with modern logging"""
        # Try to load core modules
        try:
            from core.config import get_config
            from core.menu_interface import MenuInterface
            from core.menu_operations import MenuOperations
            from core.system import get_system

            self.config = get_config()
            self.system = get_system()
            self.menu_interface = MenuInterface()
            self.menu_operations = MenuOperations()
            self.core_available = True
            success("Core modules loaded successfully")
        except ImportError as e:
            error(f"Core modules not available: {e}")
            self.config = None
            self.system = None
            self.menu_interface = None
            self.menu_operations = None

        # Modern logger availability (already checked globally)
        if MODERN_LOGGER_AVAILABLE:
            self.advanced_logger_available = True
            success("Modern logger available")
        else:
            # Try fallback advanced logger
            try:
                from src.advanced_logger import get_logger
                self.logger = get_logger()
                self.advanced_logger_available = True
                warning("Using fallback advanced logger")
            except ImportError:
                self.logger = None
                warning("Advanced logger not available")

        # Try to load AI agents
        try:
            from ai_agents_web_ultimate import run_ai_agents_analysis
            self.run_ai_agents_analysis = run_ai_agents_analysis
            self.ai_agents_available = True
            success("AI Agents loaded successfully")
        except ImportError:
            self.run_ai_agents_analysis = None
            warning("AI Agents not available")

    def setup_environment(self):
        """Setup application environment with modern logging"""
        try:
            if self.core_available and self.config:
                # Use core config to validate paths
                self.config.validate_paths()
                success("Environment setup using core configuration")
            else:
                # Basic setup with modern logging
                basic_dirs = ["datacsv", "output_default", "models", "logs"]
                for directory in basic_dirs:
                    Path(directory).mkdir(parents=True, exist_ok=True)
                info(f"Created basic directories: {', '.join(basic_dirs)}")
        except Exception as e:
            warning(f"Warning during environment setup: {e}")

    def clear_screen(self):
        """Clear terminal screen"""
        os.system("cls" if os.name == "nt" else "clear")

    def print_logo(self):
        """Print enhanced application logo"""
        try:
            # Try to use enhanced logo system
            from utils.enhanced_logo import ProjectPLogo
            print(ProjectPLogo.get_modern_logo())
        except ImportError:
            # Fallback to menu interface logo if available
            if self.core_available and self.menu_interface:
                self.menu_interface.print_logo()
            else:
                # Basic fallback logo
                logo = f"""
{self.colorize('╔══════════════════════════════════════════════════════════════════════════════╗', self.Colors.BRIGHT_CYAN)}
{self.colorize('║', self.Colors.BRIGHT_CYAN)}                    {self.colorize('🚀 NICEGOLD ProjectP v2.0', self.Colors.BRIGHT_MAGENTA)}                        {self.colorize('║', self.Colors.BRIGHT_CYAN)}
{self.colorize('║', self.Colors.BRIGHT_CYAN)}                  {self.colorize('Professional AI Trading System', self.Colors.BRIGHT_GREEN)}                   {self.colorize('║', self.Colors.BRIGHT_CYAN)}
{self.colorize('╚══════════════════════════════════════════════════════════════════════════════╝', self.Colors.BRIGHT_CYAN)}
"""
                print(logo)

    def display_system_status(self):
        """Display system status"""
        if self.core_available and self.menu_interface:
            self.menu_interface.display_system_status()
        else:
            # Fallback status display
            print(f"\n{self.colorize('📊 SYSTEM STATUS', self.Colors.BRIGHT_CYAN)}")
            print(f"{self.colorize('═' * 50, self.Colors.BRIGHT_CYAN)}")
            print(f"🐍 Python: {sys.version.split()[0]}")
            print(f"💻 Platform: {os.name}")
            print(f"📁 Directory: {os.getcwd()}")
            print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(
                f"🔧 Core Modules: {'✅ Available' if self.core_available else '❌ Not Available'}"
            )
            print(
                f"📝 Advanced Logger: {'✅ Available' if self.advanced_logger_available else '❌ Not Available'}"
            )
            print(
                f"🤖 AI Agents: {'✅ Available' if self.ai_agents_available else '❌ Not Available'}"
            )

    def show_main_menu(self):
        """Display main menu"""
        if self.core_available and self.menu_interface:
            return self.menu_interface.print_main_menu()
        else:
            # Fallback menu
            print(
                f"\n{self.colorize('🎯 MAIN MENU (FALLBACK MODE)', self.Colors.BRIGHT_CYAN)}"
            )
            print(f"{self.colorize('═' * 50, self.Colors.BRIGHT_CYAN)}")
            print("1. 🚀 Full Pipeline")
            print("2. 📊 Data Analysis")
            print("3. 🔧 Quick Test")
            print("4.  System Health Check")
            print("5. 📦 Install Dependencies")
            print("6. 🧹 Clean System")
            print("0. 👋 Exit")
            print(f"{self.colorize('═' * 50, self.Colors.BRIGHT_CYAN)}")

            choice = safe_input(
                f"\n{self.colorize('👉 เลือกตัวเลือก (0-6):', self.Colors.BRIGHT_YELLOW)} ",
                default="0",
            ).strip()
            return choice if choice else None

    def handle_menu_choice(self, choice):
        """Handle menu choice"""
        if self.core_available and self.menu_interface:
            return self.menu_interface.handle_menu_choice(choice)
        else:
            # Fallback menu handling
            return self.handle_fallback_choice(choice)

    def handle_fallback_choice(self, choice):
        """Handle menu choice when core modules not available"""
        if choice == "1":
            print("🚀 Running Full Pipeline...")
            print("⚠️ Core modules not available. Please install dependencies.")
            return True
        elif choice == "2":
            print("📊 Running Data Analysis...")
            print("⚠️ Core modules not available. Please install dependencies.")
            return True
        elif choice == "3":
            print("🔧 Running Quick Test...")
            print("✅ Basic system test completed")
            print(f"   Python version: {sys.version}")
            print(f"   Current directory: {os.getcwd()}")
            print(
                f"   Available directories: {[d for d in os.listdir('.') if os.path.isdir(d)]}"
            )
            return True
        elif choice == "4":
            print(" System Health Check...")
            self.display_system_status()
            return True
        elif choice == "5":
            print("📦 Installing Dependencies...")
            self.install_dependencies()
            return True
        elif choice == "6":
            print("🧹 Cleaning System...")
            self.clean_system()
            return True
        elif choice == "0":
            print("👋 Thank you for using NICEGOLD ProjectP!")
            return False
        else:
            print("❌ Invalid choice. Please try again.")
            return True

    def install_dependencies(self):
        """Install required dependencies"""
        essential_packages = [
            "pandas",
            "numpy",
            "scikit-learn",
            "matplotlib",
            "seaborn",
            "joblib",
            "pyyaml",
            "tqdm",
            "requests",
            "streamlit",
        ]

        ml_packages = ["catboost", "lightgbm", "xgboost", "optuna", "shap", "ta"]

        all_packages = essential_packages + ml_packages

        print(f"📦 Installing {len(all_packages)} packages...")

        import subprocess

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install"] + all_packages,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                print("✅ Dependencies installed successfully!")
                print("🔄 Please restart the application to use new features.")
            else:
                print(f"❌ Installation failed: {result.stderr}")
        except Exception as e:
            print(f"❌ Error during installation: {e}")

    def clean_system(self):
        """Clean system files"""
        print("🧹 Cleaning system files...")

        clean_patterns = [
            "**/__pycache__",
            "**/*.pyc",
            "**/*.pyo",
            "**/.pytest_cache",
            "**/.mypy_cache",
        ]

        total_cleaned = 0
        for pattern in clean_patterns:
            try:
                for path in Path(".").rglob(pattern.replace("**/", "")):
                    if path.exists():
                        if path.is_file():
                            path.unlink()
                            total_cleaned += 1
                        elif path.is_dir():
                            import shutil

                            shutil.rmtree(path)
                            total_cleaned += 1
            except Exception as e:
                print(f"⚠️ Error cleaning {pattern}: {e}")

        print(f"✅ Cleanup completed! {total_cleaned} items removed.")

    def print_status_bar(self):
        """Print status bar"""
        current_time = datetime.now().strftime("%H:%M:%S")
        uptime = datetime.now() - self.start_time
        uptime_str = str(uptime).split(".")[0]

        print(f"\n{self.colorize('═' * 80, self.Colors.BRIGHT_BLUE)}")
        status_parts = [
            f"{self.colorize('⏰ เวลา:', self.Colors.BRIGHT_BLUE)} {self.colorize(current_time, self.Colors.WHITE)}",
            f"{self.colorize('🚀 NICEGOLD ProjectP', self.Colors.BRIGHT_MAGENTA)}",
            f"{self.colorize('⏱️ Uptime:', self.Colors.BRIGHT_YELLOW)} {self.colorize(uptime_str, self.Colors.WHITE)}",
            f"{self.colorize('✅ พร้อมใช้งาน', self.Colors.BRIGHT_GREEN)}",
        ]
        print(" | ".join(status_parts))
        print(f"{self.colorize('═' * 80, self.Colors.BRIGHT_BLUE)}")

    def display_session_summary(self):
        """Display session summary with modern logging"""
        try:
            if MODERN_LOGGER_AVAILABLE:
                # Use modern logger's display_summary
                if logger:
                    logger.display_summary()
            elif self.advanced_logger_available and self.logger:
                self.logger.print_summary()
            else:
                # Fallback summary
                uptime = datetime.now() - self.start_time
                print(
                    f"\n{self.colorize('📊 SESSION SUMMARY', self.Colors.BRIGHT_CYAN)}"
                )
                print(f"{self.colorize('═' * 50, self.Colors.BRIGHT_CYAN)}")
                print(f"⏱️ Session Duration: {str(uptime).split('.')[0]}")
                print("✅ Session completed successfully")
                print(f"{self.colorize('═' * 50, self.Colors.BRIGHT_CYAN)}")
        except Exception as e:
            error(f"Error displaying summary: {e}")

    def show_loading_animation(self, message, duration=1.5):
        """Show loading animation"""
        chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        end_time = time.time() + duration
        i = 0

        while time.time() < end_time:
            char = chars[i % len(chars)]
            print(
                f"\r{self.colorize(char, self.Colors.BRIGHT_CYAN)} {self.colorize(message, self.Colors.WHITE)}",
                end="",
                flush=True,
            )
            time.sleep(0.1)
            i += 1

        print(
            f"\r{self.colorize('✅', self.Colors.BRIGHT_GREEN)} {self.colorize(message, self.Colors.WHITE)} - เสร็จสิ้น!"
        )

    def run(self):
        """Main application loop with enhanced menu interface"""
        try:
            # Try to use enhanced menu interface
            try:
                from utils.enhanced_menu import EnhancedMenuInterface
                enhanced_menu = EnhancedMenuInterface()
                info("🚀 Starting NICEGOLD ProjectP with Enhanced Interface")
                enhanced_menu.run()
                return
            except ImportError:
                warning("Enhanced menu not available, using fallback")
            
            # Use the core menu interface if available
            if self.core_available and hasattr(self, "menu_interface"):
                info("Starting NICEGOLD ProjectP with core modules")
                self.menu_interface.run()
            else:
                # Fallback for basic operation
                warning("Running in basic mode - core modules not available")
                info("🚀 NICEGOLD ProjectP v2.0 - Basic Mode")
                warning("Core modules not available - running in basic mode")
                info("For full functionality, ensure all core modules are installed.")
                
                # Simple fallback loop
                while True:
                    choice = self.show_main_menu()
                    if choice is None:
                        continue
                    if not self.handle_menu_choice(choice):
                        break

        except KeyboardInterrupt:
            print(f"\n\n{self.colorize('👋 Application interrupted by user', self.Colors.BRIGHT_YELLOW)}")
            self.display_session_summary()
        except Exception as e:
            critical(f"Critical error in main application: {e}")
            error(f"Error details: {traceback.format_exc()}")
            critical("System will shutdown...")
            time.sleep(2)
            sys.exit(1)


def main():
    """Main entry point with modern logging"""
    try:
        info("🚀 Initializing NICEGOLD ProjectP v2.0")
        app = ProjectPApplication()
        app.run()
    except Exception as e:
        critical(f"Failed to start application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
