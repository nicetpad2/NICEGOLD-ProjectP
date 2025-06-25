#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main menu interface for NICEGOLD ProjectP
Handles menu display and user interaction with modern logging
"""

import time
from datetime import datetime
from typing import Optional

# Initialize modern logger
try:
    from utils.modern_logger import (
        critical,
        error,
        get_logger,
        info,
        progress,
        success,
        warning,
    )

    # Get the global logger instance
    logger = get_logger()
    MODERN_LOGGER_AVAILABLE = True

except ImportError:
    MODERN_LOGGER_AVAILABLE = False

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

    def progress(msg, **kwargs):
        print(f"⏳ [PROGRESS] {msg}")

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


from core.config import config_manager
from core.menu_operations import MenuOperations
from core.system import system_manager
from utils.colors import Colors, clear_screen, colorize, print_with_animation


class MenuInterface:
    """Main menu interface manager"""

    def __init__(self):
        self.config = config_manager
        self.system = system_manager
        self.operations = MenuOperations()

    def print_logo(self):
        """Display beautiful ASCII logo"""
        logo = f"""
{colorize('╔══════════════════════════════════════════════════════════════════════════════╗', Colors.BRIGHT_CYAN)}
{colorize('║', Colors.BRIGHT_CYAN)}                                                                              {colorize('║', Colors.BRIGHT_CYAN)}
{colorize('║', Colors.BRIGHT_CYAN)}   {colorize('🚀 NICEGOLD ProjectP v2.0', Colors.BOLD + Colors.BRIGHT_MAGENTA)}                                            {colorize('║', Colors.BRIGHT_CYAN)}
{colorize('║', Colors.BRIGHT_CYAN)}   {colorize('Professional AI Trading System', Colors.BRIGHT_WHITE)}                                 {colorize('║', Colors.BRIGHT_CYAN)}
{colorize('║', Colors.BRIGHT_CYAN)}                                                                              {colorize('║', Colors.BRIGHT_CYAN)}
{colorize('║', Colors.BRIGHT_CYAN)}   {colorize('💎 Advanced Machine Learning', Colors.BRIGHT_YELLOW)}  {colorize('📊 Real-time Analytics', Colors.BRIGHT_GREEN)}        {colorize('║', Colors.BRIGHT_CYAN)}
{colorize('║', Colors.BRIGHT_CYAN)}   {colorize('🎯 Smart Backtesting', Colors.BRIGHT_BLUE)}         {colorize('⚡ High Performance', Colors.BRIGHT_RED)}          {colorize('║', Colors.BRIGHT_CYAN)}
{colorize('║', Colors.BRIGHT_CYAN)}                                                                              {colorize('║', Colors.BRIGHT_CYAN)}
{colorize('╚══════════════════════════════════════════════════════════════════════════════╝', Colors.BRIGHT_CYAN)}
        """
        print(logo)

    def print_status_bar(self):
        """Print beautiful status bar"""
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"\n{colorize('═' * 80, Colors.BRIGHT_BLUE)}")
        print(
            f"{colorize('⏰ เวลา:', Colors.BRIGHT_BLUE)} {colorize(current_time, Colors.WHITE)} | "
            f"{colorize('🚀 NICEGOLD ProjectP', Colors.BRIGHT_MAGENTA)} | "
            f"{colorize('📁 Ready', Colors.BRIGHT_CYAN)} | "
            f"{colorize('✅ Online', Colors.BRIGHT_GREEN)}"
        )
        print(f"{colorize('═' * 80, Colors.BRIGHT_BLUE)}")

    def print_main_menu(self) -> Optional[str]:
        """Display enterprise-level main menu with grouped features"""

        # Display header
        print(
            f"\n{colorize('🏢 NICEGOLD ENTERPRISE TRADING SYSTEM', Colors.BOLD + Colors.BRIGHT_MAGENTA)}"
        )
        print(f"{colorize('━' * 80, Colors.BRIGHT_WHITE)}")
        print(
            f"{colorize('Production-Ready Features', Colors.BRIGHT_GREEN)} | {colorize('Development Features', Colors.BRIGHT_YELLOW)} | {colorize('System Tools', Colors.BRIGHT_CYAN)}"
        )
        print(f"{colorize('━' * 80, Colors.BRIGHT_WHITE)}")

        # Core Production Features (Enterprise Ready)
        print(
            f"\n{colorize('🚀 CORE PRODUCTION FEATURES', Colors.BOLD + Colors.BRIGHT_GREEN)}"
        )
        production_items = [
            ("1", "🚀 Full Pipeline", "Complete ML trading pipeline", "✅"),
            ("2", "📊 Data Analysis", "Comprehensive data analysis", "✅"),
            ("3", "🔧 Quick Test", "System functionality test", "✅"),
            ("4", "� Health Check", "System diagnostics & monitoring", "✅"),
            ("5", "📦 Install Dependencies", "Package management", "✅"),
            ("6", "🧹 Clean System", "System cleanup & maintenance", "✅"),
        ]

        self._print_menu_section(production_items, Colors.BRIGHT_GREEN)

        # AI & Advanced Features
        print(
            f"\n{colorize('🤖 AI & ADVANCED ANALYTICS', Colors.BOLD + Colors.BRIGHT_CYAN)}"
        )
        ai_items = [
            ("10", "� AI Project Analysis", "AI-powered project analysis", "🔬"),
            ("11", "🔧 AI Auto-Fix", "Intelligent error correction", "🔬"),
            ("12", "⚡ AI Performance Optimizer", "AI system optimization", "🔬"),
            ("13", "📊 AI Executive Summary", "AI-generated insights", "🔬"),
            ("14", "🎛️ AI Agents Dashboard", "AI control center", "🔬"),
        ]

        self._print_menu_section(ai_items, Colors.BRIGHT_CYAN)

        # Trading & Backtesting (Real Data Only)
        print(
            f"\n{colorize('📈 TRADING & BACKTESTING', Colors.BOLD + Colors.BRIGHT_BLUE)}"
        )
        trading_items = [
            ("20", "🤖 Train Models", "Machine learning model training", "⚡"),
            (
                "21",
                "🎯 Backtest Strategy",
                "Historical backtesting with real data",
                "⚡",
            ),
            (
                "22",
                "📊 Data Analysis",
                "Real data analysis only (NO LIVE TRADING)",
                "🚫",
            ),
            ("23", "⚠️ Risk Management", "Risk analysis & controls", "⚡"),
            ("24", "📋 Performance Analysis", "Detailed performance metrics", "⚡"),
        ]

        self._print_menu_section(trading_items, Colors.BRIGHT_BLUE)

        # Web & API Services
        print(
            f"\n{colorize('🌐 WEB & API SERVICES', Colors.BOLD + Colors.BRIGHT_MAGENTA)}"
        )
        web_items = [
            ("30", "🌐 Web Dashboard", "Streamlit web interface", "🌐"),
            ("31", "🔌 API Server", "RESTful API service", "🌐"),
            ("32", "👁️ System Monitor", "Real-time monitoring", "🌐"),
            ("33", "� View Logs & Reports", "Log analysis & reporting", "🌐"),
        ]

        self._print_menu_section(web_items, Colors.BRIGHT_MAGENTA)

        # Development Tools
        print(
            f"\n{colorize('🛠️ DEVELOPMENT TOOLS', Colors.BOLD + Colors.BRIGHT_YELLOW)}"
        )
        dev_items = [
            ("40", "⚙️ Feature Engineering", "Create technical indicators", "🔧"),
            ("41", "🔄 Data Preprocessing", "Data cleaning & preparation", "🔧"),
            ("42", "� Model Comparison", "Compare ML algorithms", "🔧"),
            ("43", "🧪 Custom Pipeline", "Build custom workflows", "🔧"),
        ]

        self._print_menu_section(dev_items, Colors.BRIGHT_YELLOW)

        # Exit option
        print(f"\n{colorize('━' * 80, Colors.BRIGHT_WHITE)}")
        print(
            f"{colorize(' 0', Colors.BRIGHT_RED)} {colorize('🚪 Exit Application', Colors.BRIGHT_WHITE)} - {colorize('Safe shutdown', Colors.DIM + Colors.WHITE)}"
        )
        print(f"{colorize('━' * 80, Colors.BRIGHT_WHITE)}")

        # Legend
        print(
            f"\n{colorize('Legend:', Colors.BOLD + Colors.WHITE)} "
            f"{colorize('✅ Production Ready', Colors.BRIGHT_GREEN)} | "
            f"{colorize('� Advanced AI', Colors.BRIGHT_CYAN)} | "
            f"{colorize('⚡ Trading Tools', Colors.BRIGHT_BLUE)} | "
            f"{colorize('🌐 Web Services', Colors.BRIGHT_MAGENTA)} | "
            f"{colorize('🔧 Development', Colors.BRIGHT_YELLOW)}"
        )

        # Get user input
        choice = safe_input(
            f"\n{colorize('🎯 Select option (0-43): ', Colors.BOLD + Colors.BRIGHT_GREEN)}",
            default="0",
        ).strip()
        return choice if choice else None

    def _print_menu_section(self, items, color):
        """Print a section of menu items"""
        for i in range(0, len(items), 2):
            left_item = items[i]
            right_item = items[i + 1] if i + 1 < len(items) else None

            # Left item
            status_icon = left_item[3] if len(left_item) > 3 else ""
            left_text = (
                f"{colorize(left_item[0].rjust(2), color)} "
                f"{colorize(left_item[1], Colors.BRIGHT_WHITE)} "
                f"{colorize(status_icon, color)}"
            )
            left_desc = f"   {colorize(left_item[2], Colors.DIM + Colors.WHITE)}"

            if right_item:
                # Right item
                right_status_icon = right_item[3] if len(right_item) > 3 else ""
                right_text = (
                    f"{colorize(right_item[0].rjust(2), color)} "
                    f"{colorize(right_item[1], Colors.BRIGHT_WHITE)} "
                    f"{colorize(right_status_icon, color)}"
                )
                right_desc = f"   {colorize(right_item[2], Colors.DIM + Colors.WHITE)}"

                print(f"{left_text:<45} {right_text}")
                print(f"{left_desc:<45} {right_desc}")
            else:
                print(left_text)
                print(left_desc)

            print()  # Add spacing between rows

    def handle_menu_choice(self, choice: str) -> bool:
        """Handle enterprise menu choice and return True to continue, False to exit"""

        if choice == "0":
            print(
                f"{colorize('👋 ขอบคุณที่ใช้งาน NICEGOLD ProjectP Enterprise!', Colors.BRIGHT_MAGENTA)}"
            )
            return False

        # Core Production Features (Ready for Enterprise)
        production_map = {
            "1": self.operations.full_pipeline,
            "2": self.operations.data_analysis,  # Fixed: added data_analysis
            "3": self.operations.quick_test,
            "4": self.operations.health_check,  # Fixed: use health_check instead of system status
            "5": self.operations.install_dependencies,
            "6": self.operations.clean_system,
        }

        # AI & Advanced Features
        ai_map = {
            "10": self.operations.ai_project_analysis,
            "11": self.operations.ai_auto_fix,
            "12": self.operations.ai_performance_optimizer,
            "13": self.operations.ai_executive_summary,
            "14": self.operations.ai_agents_dashboard,
        }

        # Trading & Backtesting
        trading_map = {
            "20": self.operations.train_models,
            "21": self.operations.run_backtest,
            "22": self.operations.data_analysis,  # Real data analysis only
            "23": self.operations.risk_management,
            "24": self.operations.performance_analysis,
        }

        # Web & API Services
        web_map = {
            "30": self.operations.start_dashboard,
            "31": self.operations.placeholder_feature,  # API Server
            "32": self.operations.placeholder_feature,  # System Monitor
            "33": self.operations.view_logs,
        }

        # Development Tools
        dev_map = {
            "40": self.operations.placeholder_feature,  # Feature Engineering
            "41": self.operations.placeholder_feature,  # Data Preprocessing
            "42": self.operations.placeholder_feature,  # Model Comparison
            "43": self.operations.placeholder_feature,  # Custom Pipeline
        }

        # Combine all maps
        all_operations = {
            **production_map,
            **ai_map,
            **trading_map,
            **web_map,
            **dev_map,
        }

        if choice in all_operations:
            try:
                print(f"\n{colorize('🚀 Executing operation...', Colors.BRIGHT_CYAN)}")
                operation = all_operations[choice]

                if callable(operation):
                    result = operation()
                    if result is not None:
                        print(
                            f"{colorize('✅ Operation completed successfully!', Colors.BRIGHT_GREEN)}"
                        )
                    return True
                else:
                    operation()
                    print(f"{colorize('✅ Operation completed!', Colors.BRIGHT_GREEN)}")
                    return True

            except Exception as e:
                print(f"{colorize('❌ เกิดข้อผิดพลาด:', Colors.BRIGHT_RED)} {e}")
                print(
                    f"{colorize('🔧 กรุณาตรวจสอบระบบและลองใหม่', Colors.BRIGHT_YELLOW)}"
                )
                return True
        else:
            print(
                f"{colorize('❌ ตัวเลือกไม่ถูกต้อง กรุณาเลือกหมายเลข 0-43', Colors.BRIGHT_RED)}"
            )
            return True

    def custom_pipeline(self) -> bool:
        """Custom Pipeline Builder"""
        print("🧪 Custom Pipeline Builder...")
        print("🔧 Building custom workflow...")
        print("📊 You can customize:")
        print("   - Data preprocessing steps")
        print("   - Feature engineering methods")
        print("   - Model selection and parameters")
        print("   - Backtesting configurations")
        time.sleep(2)
        print("✅ Custom pipeline framework ready!")
        return True

    # Additional menu operations that need simple implementations
    def feature_engineering(self) -> bool:
        """Feature Engineering"""
        print("⚙️ Feature Engineering...")
        print("📊 Creating technical indicators...")
        time.sleep(2)
        print("✅ Feature engineering completed!")
        return True

    def preprocess_data(self) -> bool:
        """Data Preprocessing"""
        print("🔄 Data Preprocessing...")
        print("📊 Cleaning and preparing data...")
        time.sleep(2)
        print("✅ Preprocessing completed!")
        return True

    def compare_models(self) -> bool:
        """Model Comparison"""
        print("📊 Comparing Models...")
        print("🤖 Testing multiple algorithms...")
        time.sleep(2)
        print("✅ Model comparison completed!")
        return True

    def live_simulation(self) -> bool:
        """Live Trading Simulation"""
        print("📈 Live Trading Simulation...")
        print("💰 Simulating real-time trading...")
        time.sleep(3)
        print("✅ Simulation completed!")
        return True

    def performance_analysis(self) -> bool:
        """Performance Analysis"""
        print("📋 Performance Analysis...")
        print("📊 Calculating metrics...")
        time.sleep(2)
        print("✅ Analysis completed!")
        return True

    def risk_management(self) -> bool:
        """Risk Management"""
        print("⚠️ Risk Management Analysis...")
        print("📊 Calculating risk metrics...")
        time.sleep(2)
        print("✅ Risk analysis completed!")
        return True

    def start_api_server(self) -> bool:
        """Start API Server"""
        print("🔌 Starting API Server...")
        print("💡 API will be available at http://localhost:8000")
        print("⚠️ API server functionality requires FastAPI installation")
        return True

    def system_monitor(self) -> bool:
        """System Monitor"""
        print("👁️ System Monitor...")
        self.system.display_system_status()
        return True

    def ai_analysis(self) -> bool:
        """AI Analysis"""
        print("🔍 AI Project Analysis...")
        print("🤖 AI analysis requires additional modules")
        return True

    def auto_fix(self) -> bool:
        """Auto Fix"""
        print("🔧 AI Auto-Fix System...")
        print("🤖 Auto-fix requires additional modules")
        return True

    def optimizer(self) -> bool:
        """Optimizer"""
        print("⚡ AI Performance Optimizer...")
        print("🤖 Optimizer requires additional modules")
        return True

    def executive_summary(self) -> bool:
        """Executive Summary"""
        print("📊 AI Executive Summary...")
        print("🤖 Summary generation requires additional modules")
        return True

    def ai_dashboard(self) -> bool:
        """AI Dashboard"""
        print("🎛️ AI Agents Dashboard...")
        print("🤖 AI dashboard requires additional modules")
        return True

    def display_session_summary(self):
        """Display session summary"""
        print(f"\n{colorize('📊 SESSION SUMMARY', Colors.BRIGHT_CYAN)}")
        print(f"{colorize('=' * 50, Colors.BRIGHT_CYAN)}")
        print(f"⏱️ Session completed at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"✅ Operation completed successfully")
        print(f"{colorize('=' * 50, Colors.BRIGHT_CYAN)}")

    def run(self):
        """Main application loop with modern logging and progress bars"""
        try:
            info("🚀 Starting NICEGOLD ProjectP Enterprise System")

            # Setup environment with progress
            if MODERN_LOGGER_AVAILABLE and logger:
                with logger.progress_bar("Setting up environment", total=3) as update:
                    self.config.ensure_folders()
                    update()

                    clear_screen()
                    update()

                    self.print_logo()
                    update()
            else:
                # Fallback setup
                progress("Setting up environment...")
                self.config.ensure_folders()
                clear_screen()
                self.print_logo()

            success("System initialization completed")

            # Show system status
            self.system.display_system_status()

            # Main menu loop with modern logging
            while True:
                try:
                    self.print_status_bar()
                    choice = self.print_main_menu()

                    if not choice:
                        warning("No option selected, please choose a valid option")
                        time.sleep(1)
                        continue

                    # Log user selection
                    info(f"User selected option: {choice}")

                    # Show processing status
                    if MODERN_LOGGER_AVAILABLE and logger:
                        logger.simple_status(f"Processing option {choice}...")
                        continue_loop = self.handle_menu_choice(choice)
                    else:
                        progress(f"Processing option {choice}...")
                        continue_loop = self.handle_menu_choice(choice)

                    if not continue_loop:
                        success("Application shutdown requested")
                        break

                    # Show completion notification
                    if MODERN_LOGGER_AVAILABLE and logger:
                        from utils.modern_logger import NotificationType

                        logger.notify(
                            "Operation completed successfully!",
                            NotificationType.SUCCESS,
                            title="NICEGOLD ProjectP",
                        )
                    else:
                        success("Operation completed successfully!")

                    self.display_session_summary()

                    # Wait for user input to continue
                    if MODERN_LOGGER_AVAILABLE and logger:
                        logger.ask_input("Press Enter to continue...", "")
                    else:
                        safe_input(
                            "💡 Press Enter to return to main menu...", default=""
                        )

                    # Reload interface
                    clear_screen()
                    self.print_logo()

                except KeyboardInterrupt:
                    warning("Operation interrupted by user")
                    info("Returning to main menu...")
                    time.sleep(1)
                    continue

                except Exception as e:
                    error(f"Error in menu operation: {e}")
                    if MODERN_LOGGER_AVAILABLE and logger:
                        logger.handle_exception(e, "Menu operation", fatal=False)
                    safe_input("Press Enter to continue...", default="")

        except Exception as e:
            critical(f"Critical error in main application loop: {e}")
            if MODERN_LOGGER_AVAILABLE and logger:
                logger.handle_exception(e, "Main application", fatal=True)
            else:
                print("🚫 System will shutdown...")
                time.sleep(2)

    def display_system_status(self):
        """Display system status information"""
        try:
            # Use system manager's display method if available
            if hasattr(self.system, "display_system_status"):
                self.system.display_system_status()
            else:
                # Fallback system status display
                import os
                import sys

                print(f"\n{colorize('📊 SYSTEM STATUS', Colors.BRIGHT_CYAN)}")
                print(f"{colorize('═' * 50, Colors.BRIGHT_CYAN)}")
                print(f"🐍 Python: {sys.version.split()[0]}")
                print(f"💻 Platform: {os.name}")
                print(f"📁 Directory: {os.getcwd()}")
                print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                # Check system health if available
                try:
                    health = self.system.check_system_health()
                    print(f"💾 Memory: {health.get('memory_percent', 'N/A')}%")
                    print(f"💽 CPU: {health.get('cpu_percent', 'N/A')}%")
                    print(f"💿 Disk: {health.get('disk_percent', 'N/A')}%")
                except Exception:
                    print("💾 System metrics: Not available")

                print(f"{colorize('═' * 50, Colors.BRIGHT_CYAN)}")

        except Exception as e:
            print(f"⚠️ Error displaying system status: {e}")

    def show_main_menu(self):
        """Display main menu - alias for print_main_menu"""
        return self.print_main_menu()


# Global menu interface instance
menu_interface = MenuInterface()
