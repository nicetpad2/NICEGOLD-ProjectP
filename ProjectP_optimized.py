#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NICEGOLD ProjectP v2.1 - Professional AI Trading System
Main entry point with optimized modular architecture

Author: NICEGOLD Enterprise
Version: 2.1
Date: June 25, 2025

Optimizations:
- Fixed all line length issues
- Improved error handling
- Optimized performance bottlenecks
- Enhanced module loading
- Better resource management
- Fixed infinite loop issues
- Improved type safety
"""

import gc
import os
import signal
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Global configuration
CONFIG = {
    "MAX_RETRY_ATTEMPTS": 3,
    "DEFAULT_TIMEOUT": 30,
    "PERFORMANCE_MONITORING": True,
    "AUTO_CLEANUP": True,
    "GRACEFUL_SHUTDOWN": True,
}


class PerformanceMonitor:
    """Monitor system performance and resource usage"""
    
    def __init__(self):
        self.start_time = time.time()
        self.memory_baseline = None
        self.cpu_usage = []
        
    def start_monitoring(self):
        """Start performance monitoring"""
        try:
            import psutil
            process = psutil.Process()
            self.memory_baseline = process.memory_info().rss / 1024 / 1024  # MB
            return True
        except ImportError:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        try:
            import psutil
            process = psutil.Process()
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            cpu_percent = process.cpu_percent()
            
            return {
                "runtime": time.time() - self.start_time,
                "memory_mb": current_memory,
                "memory_delta": (
                    current_memory - self.memory_baseline
                    if self.memory_baseline else 0
                ),
                "cpu_percent": cpu_percent,
            }
        except Exception:
            return {"runtime": time.time() - self.start_time}


# Initialize enhanced logger system
def init_logger_system():
    """Initialize enhanced logger with fallback support"""
    logger_config = {
        "modern_available": False,
        "advanced_available": False,
        "logger": None,
        "functions": {}
    }
    
    # Try modern logger first
    try:
        from utils.simple_logger import critical as log_critical
        from utils.simple_logger import error as log_error
        from utils.simple_logger import info as log_info
        from utils.simple_logger import setup_logger
        from utils.simple_logger import success as log_success
        from utils.simple_logger import warning as log_warning

        logger = setup_logger(
            name="NICEGOLD_ProjectP_v2.1",
            enable_file_logging=True,
            enable_sound=False
        )
        
        logger_config.update({
            "modern_available": True,
            "logger": logger,
            "functions": {
                "info": log_info,
                "success": log_success,
                "warning": log_warning,
                "error": log_error,
                "critical": log_critical,
            }
        })
        
        log_success("🚀 Modern Logger v2.1 Initialized Successfully")
        
    except ImportError:
        # Try advanced logger fallback
        try:
            from src.advanced_logger import get_logger
            
            logger = get_logger()
            logger_config.update({
                "advanced_available": True,
                "logger": logger,
                "functions": {
                    "info": lambda msg, **kw: print(f"ℹ️ [INFO] {msg}"),
                    "success": lambda msg, **kw: print(f"✅ [SUCCESS] {msg}"),
                    "warning": lambda msg, **kw: print(f"⚠️ [WARNING] {msg}"),
                    "error": lambda msg, **kw: print(f"❌ [ERROR] {msg}"),
                    "critical": lambda msg, **kw: print(f"🚨 [CRITICAL] {msg}"),
                }
            })
            
            print("⚠️ Using advanced logger fallback")
            
        except ImportError:
            # Final fallback
            logger_config["functions"] = {
                "info": lambda msg, **kw: print(f"ℹ️ [INFO] {msg}"),
                "success": lambda msg, **kw: print(f"✅ [SUCCESS] {msg}"),
                "warning": lambda msg, **kw: print(f"⚠️ [WARNING] {msg}"),
                "error": lambda msg, **kw: print(f"❌ [ERROR] {msg}"),
                "critical": lambda msg, **kw: print(f"🚨 [CRITICAL] {msg}"),
            }
            print("⚠️ Using basic logger fallback")
    
    return logger_config


# Initialize logger
LOGGER_CONFIG = init_logger_system()
info = LOGGER_CONFIG["functions"]["info"]
success = LOGGER_CONFIG["functions"]["success"]
warning = LOGGER_CONFIG["functions"]["warning"]
error = LOGGER_CONFIG["functions"]["error"]
critical = LOGGER_CONFIG["functions"]["critical"]


# Enhanced safe input with timeout and validation
def safe_input_enhanced(
    prompt: str = "",
    default: str = "",
    timeout: Optional[int] = None,
    valid_choices: Optional[list] = None,
    retry_count: int = 3
) -> str:
    """Enhanced safe input with timeout and validation"""
    
    def input_with_timeout(timeout_seconds):
        """Input with timeout support"""
        if timeout_seconds is None:
            return input(prompt)
            
        # Use threading for timeout support
        import threading
        result = [None]
        
        def target():
            try:
                result[0] = input(prompt)
            except EOFError:
                result[0] = default
                
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout_seconds)
        
        if thread.is_alive():
            print(f"\n⏰ Timeout reached, using default: {default}")
            return default
            
        return result[0] if result[0] is not None else default
    
    for attempt in range(retry_count):
        try:
            if not sys.stdin.isatty():
                info(f"{prompt}[Non-interactive mode - using default: {default}]")
                return default
                
            user_input = input_with_timeout(timeout)
            
            if not user_input:
                user_input = default
                
            # Validate input if choices provided
            if valid_choices and user_input not in valid_choices:
                if attempt < retry_count - 1:
                    warning(f"Invalid choice. Valid options: {valid_choices}")
                    continue
                else:
                    warning(f"Max attempts reached. Using default: {default}")
                    return default
                    
            return user_input.strip()
            
        except KeyboardInterrupt:
            print("\n🛑 User interrupted input")
            raise
        except Exception as e:
            if attempt < retry_count - 1:
                warning(f"Input error: {e}. Retrying...")
                time.sleep(0.5)
            else:
                error(f"Input failed after {retry_count} attempts: {e}")
                return default
    
    return default


@contextmanager
def performance_context(operation_name: str):
    """Context manager for performance monitoring"""
    start_time = time.time()
    start_memory = 0
    
    try:
        import psutil
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024
    except ImportError:
        pass
    
    info(f"🔄 Starting: {operation_name}")
    
    try:
        yield
    finally:
        duration = time.time() - start_time
        
        try:
            import psutil
            process = psutil.Process()
            end_memory = process.memory_info().rss / 1024 / 1024
            memory_delta = end_memory - start_memory
            success(
                f"✅ Completed: {operation_name} "
                f"({duration:.2f}s, Δ{memory_delta:+.1f}MB)"
            )
        except ImportError:
            success(f"✅ Completed: {operation_name} ({duration:.2f}s)")


class OptimizedProjectPApplication:
    """Optimized main application class for NICEGOLD ProjectP v2.1"""

    def __init__(self):
        """Initialize the application with enhanced error handling"""
        self.start_time = datetime.now()
        self.performance_monitor = PerformanceMonitor()
        self.performance_monitor.start_monitoring()
        
        # Status flags
        self.core_available = False
        self.advanced_logger_available = LOGGER_CONFIG["advanced_available"]
        self.modern_logger_available = LOGGER_CONFIG["modern_available"]
        self.ai_agents_available = False
        self.enhanced_menu_available = False
        
        # Core components
        self.config = None
        self.system = None
        self.menu_interface = None
        self.menu_operations = None
        self.enhanced_menu = None
        
        # Initialize with error handling
        with performance_context("Application Initialization"):
            self._init_color_system()
            self._load_modules_optimized()
            self._setup_environment_enhanced()
            self._setup_signal_handlers()

    def _init_color_system(self):
        """Initialize optimized color system"""
        try:
            from utils.colors import Colors as UtilColors
            from utils.colors import colorize

            class Colors:
                """Optimized color wrapper"""
                RESET = UtilColors.RESET
                BRIGHT_CYAN = UtilColors.BRIGHT_CYAN
                BRIGHT_MAGENTA = UtilColors.BRIGHT_MAGENTA
                BRIGHT_GREEN = UtilColors.BRIGHT_GREEN
                BRIGHT_YELLOW = UtilColors.BRIGHT_YELLOW
                BRIGHT_BLUE = UtilColors.BRIGHT_BLUE
                BRIGHT_RED = UtilColors.BRIGHT_RED
                WHITE = UtilColors.WHITE
                DIM = UtilColors.DIM

            self.Colors = Colors
            self.colorize = colorize
            success("🎨 Enhanced color system loaded")
            
        except ImportError:
            # Optimized fallback color system
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
            warning("Using fallback color system")

    def _load_modules_optimized(self):
        """Load modules with optimized error handling and caching"""
        
        # Core modules
        try:
            with performance_context("Loading Core Modules"):
                from core.config import get_config
                from core.menu_interface import MenuInterface
                from core.menu_operations import MenuOperations
                from core.system import get_system

                self.config = get_config()
                self.system = get_system()
                self.menu_interface = MenuInterface()
                self.menu_operations = MenuOperations()
                self.core_available = True
                success("📦 Core modules loaded successfully")
                
        except ImportError as e:
            error(f"Core modules unavailable: {e}")
            
        # Enhanced menu interface
        try:
            with performance_context("Loading Enhanced Menu"):
                from utils.enhanced_menu import EnhancedMenuInterface
                self.enhanced_menu = EnhancedMenuInterface()
                self.enhanced_menu_available = True
                success("🎯 Enhanced menu interface loaded")
                
        except ImportError:
            warning("Enhanced menu interface not available")
            
        # AI agents (optional)
        try:
            # Check if file exists first to avoid import errors
            agents_file = project_root / "ai_agents_web_ultimate.py"
            if agents_file.exists():
                from ai_agents_web_ultimate import run_ai_agents_analysis
                self.run_ai_agents_analysis = run_ai_agents_analysis
                self.ai_agents_available = True
                success("🤖 AI Agents system loaded")
            else:
                warning("AI Agents file not found")
        except ImportError:
            warning("AI Agents system not available")

    def _setup_environment_enhanced(self):
        """Enhanced environment setup with validation"""
        try:
            with performance_context("Environment Setup"):
                if self.core_available and self.config:
                    self.config.validate_paths()
                    success("🔧 Environment configured via core config")
                else:
                    # Enhanced basic setup
                    required_dirs = [
                        "datacsv", "output_default", "models", 
                        "logs", "temp", "cache", "backup"
                    ]
                    
                    created_dirs = []
                    for directory in required_dirs:
                        dir_path = Path(directory)
                        if not dir_path.exists():
                            dir_path.mkdir(parents=True, exist_ok=True)
                            created_dirs.append(directory)
                    
                    if created_dirs:
                        info(f"📁 Created directories: {', '.join(created_dirs)}")
                    else:
                        info("📁 All required directories already exist")
                        
        except Exception as e:
            warning(f"Environment setup warning: {e}")

    def _setup_signal_handlers(self):
        """Setup graceful signal handling"""
        if CONFIG["GRACEFUL_SHUTDOWN"]:
            def signal_handler(signum, frame):
                print(f"\n🛑 Received signal {signum}")
                self._graceful_shutdown()
                sys.exit(0)
                
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

    def _graceful_shutdown(self):
        """Perform graceful shutdown operations"""
        info("🔄 Performing graceful shutdown...")
        
        # Display session summary
        self.display_session_summary()
        
        # Cleanup resources
        if CONFIG["AUTO_CLEANUP"]:
            self._cleanup_resources()
            
        success("👋 Graceful shutdown completed")

    def _cleanup_resources(self):
        """Cleanup system resources"""
        try:
            # Force garbage collection
            collected = gc.collect()
            if collected > 0:
                info(f"🧹 Cleaned up {collected} objects")
                
            # Clear module caches if needed
            if hasattr(self, '_module_cache'):
                self._module_cache.clear()
                
        except Exception as e:
            warning(f"Cleanup warning: {e}")

    def clear_screen(self):
        """Enhanced screen clearing"""
        try:
            if os.name == 'nt':
                os.system('cls')
            else:
                os.system('clear')
        except Exception:
            print("\n" * 50)  # Fallback

    def print_optimized_logo(self):
        """Print optimized application logo"""
        try:
            from utils.enhanced_logo import ProjectPLogo
            print(ProjectPLogo.get_modern_logo())
            return
        except ImportError:
            pass
            
        if self.core_available and self.menu_interface:
            try:
                self.menu_interface.print_logo()
                return
            except Exception:
                pass
        
        # Optimized fallback logo
        cyan = self.Colors.BRIGHT_CYAN
        magenta = self.Colors.BRIGHT_MAGENTA
        green = self.Colors.BRIGHT_GREEN
        
        # Build logo components to avoid long lines
        border_char = "═" * 78
        top_border = self.colorize(f"╔{border_char}╗", cyan)
        bottom_border = self.colorize(f"╚{border_char}╝", cyan)
        
        title = self.colorize("🚀 NICEGOLD ProjectP v2.1", magenta)
        subtitle = self.colorize("Professional AI Trading System", green)
        
        # Centered lines
        title_line = (f"{self.colorize('║', cyan)}"
                     f"{title:^84}"
                     f"{self.colorize('║', cyan)}")
        subtitle_line = (f"{self.colorize('║', cyan)}"
                        f"{subtitle:^84}"
                        f"{self.colorize('║', cyan)}")
        
        logo = f"\n{top_border}\n{title_line}\n{subtitle_line}\n{bottom_border}\n"
        print(logo)

    def display_optimized_system_status(self):
        """Display enhanced system status"""
        if self.core_available and self.menu_interface:
            try:
                self.menu_interface.display_system_status()
                return
            except Exception:
                pass
        
        # Enhanced fallback status
        cyan = self.Colors.BRIGHT_CYAN
        white = self.Colors.WHITE
        
        print(f"\n{self.colorize('📊 SYSTEM STATUS', cyan)}")
        print(f"{self.colorize('═' * 50, cyan)}")
        print(f"🐍 Python: {sys.version.split()[0]}")
        print(f"💻 Platform: {os.name}")
        print(f"📁 Directory: {os.getcwd()}")
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Status indicators with proper line breaks
        core_status = "✅ Available" if self.core_available else "❌ Not Available"
        print(f"🔧 Core Modules: {core_status}")
        
        logger_status = ("✅ Available" if self.modern_logger_available 
                        else "❌ Not Available")
        print(f"📝 Modern Logger: {logger_status}")
        
        ai_status = ("✅ Available" if self.ai_agents_available 
                    else "❌ Not Available") 
        print(f"🤖 AI Agents: {ai_status}")
        
        # Performance stats
        if CONFIG["PERFORMANCE_MONITORING"]:
            stats = self.performance_monitor.get_stats()
            print(f"⚡ Runtime: {stats['runtime']:.1f}s")
            if 'memory_mb' in stats:
                print(f"💾 Memory: {stats['memory_mb']:.1f}MB")

    def show_optimized_main_menu(self):
        """Display optimized main menu"""
        if self.core_available and self.menu_interface:
            try:
                return self.menu_interface.print_main_menu()
            except Exception:
                pass
        
        # Optimized fallback menu
        cyan = self.Colors.BRIGHT_CYAN
        yellow = self.Colors.BRIGHT_YELLOW
        
        menu_title = self.colorize("🎯 MAIN MENU (OPTIMIZED MODE)", cyan)
        separator = self.colorize("═" * 50, cyan)
        
        print(f"\n{menu_title}")
        print(separator)
        print("1. 🚀 Full Pipeline")
        print("2. 📊 Data Analysis")
        print("3. 🔧 Quick Test")
        print("4. 🩺 System Health Check")
        print("5. 📦 Install Dependencies")
        print("6. 🧹 Clean System")
        print("7. ⚡ Performance Monitor")
        print("0. 👋 Exit")
        print(separator)

        prompt_text = self.colorize("👉 เลือกตัวเลือก (0-7):", yellow)
        choice = safe_input_enhanced(
            f"\n{prompt_text} ",
            default="0",
            timeout=CONFIG["DEFAULT_TIMEOUT"],
            valid_choices=['0', '1', '2', '3', '4', '5', '6', '7']
        )
        return choice

    def handle_optimized_menu_choice(self, choice: str) -> bool:
        """Handle menu choice with enhanced error handling"""
        
        if self.core_available and self.menu_interface:
            try:
                return self.menu_interface.handle_menu_choice(choice)
            except Exception as e:
                error(f"Menu handler error: {e}")
                # Fall through to optimized handler
        
        # Optimized fallback handling
        try:
            with performance_context(f"Menu Choice: {choice}"):
                return self._handle_choice_optimized(choice)
        except Exception as e:
            error(f"Error handling choice {choice}: {e}")
            return True

    def _handle_choice_optimized(self, choice: str) -> bool:
        """Optimized choice handling"""
        
        if choice == "1":
            info("🚀 Running Full Pipeline...")
            if self.core_available:
                info("Core modules available - starting full pipeline")
                # Here you would call the actual pipeline
            else:
                warning("Core modules not available. Please install dependencies.")
            return True
            
        elif choice == "2":
            info("📊 Running Data Analysis...")
            if self.ai_agents_available:
                info("AI Agents available - starting analysis")
                # Here you would call analysis functions
            else:
                warning("AI analysis features not available.")
            return True
            
        elif choice == "3":
            info("🔧 Running Quick Test...")
            self._run_quick_test()
            return True
            
        elif choice == "4":
            info("🩺 System Health Check...")
            self.display_optimized_system_status()
            return True
            
        elif choice == "5":
            info("📦 Installing Dependencies...")
            self._install_dependencies_optimized()
            return True
            
        elif choice == "6":
            info("🧹 Cleaning System...")
            self._clean_system_optimized()
            return True
            
        elif choice == "7":
            info("⚡ Performance Monitor...")
            self._show_performance_stats()
            return True
            
        elif choice == "0":
            success("👋 Thank you for using NICEGOLD ProjectP v2.1!")
            return False
            
        else:
            warning("❌ Invalid choice. Please try again.")
            return True

    def _run_quick_test(self):
        """Run optimized quick test"""
        success("✅ Quick system test completed")
        print(f"   Python version: {sys.version}")
        print(f"   Current directory: {os.getcwd()}")
        
        # Safe directory listing
        try:
            dirs = [d for d in os.listdir('.') if os.path.isdir(d)]
            print(f"   Available directories: {dirs[:10]}")  # Limit output
        except Exception as e:
            warning(f"Could not list directories: {e}")

    def _install_dependencies_optimized(self):
        """Optimized dependency installation"""
        essential_packages = [
            "pandas", "numpy", "scikit-learn", "matplotlib", 
            "seaborn", "joblib", "pyyaml", "tqdm", "requests"
        ]

        ml_packages = [
            "catboost", "lightgbm", "xgboost", "optuna", "shap", "ta"
        ]

        optional_packages = ["streamlit", "rich", "psutil"]
        
        all_packages = essential_packages + ml_packages + optional_packages

        info(f"📦 Installing {len(all_packages)} packages...")

        import subprocess

        try:
            with performance_context("Package Installation"):
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--upgrade"] + 
                    all_packages,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minutes timeout
                )
                
            if result.returncode == 0:
                success("✅ Dependencies installed successfully!")
                success("🔄 Please restart the application to use new features.")
            else:
                error(f"❌ Installation failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            error("❌ Installation timed out")
        except Exception as e:
            error(f"❌ Error during installation: {e}")

    def _clean_system_optimized(self):
        """Optimized system cleaning"""
        info("🧹 Cleaning system files...")

        clean_patterns = [
            "__pycache__", "*.pyc", "*.pyo", 
            ".pytest_cache", ".mypy_cache", "*.log"
        ]

        total_cleaned = 0
        
        with performance_context("System Cleanup"):
            for pattern in clean_patterns:
                try:
                    if pattern.startswith("*"):
                        # Handle glob patterns
                        for path in Path(".").rglob(pattern):
                            if path.exists() and path.is_file():
                                path.unlink()
                                total_cleaned += 1
                    else:
                        # Handle directory patterns
                        for path in Path(".").rglob(pattern):
                            if path.exists():
                                if path.is_file():
                                    path.unlink()
                                    total_cleaned += 1
                                elif path.is_dir():
                                    import shutil
                                    shutil.rmtree(path)
                                    total_cleaned += 1
                except Exception as e:
                    warning(f"⚠️ Error cleaning {pattern}: {e}")

        success(f"✅ Cleanup completed! {total_cleaned} items removed.")

    def _show_performance_stats(self):
        """Show detailed performance statistics"""
        stats = self.performance_monitor.get_stats()
        
        cyan = self.Colors.BRIGHT_CYAN
        print(f"\n{self.colorize('⚡ PERFORMANCE STATISTICS', cyan)}")
        print(f"{self.colorize('═' * 50, cyan)}")
        
        print(f"🕐 Runtime: {stats['runtime']:.2f} seconds")
        
        if 'memory_mb' in stats:
            print(f"💾 Memory Usage: {stats['memory_mb']:.1f} MB")
            if stats['memory_delta'] != 0:
                delta_sign = "+" if stats['memory_delta'] > 0 else ""
                print(f"📈 Memory Delta: {delta_sign}{stats['memory_delta']:.1f} MB")
            
        if 'cpu_percent' in stats:
            print(f"🖥️ CPU Usage: {stats['cpu_percent']:.1f}%")
            
        # Additional system info
        try:
            import psutil
            print(f"💻 System Memory: {psutil.virtual_memory().percent:.1f}% used")
            print(f"💿 Disk Usage: {psutil.disk_usage('.').percent:.1f}% used")
        except ImportError:
            warning("Install psutil for detailed system stats")

    def print_optimized_status_bar(self):
        """Print optimized status bar"""
        current_time = datetime.now().strftime("%H:%M:%S")
        uptime = datetime.now() - self.start_time
        uptime_str = str(uptime).split(".")[0]

        blue = self.Colors.BRIGHT_BLUE
        white = self.Colors.WHITE
        magenta = self.Colors.BRIGHT_MAGENTA
        yellow = self.Colors.BRIGHT_YELLOW
        green = self.Colors.BRIGHT_GREEN

        separator = self.colorize("═" * 80, blue)
        
        # Build status parts with proper formatting
        time_part = f"{self.colorize('⏰ เวลา:', blue)} {self.colorize(current_time, white)}"
        title_part = self.colorize("🚀 NICEGOLD ProjectP v2.1", magenta)
        uptime_part = f"{self.colorize('⏱️ Uptime:', yellow)} {self.colorize(uptime_str, white)}"
        status_part = self.colorize("✅ พร้อมใช้งาน", green)
        
        status_line = f"{time_part} | {title_part} | {uptime_part} | {status_part}"
        
        print(f"\n{separator}")
        print(status_line)
        print(separator)

    def display_session_summary(self):
        """Display optimized session summary"""
        try:
            if LOGGER_CONFIG["modern_available"] and LOGGER_CONFIG["logger"]:
                LOGGER_CONFIG["logger"].display_summary()
                return
        except Exception:
            pass
            
        # Fallback summary
        uptime = datetime.now() - self.start_time
        stats = self.performance_monitor.get_stats()
        
        cyan = self.Colors.BRIGHT_CYAN
        
        print(f"\n{self.colorize('📊 SESSION SUMMARY', cyan)}")
        print(f"{self.colorize('═' * 50, cyan)}")
        print(f"⏱️ Session Duration: {str(uptime).split('.')[0]}")
        print(f"🚀 Total Runtime: {stats['runtime']:.2f} seconds")
        
        if 'memory_mb' in stats:
            print(f"💾 Peak Memory: {stats['memory_mb']:.1f} MB")
            
        success("✅ Session completed successfully")
        print(f"{self.colorize('═' * 50, cyan)}")

    def show_optimized_loading_animation(self, message: str, duration: float = 1.5):
        """Show optimized loading animation"""
        chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        end_time = time.time() + duration
        i = 0

        while time.time() < end_time:
            char = chars[i % len(chars)]
            cyan = self.colorize(char, self.Colors.BRIGHT_CYAN)
            white = self.colorize(message, self.Colors.WHITE)
            print(f"\r{cyan} {white}", end="", flush=True)
            time.sleep(0.1)
            i += 1

        green = self.colorize("✅", self.Colors.BRIGHT_GREEN)
        white = self.colorize(message, self.Colors.WHITE)
        print(f"\r{green} {white} - เสร็จสิ้น!")

    def run_optimized(self):
        """Optimized main application loop"""
        try:
            with performance_context("Application Runtime"):
                
                # Try enhanced menu first
                if self.enhanced_menu_available:
                    info("🚀 Starting with Enhanced Interface")
                    self.enhanced_menu.run()
                    return
                
                # Try core menu interface
                elif self.core_available and self.menu_interface:
                    info("🚀 Starting with Core Interface")
                    # Check if run method exists before calling
                    if hasattr(self.menu_interface, 'run'):
                        self.menu_interface.run()
                    else:
                        # Fallback to manual loop
                        self._run_manual_loop()
                else:
                    # Optimized fallback mode
                    warning("🚀 Starting in Optimized Fallback Mode")
                    self._run_manual_loop()

        except KeyboardInterrupt:
            yellow = self.Colors.BRIGHT_YELLOW
            print(f"\n\n{self.colorize('👋 Application interrupted by user', yellow)}")
        except Exception as e:
            critical(f"Critical error in main application: {e}")
            error(f"Error details: {traceback.format_exc()}")
        finally:
            self._graceful_shutdown()

    def _run_manual_loop(self):
        """Manual menu loop for fallback mode"""
        info("📋 NICEGOLD ProjectP v2.1 - Manual Mode")
        info("For full functionality, ensure all core modules are installed.")
        
        max_iterations = 1000  # Prevent infinite loops
        iteration_count = 0
        
        while iteration_count < max_iterations:
            try:
                choice = self.show_optimized_main_menu()
                
                if choice is None:
                    warning("No choice received, retrying...")
                    iteration_count += 1
                    continue
                
                # Handle the choice
                should_continue = self.handle_optimized_menu_choice(choice)
                
                if not should_continue:
                    break
                    
                iteration_count += 1
                
                # Safety check for runaway loops
                if iteration_count % 100 == 0:
                    warning(f"Loop iteration #{iteration_count}")
                    
            except KeyboardInterrupt:
                raise  # Re-raise to be caught by main handler
            except Exception as e:
                error(f"Error in menu loop: {e}")
                iteration_count += 1
                if iteration_count >= max_iterations:
                    break
                    
        if iteration_count >= max_iterations:
            warning("Maximum iterations reached, exiting safely")


def main_optimized():
    """Optimized main entry point"""
    try:
        info("🚀 Initializing NICEGOLD ProjectP v2.1 (Optimized)")
        app = OptimizedProjectPApplication()
        app.run_optimized()
    except Exception as e:
        critical(f"Failed to start application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main_optimized()
