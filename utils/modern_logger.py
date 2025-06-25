#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 NICEGOLD Modern Logger - Advanced Terminal Logging System
═══════════════════════════════════════════════════════════════════════════════

Modern, beautiful terminal logging with progress bars, notifications, and error handling.

Features:
- 🎨 Beautiful colored output with rich formatting
- 📊 Progress bars with customizable styles
- 🚨 Advanced error handling and notifications
- 💫 Animated status indicators
- 📋 Structured logging with context
- 🔔 Sound notifications (optional)
- 📈 Performance metrics tracking
- 🎯 Interactive user feedback

Author: NICEGOLD Enterprise
Version: 3.0
Date: June 25, 2025
"""

import hashlib
import json
import logging
import os
import platform
import signal
import subprocess
import sys
import threading
import time
import traceback
from contextlib import contextmanager

# Standard library imports
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

# Try to import rich for beautiful terminal output
try:
    from rich import box
    from rich.align import Align
    from rich.columns import Columns
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.logging import RichHandler
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.prompt import Confirm, Prompt
    from rich.status import Status
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text
    from rich.traceback import install as install_rich_traceback
    from rich.tree import Tree
    RICH_AVAILABLE = True
    
    # Install rich traceback handler
    install_rich_traceback(show_locals=True)
    
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️ Rich library not available. Using fallback terminal output.")


class LogLevel(Enum):
    """Modern log levels with enhanced styling"""
    TRACE = ("TRACE", "dim", 0, "🔍")
    DEBUG = ("DEBUG", "blue", 1, "🐛")
    INFO = ("INFO", "white", 2, "ℹ️")
    SUCCESS = ("SUCCESS", "bold green", 3, "✅")
    WARNING = ("WARNING", "yellow", 4, "⚠️")
    ERROR = ("ERROR", "bold red", 5, "❌")
    CRITICAL = ("CRITICAL", "bold white on red", 6, "🚨")
    PROGRESS = ("PROGRESS", "cyan", 2, "⏳")
    
    def __init__(self, label: str, color: str, priority: int, icon: str):
        self.label = label
        self.color = color
        self.priority = priority
        self.icon = icon


class NotificationType(Enum):
    """Notification types for user feedback"""
    INFO = ("info", "blue", "ℹ️")
    SUCCESS = ("success", "green", "✅")
    WARNING = ("warning", "yellow", "⚠️")
    ERROR = ("error", "red", "❌")
    QUESTION = ("question", "cyan", "❓")
    
    def __init__(self, type_name: str, color: str, icon: str):
        self.type_name = type_name
        self.color = color
        self.icon = icon


@dataclass
class LogEntry:
    """Enhanced log entry with metadata"""
    timestamp: datetime
    level: LogLevel
    message: str
    module: str = "MAIN"
    function: str = ""
    line_number: int = 0
    context: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[Exception] = None
    duration: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    session_id: str = ""


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics"""
    start_time: datetime = field(default_factory=datetime.now)
    operations: int = 0
    errors: int = 0
    warnings: int = 0
    total_duration: float = 0.0
    peak_memory: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'start_time': self.start_time.isoformat(),
            'operations': self.operations,
            'errors': self.errors,
            'warnings': self.warnings,
            'total_duration': self.total_duration,
            'peak_memory': self.peak_memory,
            'uptime': (datetime.now() - self.start_time).total_seconds()
        }


class ModernLogger:
    """
    Modern terminal logger with rich formatting and progress tracking
    """
    
    def __init__(self, 
                 name: str = "NICEGOLD",
                 level: LogLevel = LogLevel.INFO,
                 enable_file_logging: bool = True,
                 enable_sound: bool = False):
        
        self.name = name
        self.level = level
        self.enable_file_logging = enable_file_logging
        self.enable_sound = enable_sound
        
        # Initialize Rich console
        if RICH_AVAILABLE:
            self.console = Console(
                force_terminal=True,
                width=120,
                legacy_windows=False
            )
        else:
            self.console = None
        
        # Session tracking
        self.session_id = self._generate_session_id()
        self.start_time = datetime.now()
        
        # Performance metrics
        self.metrics = PerformanceMetrics()
        
        # Log storage
        self.log_entries: List[LogEntry] = []
        self.max_entries = 1000
        
        # Progress tracking
        self.active_progress: Optional[Progress] = None
        self.progress_tasks: Dict[str, Any] = {}
        
        # Live display management to prevent Rich conflicts
        self._active_live_display = None
        self._live_display_stack = []
        self._display_lock = threading.Lock()
        
        # Setup file logging
        if self.enable_file_logging:
            self._setup_file_logging()
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Animation characters for loading
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.progress_chars = "▁▂▃▄▅▆▇█"
        
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_part = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"{timestamp}_{random_part}"
    
    def _setup_file_logging(self):
        """Setup file logging with rotation"""
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        log_file = logs_dir / f"nicegold_{self.session_id}.log"
        
        # Setup standard logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                RichHandler(console=self.console) if RICH_AVAILABLE else logging.StreamHandler()
            ]
        )
        self.file_logger = logging.getLogger(self.name)
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.info("Received shutdown signal, cleaning up...")
            self.shutdown()
            sys.exit(0)
        
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except AttributeError:
            # Windows doesn't support all signals
            pass
    
    def _get_caller_info(self) -> tuple:
        """Get caller function information"""
        frame = sys._getframe(2)
        return frame.f_code.co_name, frame.f_lineno
    
    def _format_message(self, level: LogLevel, message: str, **kwargs) -> str:
        """Format message with rich styling"""
        if not RICH_AVAILABLE:
            return f"{level.icon} [{level.label}] {message}"
        
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Format with rich markup
        formatted = f"[dim]{timestamp}[/dim] {level.icon} [{level.color}]{message}[/{level.color}]"
        
        # Add context if provided
        if kwargs:
            context_str = " ".join([f"{k}={v}" for k, v in kwargs.items()])
            formatted += f" [dim]({context_str})[/dim]"
        
        return formatted
    
    def _log(self, level: LogLevel, message: str, **kwargs):
        """Internal logging method"""
        if level.priority < self.level.priority:
            return
        
        # Get caller info
        function, line_number = self._get_caller_info()
        
        # Create log entry
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            function=function,
            line_number=line_number,
            context=kwargs,
            session_id=self.session_id
        )
        
        # Store entry
        self.log_entries.append(entry)
        if len(self.log_entries) > self.max_entries:
            self.log_entries.pop(0)
        
        # Update metrics
        self.metrics.operations += 1
        if level == LogLevel.ERROR:
            self.metrics.errors += 1
        elif level == LogLevel.WARNING:
            self.metrics.warnings += 1
        
        # Display message
        formatted_message = self._format_message(level, message, **kwargs)
        
        if RICH_AVAILABLE and self.console:
            self.console.print(formatted_message)
        else:
            print(formatted_message)
        
        # File logging
        if self.enable_file_logging and hasattr(self, 'file_logger'):
            log_level_map = {
                LogLevel.TRACE: logging.DEBUG,
                LogLevel.DEBUG: logging.DEBUG,
                LogLevel.INFO: logging.INFO,
                LogLevel.SUCCESS: logging.INFO,
                LogLevel.WARNING: logging.WARNING,
                LogLevel.ERROR: logging.ERROR,
                LogLevel.CRITICAL: logging.CRITICAL,
                LogLevel.PROGRESS: logging.INFO,
            }
            self.file_logger.log(log_level_map.get(level, logging.INFO), message)
        
        # Sound notification for errors
        if self.enable_sound and level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            self._play_notification_sound()
    
    def _play_notification_sound(self):
        """Play notification sound (platform-specific)"""
        try:
            if platform.system() == "Windows":
                import winsound
                winsound.Beep(1000, 500)
            elif platform.system() == "Darwin":  # macOS
                os.system("afplay /System/Library/Sounds/Glass.aiff")
            else:  # Linux
                os.system("paplay /usr/share/sounds/alsa/Front_Left.wav 2>/dev/null")
        except:
            pass  # Ignore if sound system not available
    
    # Public logging methods
    def trace(self, message: str, **kwargs):
        """Log trace message"""
        self._log(LogLevel.TRACE, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log(LogLevel.INFO, message, **kwargs)
    
    def success(self, message: str, **kwargs):
        """Log success message"""
        self._log(LogLevel.SUCCESS, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error message with optional exception"""
        if exception:
            kwargs['exception'] = str(exception)
            kwargs['traceback'] = traceback.format_exc()
        self._log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log critical message with optional exception"""
        if exception:
            kwargs['exception'] = str(exception)
            kwargs['traceback'] = traceback.format_exc()
        self._log(LogLevel.CRITICAL, message, **kwargs)
    
    def progress(self, message: str, **kwargs):
        """Log progress message"""
        self._log(LogLevel.PROGRESS, message, **kwargs)
    
    # Progress bar methods
    @contextmanager
    def progress_bar(self, 
                    description: str = "Processing...",
                    total: Optional[int] = None,
                    transient: bool = False):
        """Context manager for progress bar with display conflict prevention"""
        if not RICH_AVAILABLE:
            print(f"🔄 {description}")
            yield lambda: None
            print("✅ Complete!")
            return
        
        # Check for active live displays
        with self._display_lock:
            if self._active_live_display is not None:
                # Fallback to simple progress logging if display is active
                self.progress(f"Started: {description}")
                yield lambda advance=1: self.progress(f"Progress: {description}")
                self.success(f"Completed: {description}")
                return
            
            # Mark this display as active
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
                transient=transient
            )
            self._active_live_display = progress
        
        try:
            with progress:
                task = progress.add_task(description, total=total)
                
                def update(advance: int = 1):
                    progress.update(task, advance=advance)
                
                yield update
                
        except Exception as e:
            self.error(f"Progress bar error: {e}", exception=e)
        finally:
            with self._display_lock:
                self._active_live_display = None
    
    @contextmanager
    def status(self, message: str, spinner: str = "dots"):
        """Context manager for status spinner with display conflict prevention"""
        if not RICH_AVAILABLE:
            print(f"🔄 {message}")
            yield
            return
        
        # Check for active live displays
        with self._display_lock:
            if self._active_live_display is not None:
                # Fallback to simple status logging if display is active
                self.info(f"🔄 {message}")
                yield
                return
            
            # Create status display
            status_display = Status(message, spinner=spinner, console=self.console)
            self._active_live_display = status_display
        
        try:
            with status_display:
                yield
        except Exception as e:
            self.error(f"Status display error: {e}", exception=e)
        finally:
            with self._display_lock:
                self._active_live_display = None
    
    # Notification methods
    def notify(self, 
              message: str, 
              notification_type: NotificationType = NotificationType.INFO,
              title: str = "",
              duration: float = 3.0):
        """Display notification"""
        if not RICH_AVAILABLE:
            print(f"{notification_type.icon} {title}: {message}")
            return
        
        panel_title = (title or
                       f"{notification_type.icon} "
                       f"{notification_type.type_name.title()}")
        
        panel = Panel(
            message,
            title=panel_title,
            border_style=notification_type.color,
            box=box.ROUNDED
        )
        
        self.console.print(panel)
        
        # Auto-hide notification after duration
        if duration > 0:
            def hide_notification():
                time.sleep(duration)
                # In a real implementation, we would remove the panel
                # For now, just log that it would be hidden
                pass
            
            threading.Thread(target=hide_notification, daemon=True).start()
    
    def ask_confirmation(self, message: str, default: bool = False) -> bool:
        """Ask user for confirmation"""
        if not RICH_AVAILABLE:
            response = input(f"❓ {message} [y/N]: ").strip().lower()
            return response in ['y', 'yes']
        
        return Confirm.ask(f"❓ {message}", default=default, console=self.console)
    
    def ask_input(self, message: str, default: str = "") -> str:
        """Ask user for input"""
        if not RICH_AVAILABLE:
            response = input(f"❓ {message}: ").strip()
            return response or default
        
        return Prompt.ask(f"❓ {message}", default=default, console=self.console)
    
    # Error handling methods
    def handle_exception(self, 
                        exception: Exception, 
                        context: str = "", 
                        fatal: bool = False):
        """Handle exception with rich formatting"""
        error_msg = f"Exception in {context}: {str(exception)}" if context else str(exception)
        
        if RICH_AVAILABLE:
            # Display rich traceback
            self.console.print_exception(show_locals=True)
        else:
            # Fallback traceback
            print(f"❌ {error_msg}")
            traceback.print_exc()
        
        # Log the error
        self.error(error_msg, exception=exception)
        
        if fatal:
            self.critical("Fatal error encountered, shutting down...")
            self.shutdown()
            sys.exit(1)
    
    # Display methods
    def display_table(self, 
                     data: List[Dict[str, Any]], 
                     title: str = "",
                     columns: Optional[List[str]] = None):
        """Display data in a formatted table"""
        if not data:
            self.warning("No data to display")
            return
        
        if not RICH_AVAILABLE:
            # Fallback table display
            print(f"\n📊 {title}")
            print("-" * 50)
            for row in data:
                for key, value in row.items():
                    print(f"{key}: {value}")
                print("-" * 20)
            return
        
        table = Table(title=title, box=box.ROUNDED)
        
        # Add columns
        if columns:
            for col in columns:
                table.add_column(col, style="cyan")
        else:
            for key in data[0].keys():
                table.add_column(key, style="cyan")
        
        # Add rows
        for row in data:
            if columns:
                table.add_row(*[str(row.get(col, "")) for col in columns])
            else:
                table.add_row(*[str(value) for value in row.values()])
        
        self.console.print(table)
    
    def display_tree(self, data: Dict[str, Any], title: str = "Data Structure"):
        """Display hierarchical data as a tree"""
        if not RICH_AVAILABLE:
            print(f"\n🌳 {title}")
            self._print_dict_recursive(data)
            return
        
        tree = Tree(title)
        self._add_tree_nodes(tree, data)
        self.console.print(tree)
    
    def _print_dict_recursive(self, data: Dict[str, Any], indent: int = 0):
        """Fallback recursive dict printing"""
        for key, value in data.items():
            print("  " * indent + f"├─ {key}: ", end="")
            if isinstance(value, dict):
                print()
                self._print_dict_recursive(value, indent + 1)
            else:
                print(value)
    
    def _add_tree_nodes(self, parent, data):
        """Add nodes to rich tree"""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    branch = parent.add(f"📁 {key}")
                    self._add_tree_nodes(branch, value)
                else:
                    parent.add(f"📄 {key}: {value}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    branch = parent.add(f"📁 [{i}]")
                    self._add_tree_nodes(branch, item)
                else:
                    parent.add(f"📄 [{i}]: {item}")
    
    # Performance tracking
    def start_timer(self, name: str):
        """Start a named timer"""
        if not hasattr(self, 'timers'):
            self.timers = {}
        self.timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End a named timer and return duration"""
        if not hasattr(self, 'timers') or name not in self.timers:
            self.warning(f"Timer '{name}' not found")
            return 0.0
        
        duration = time.time() - self.timers[name]
        del self.timers[name]
        
        self.success(f"Timer '{name}' completed", duration=f"{duration:.2f}s")
        return duration
    
    @contextmanager
    def timer(self, name: str):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.success(f"Operation '{name}' completed", duration=f"{duration:.2f}s")
    
    # System information
    def display_system_info(self):
        """Display system information"""
        import psutil
        
        system_info = {
            "Python Version": sys.version.split()[0],
            "Platform": platform.system(),
            "CPU Usage": f"{psutil.cpu_percent()}%",
            "Memory Usage": f"{psutil.virtual_memory().percent}%",
            "Session ID": self.session_id,
            "Uptime": str(datetime.now() - self.start_time).split('.')[0],
            "Log Entries": len(self.log_entries),
            "Operations": self.metrics.operations,
            "Errors": self.metrics.errors,
            "Warnings": self.metrics.warnings
        }
        
        self.display_table([system_info], title="🖥️ System Information")
    
    # Summary and cleanup
    def display_summary(self):
        """Display session summary"""
        duration = datetime.now() - self.start_time
        
        summary_data = [
            {
                "Metric": "Session Duration",
                "Value": str(duration).split('.')[0]
            },
            {
                "Metric": "Total Operations",
                "Value": self.metrics.operations
            },
            {
                "Metric": "Errors",
                "Value": self.metrics.errors
            },
            {
                "Metric": "Warnings", 
                "Value": self.metrics.warnings
            },
            {
                "Metric": "Success Rate",
                "Value": f"{((self.metrics.operations - self.metrics.errors) / max(self.metrics.operations, 1) * 100):.1f}%"
            }
        ]
        
        self.display_table(summary_data, title="📊 Session Summary")
    
    def show_session_summary(self):
        """Alias for display_summary - Display session summary"""
        self.display_summary()

    def export_logs(self, filename: Optional[str] = None) -> str:
        """Export logs to JSON file"""
        if not filename:
            filename = f"logs/nicegold_session_{self.session_id}.json"
        
        export_data = {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "metrics": self.metrics.to_dict(),
            "entries": [
                {
                    "timestamp": entry.timestamp.isoformat(),
                    "level": entry.level.label,
                    "message": entry.message,
                    "module": entry.module,
                    "function": entry.function,
                    "context": entry.context
                }
                for entry in self.log_entries
            ]
        }
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.success(f"Logs exported to {filename}")
        return filename
    
    def shutdown(self):
        """Graceful shutdown"""
        self.info("Shutting down logger...")
        self.display_summary()
        
        # Export logs
        try:
            self.export_logs()
        except Exception as e:
            self.error("Failed to export logs", exception=e)
        
        self.success("Logger shutdown complete")
    
    def show_progress(self, message: str, 
                     step: Optional[int] = None, 
                     total: Optional[int] = None):
        """Show progress without live displays (safe for concurrent use)"""
        if step is not None and total is not None:
            percentage = (step / total) * 100
            self.progress(f"{message} ({step}/{total} - {percentage:.1f}%)")
        else:
            self.progress(message)
    
    def simple_status(self, message: str):
        """Show status without live displays (safe for concurrent use)"""
        self.info(f"🔄 {message}")


# Global logger instance
_global_logger: Optional[ModernLogger] = None


def get_logger(name: str = "NICEGOLD", **kwargs) -> ModernLogger:
    """Get or create global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = ModernLogger(name=name, **kwargs)
    return _global_logger


def setup_logger(name: str = "NICEGOLD", 
                level: LogLevel = LogLevel.INFO,
                enable_file_logging: bool = True,
                enable_sound: bool = False) -> ModernLogger:
    """Setup and configure the global logger"""
    global _global_logger
    _global_logger = ModernLogger(
        name=name,
        level=level,
        enable_file_logging=enable_file_logging,
        enable_sound=enable_sound
    )
    return _global_logger


# Convenience functions
def info(message: str, **kwargs):
    """Log info message using global logger"""
    get_logger().info(message, **kwargs)


def success(message: str, **kwargs):
    """Log success message using global logger"""
    get_logger().success(message, **kwargs)


def warning(message: str, **kwargs):
    """Log warning message using global logger"""
    get_logger().warning(message, **kwargs)


def error(message: str, exception: Optional[Exception] = None, **kwargs):
    """Log error message using global logger"""
    get_logger().error(message, exception=exception, **kwargs)


def critical(message: str, exception: Optional[Exception] = None, **kwargs):
    """Log critical message using global logger"""
    get_logger().critical(message, exception=exception, **kwargs)


def debug(message: str, **kwargs):
    """Log debug message using global logger"""
    get_logger().debug(message, **kwargs)


def progress(message: str, **kwargs):
    """Log progress message using global logger"""
    get_logger().progress(message, **kwargs)


# Example usage and test function
def test_logger():
    """Test the modern logger functionality"""
    logger = setup_logger("TEST", level=LogLevel.DEBUG)
    
    logger.info("Testing modern logger functionality")
    
    # Test different log levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.success("Success message")
    logger.warning("Warning message") 
    logger.error("Error message")
    
    # Test progress bar
    with logger.progress_bar("Testing progress", total=100) as update:
        for i in range(100):
            time.sleep(0.01)
            update(1)
    
    # Test status
    with logger.status("Processing data..."):
        time.sleep(2)
    
    # Test notifications
    logger.notify("Process completed successfully!", NotificationType.SUCCESS)
    
    # Test table display
    data = [
        {"Name": "Test 1", "Status": "Pass", "Duration": "1.2s"},
        {"Name": "Test 2", "Status": "Pass", "Duration": "0.8s"},
        {"Name": "Test 3", "Status": "Fail", "Duration": "2.1s"}
    ]
    logger.display_table(data, "Test Results")
    
    # Test system info
    logger.display_system_info()
    
    logger.info("Logger test completed")


if __name__ == "__main__":
    test_logger()
