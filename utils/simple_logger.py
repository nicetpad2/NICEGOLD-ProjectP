#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Modern Logger for NICEGOLD ProjectP v2.0
Simple, beautiful and effective logging system
"""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Try to import rich for enhanced output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.status import Status
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Import colors
try:
    from utils.colors import Colors, colorize
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    # Fallback colors
    class Colors:
        RESET = "\033[0m"
        BRIGHT_CYAN = "\033[96m"
        BRIGHT_GREEN = "\033[92m"
        BRIGHT_YELLOW = "\033[93m"
        BRIGHT_RED = "\033[91m"
        BRIGHT_BLUE = "\033[94m"
        WHITE = "\033[97m"
        DIM = "\033[2m"
    
    def colorize(text, color):
        return f"{color}{text}{Colors.RESET}"


class SimpleModernLogger:
    """Simple modern logger with enhanced features"""
    
    def __init__(self, name="ProjectP", enable_file_logging=True):
        self.name = name
        self.start_time = datetime.now()
        self.stats = {"info": 0, "success": 0, "warning": 0, "error": 0, "critical": 0}
        
        # Setup console
        self.console = Console() if RICH_AVAILABLE else None
        
        # Setup file logging
        if enable_file_logging:
            self._setup_file_logging()
    
    def _setup_file_logging(self):
        """Setup file logging"""
        try:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            log_file = log_dir / f"projectp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
            self.file_logger = logging.getLogger(self.name)
            self.file_logger.setLevel(logging.DEBUG)
            
            handler = logging.FileHandler(log_file, encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            
            self.file_logger.addHandler(handler)
            self.log_file = log_file
        except Exception:
            self.file_logger = None
            self.log_file = None
    
    def _log_to_file(self, level, message):
        """Log to file"""
        if self.file_logger:
            getattr(self.file_logger, level.lower())(message)
    
    def log_info(self, message, **kwargs):
        """Log info message"""
        self.stats["info"] += 1
        self._log_to_file("INFO", message)
        
        if RICH_AVAILABLE and self.console:
            self.console.print(f"ℹ️ [blue][INFO][/blue] {message}")
        else:
            print(f"ℹ️ [INFO] {message}")
    
    def log_success(self, message, **kwargs):
        """Log success message"""
        self.stats["success"] += 1
        self._log_to_file("INFO", f"SUCCESS: {message}")
        
        if RICH_AVAILABLE and self.console:
            self.console.print(f"✅ [green][SUCCESS][/green] {message}")
        else:
            print(f"✅ [SUCCESS] {message}")
    
    def log_warning(self, message, **kwargs):
        """Log warning message"""
        self.stats["warning"] += 1
        self._log_to_file("WARNING", message)
        
        if RICH_AVAILABLE and self.console:
            self.console.print(f"⚠️ [yellow][WARNING][/yellow] {message}")
        else:
            print(f"⚠️ [WARNING] {message}")
    
    def log_error(self, message, **kwargs):
        """Log error message"""
        self.stats["error"] += 1
        self._log_to_file("ERROR", message)
        
        if RICH_AVAILABLE and self.console:
            self.console.print(f"❌ [red][ERROR][/red] {message}")
        else:
            print(f"❌ [ERROR] {message}")
    
    def log_critical(self, message, **kwargs):
        """Log critical message"""
        self.stats["critical"] += 1
        self._log_to_file("CRITICAL", message)
        
        if RICH_AVAILABLE and self.console:
            self.console.print(f"🚨 [bold red][CRITICAL][/bold red] {message}")
        else:
            print(f"🚨 [CRITICAL] {message}")
    
    def log_progress(self, message, **kwargs):
        """Log progress message"""
        if RICH_AVAILABLE and self.console:
            self.console.print(f"⏳ [cyan][PROGRESS][/cyan] {message}")
        else:
            print(f"⏳ [PROGRESS] {message}")
    
    def display_summary(self):
        """Display session summary"""
        uptime = datetime.now() - self.start_time
        total = sum(self.stats.values())
        
        if RICH_AVAILABLE and self.console:
            table = Table(title="📊 Session Summary", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("⏱️ Duration", str(uptime).split('.')[0])
            table.add_row("📝 Total Messages", str(total))
            table.add_row("ℹ️ Info", str(self.stats["info"]))
            table.add_row("✅ Success", str(self.stats["success"]))
            table.add_row("⚠️ Warning", str(self.stats["warning"]))
            table.add_row("❌ Error", str(self.stats["error"]))
            table.add_row("🚨 Critical", str(self.stats["critical"]))
            
            self.console.print(table)
            self.console.print("[bold green]Thank you for using NICEGOLD ProjectP![/bold green]")
        else:
            print(f"\n{colorize('📊 SESSION SUMMARY', Colors.BRIGHT_CYAN)}")
            print(f"{colorize('═' * 50, Colors.BRIGHT_CYAN)}")
            print(f"⏱️ Duration: {str(uptime).split('.')[0]}")
            print(f"📝 Messages: {total}")
            print(f"ℹ️ Info: {self.stats['info']}")
            print(f"✅ Success: {self.stats['success']}")
            print(f"⚠️ Warning: {self.stats['warning']}")
            print(f"❌ Error: {self.stats['error']}")
            print(f"🚨 Critical: {self.stats['critical']}")
            print(f"{colorize('═' * 50, Colors.BRIGHT_CYAN)}")
    
    def show_session_summary(self):
        """Backward compatibility"""
        self.display_summary()


# Global logger instance
_logger: Optional[SimpleModernLogger] = None


def setup_logger(name="NICEGOLD_ProjectP", enable_file_logging=True, enable_sound=False):
    """Setup global logger"""
    global _logger
    _logger = SimpleModernLogger(name, enable_file_logging)
    return _logger


def get_logger():
    """Get global logger"""
    return _logger


# Convenience functions
def info(message, **kwargs):
    if _logger:
        _logger.log_info(message, **kwargs)
    else:
        print(f"ℹ️ [INFO] {message}")


def success(message, **kwargs):
    if _logger:
        _logger.log_success(message, **kwargs)
    else:
        print(f"✅ [SUCCESS] {message}")


def warning(message, **kwargs):
    if _logger:
        _logger.log_warning(message, **kwargs)
    else:
        print(f"⚠️ [WARNING] {message}")


def error(message, **kwargs):
    if _logger:
        _logger.log_error(message, **kwargs)
    else:
        print(f"❌ [ERROR] {message}")


def critical(message, **kwargs):
    if _logger:
        _logger.log_critical(message, **kwargs)
    else:
        print(f"🚨 [CRITICAL] {message}")


def progress(message, **kwargs):
    if _logger:
        _logger.log_progress(message, **kwargs)
    else:
        print(f"⏳ [PROGRESS] {message}")


def display_summary():
    if _logger:
        _logger.display_summary()
    else:
        print("📊 No session to summarize")
