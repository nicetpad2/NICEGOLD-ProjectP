# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.status import Status
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

"""
NICEGOLD ProjectP - Advanced Terminal Logger
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

Beautiful, modern terminal logging system with progress bars, 
colored output, and comprehensive error tracking.

Features:
- Animated progress bars with status updates
- Color - coded log levels (INFO, WARNING, ERROR, CRITICAL)
- Real - time status tracking with summary reports
- Beautiful ASCII art and formatting
- Thread - safe logging with performance monitoring
- Comprehensive error collection and reporting

Author: NICEGOLD Team
Version: 3.0
Created: 2025 - 01 - 05
"""


# Try to import rich for advanced terminal formatting
try:
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeRemainingColumn,
    )

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️  Rich library not available. Using basic terminal output.")


class LogLevel(Enum):
    """Log level enumeration with colors and priorities"""

    DEBUG = ("DEBUG", "dim white", 0)
    INFO = ("INFO", "bright_blue", 1)
    SUCCESS = ("SUCCESS", "bright_green", 2)
    WARNING = ("WARNING", "bright_yellow", 3)
    ERROR = ("ERROR", "bright_red", 4)
    CRITICAL = ("CRITICAL", "bold red on white", 5)

    def __init__(self, label: str, color: str, priority: int):
        self.label = label
        self.color = color
        self.priority = priority


@dataclass
class LogEntry:
    """Structured log entry with metadata"""

    timestamp: datetime
    level: LogLevel
    message: str
    module: str = "MAIN"
    details: Optional[str] = None
    exception: Optional[Exception] = None
    duration: Optional[float] = None


@dataclass
class LogSummary:
    """Summary of log session with statistics"""

    start_time: datetime
    end_time: Optional[datetime] = None
    total_entries: int = 0
    entries_by_level: Dict[LogLevel, int] = field(default_factory=dict)
    errors: List[LogEntry] = field(default_factory=list)
    warnings: List[LogEntry] = field(default_factory=list)
    critical_issues: List[LogEntry] = field(default_factory=list)
    total_duration: float = 0.0

    def add_entry(self, entry: LogEntry):
        """Add a log entry to the summary"""
        self.total_entries += 1
        if entry.level not in self.entries_by_level:
            self.entries_by_level[entry.level] = 0
        self.entries_by_level[entry.level] += 1

        # Collect issues for final report
        if entry.level == LogLevel.ERROR:
            self.errors.append(entry)
        elif entry.level == LogLevel.WARNING:
            self.warnings.append(entry)
        elif entry.level == LogLevel.CRITICAL:
            self.critical_issues.append(entry)


class AdvancedTerminalLogger:
    """
    Advanced terminal logger with beautiful formatting and progress tracking
    """

    def __init__(self, title: str = "NICEGOLD ProjectP", show_timestamps: bool = True):
        self.title = title
        self.show_timestamps = show_timestamps
        self.summary = LogSummary(start_time=datetime.now())
        self.active_tasks = {}
        self.lock = threading.Lock()

        # Initialize console
        if RICH_AVAILABLE:
            self.console = Console(force_terminal=True, color_system="256")
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                console=self.console,
                expand=True,
            )
        else:
            self.console = None
            self.progress = None

        # Color mapping for basic terminal
        self.basic_colors = {
            LogLevel.DEBUG: "\033[90m",  # Dim white
            LogLevel.INFO: "\033[94m",  # Bright blue
            LogLevel.SUCCESS: "\033[92m",  # Bright green
            LogLevel.WARNING: "\033[93m",  # Bright yellow
            LogLevel.ERROR: "\033[91m",  # Bright red
            LogLevel.CRITICAL: "\033[97;41m",  # White on red
        }
        self.reset_color = "\033[0m"

        # Start session
        self._print_header()

    def _print_header(self):
        """Print beautiful session header"""
        if RICH_AVAILABLE:
            # Rich header
            header_text = Text(
                f"🚀 {self.title} - Advanced Logging Session", style="bold bright_cyan"
            )
            subtitle = Text(
                f"Session started: {self.summary.start_time.strftime('%Y - %m - %d %H:%M:%S')}",
                style="dim",
            )

            header_panel = Panel(
                Align.center(f"{header_text}\n{subtitle}"),
                border_style="bright_yellow",
                padding=(1, 2),
            )
            self.console.print(header_panel)
        else:
            # Basic header
            print("\n" + " = " * 80)
            print(f"🚀 {self.title} - Advanced Logging Session")
            print(
                f"Session started: {self.summary.start_time.strftime('%Y - %m - %d %H:%M:%S')}"
            )
            print(" = " * 80 + "\n")

    def log(
        self,
        level: LogLevel,
        message: str,
        module: str = "MAIN",
        details: Optional[str] = None,
        exception: Optional[Exception] = None,
    ):
        """Log a message with specified level"""
        with self.lock:
            entry = LogEntry(
                timestamp=datetime.now(),
                level=level,
                message=message,
                module=module,
                details=details,
                exception=exception,
            )

            self.summary.add_entry(entry)
            self._print_log_entry(entry)

    def _print_log_entry(self, entry: LogEntry):
        """Print a single log entry with formatting"""
        timestamp_str = (
            entry.timestamp.strftime("%H:%M:%S") if self.show_timestamps else ""
        )

        if RICH_AVAILABLE:
            # Rich formatting
            timestamp_text = (
                Text(f"[{timestamp_str}]", style="dim") if timestamp_str else Text("")
            )
            level_text = Text(f"[{entry.level.label:>8}]", style=entry.level.color)
            module_text = Text(f"[{entry.module}]", style="cyan")
            message_text = Text(entry.message, style="white")

            # Combine elements
            log_line = Text.assemble(
                timestamp_text,
                " " if timestamp_str else "",
                level_text,
                " ",
                module_text,
                " ",
                message_text,
            )

            self.console.print(log_line)

            # Print additional details if available
            if entry.details:
                details_text = Text(f"    ℹ️  {entry.details}", style="dim")
                self.console.print(details_text)

            if entry.exception:
                error_text = Text(f"    ❌ {str(entry.exception)}", style="red")
                self.console.print(error_text)
        else:
            # Basic formatting
            color = self.basic_colors.get(entry.level, "")
            timestamp_part = f"[{timestamp_str}]" if timestamp_str else ""
            level_part = f"[{entry.level.label:>8}]"
            module_part = f"[{entry.module}]"

            print(
                f"{color}{timestamp_part} {level_part} {module_part} {entry.message}{self.reset_color}"
            )

            if entry.details:
                print(f"    ℹ️  {entry.details}")

            if entry.exception:
                print(f"    ❌ {str(entry.exception)}")

    def debug(self, message: str, module: str = "MAIN", details: Optional[str] = None):
        """Log debug message"""
        self.log(LogLevel.DEBUG, message, module, details)

    def info(self, message: str, module: str = "MAIN", details: Optional[str] = None):
        """Log info message"""
        self.log(LogLevel.INFO, message, module, details)

    def success(
        self, message: str, module: str = "MAIN", details: Optional[str] = None
    ):
        """Log success message"""
        self.log(LogLevel.SUCCESS, message, module, details)

    def warning(
        self, message: str, module: str = "MAIN", details: Optional[str] = None
    ):
        """Log warning message"""
        self.log(LogLevel.WARNING, message, module, details)

    def error(
        self,
        message: str,
        module: str = "MAIN",
        details: Optional[str] = None,
        exception: Optional[Exception] = None,
    ):
        """Log error message"""
        self.log(LogLevel.ERROR, message, module, details, exception)

    def critical(
        self,
        message: str,
        module: str = "MAIN",
        details: Optional[str] = None,
        exception: Optional[Exception] = None,
    ):
        """Log critical message"""
        self.log(LogLevel.CRITICAL, message, module, details, exception)

    def start_progress(
        self, task_id: str, description: str, total: Optional[int] = None
    ):
        """Start a progress task"""
        if RICH_AVAILABLE and self.progress:
            with self.lock:
                if task_id not in self.active_tasks:
                    task = self.progress.add_task(description, total=total)
                    self.active_tasks[task_id] = task
                    return task
        else:
            print(f"🔄 Starting: {description}")
            return None

    def update_progress(
        self, task_id: str, advance: int = 1, description: Optional[str] = None
    ):
        """Update progress for a task"""
        if RICH_AVAILABLE and self.progress and task_id in self.active_tasks:
            with self.lock:
                task = self.active_tasks[task_id]
                self.progress.update(task, advance=advance)
                if description:
                    self.progress.update(task, description=description)
        else:
            if description:
                print(f"⏳ Progress: {description}")

    def complete_progress(self, task_id: str, message: Optional[str] = None):
        """Complete a progress task"""
        if RICH_AVAILABLE and self.progress and task_id in self.active_tasks:
            with self.lock:
                task = self.active_tasks[task_id]
                self.progress.update(task, completed=True)
                if message:
                    self.progress.update(task, description=f"✅ {message}")
                del self.active_tasks[task_id]
        else:
            if message:
                print(f"✅ Completed: {message}")

    def progress_context(self):
        """Context manager for progress display"""
        if RICH_AVAILABLE and self.progress:
            return self.progress
        else:
            return self._dummy_context()

    def _dummy_context(self):
        """Dummy context manager for non - rich environments"""

        class DummyContext:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        return DummyContext()

    def status_context(self, message: str):
        """Context manager for status display"""
        if RICH_AVAILABLE:
            return Status(message, console=self.console, spinner="dots")
        else:
            return self._status_context(message)

    def _status_context(self, message: str):
        """Basic status context for non - rich environments"""

        class BasicStatus:
            def __init__(self, msg):
                self.message = msg

            def __enter__(self):
                print(f"🔄 {self.message}")
                return self

            def __exit__(self, *args):
                pass

        return BasicStatus(message)

    def print_table(
        self,
        title: str,
        data: List[Dict[str, Any]],
        headers: Optional[List[str]] = None,
    ):
        """Print a formatted table"""
        if RICH_AVAILABLE and data:
            table = Table(title=title, border_style="cyan")

            # Add columns
            if headers:
                for header in headers:
                    table.add_column(header, style="bright_white")
            else:
                for key in data[0].keys():
                    table.add_column(key.title(), style="bright_white")

            # Add rows
            for row in data:
                if headers:
                    table.add_row(*[str(row.get(h, "")) for h in headers])
                else:
                    table.add_row(*[str(v) for v in row.values()])

            self.console.print(table)
        else:
            # Basic table
            print(f"\n📊 {title}")
            print(" - " * 50)
            for item in data:
                for key, value in item.items():
                    print(f"  {key}: {value}")
                print()

    def print_summary(self):
        """Print comprehensive session summary"""
        self.summary.end_time = datetime.now()
        self.summary.total_duration = (
            self.summary.end_time - self.summary.start_time
        ).total_seconds()

        if RICH_AVAILABLE:
            self._print_rich_summary()
        else:
            self._print_basic_summary()

    def _print_rich_summary(self):
        """Print rich formatted summary"""
        # Summary statistics
        stats_table = Table(title="📊 Session Statistics", border_style="green")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="bright_white")

        stats_table.add_row("Duration", f"{self.summary.total_duration:.2f} seconds")
        stats_table.add_row("Total Log Entries", str(self.summary.total_entries))

        for level, count in self.summary.entries_by_level.items():
            stats_table.add_row(f"{level.label} Messages", str(count))

        self.console.print(stats_table)

        # Issues summary
        if self.summary.critical_issues or self.summary.errors or self.summary.warnings:
            issues_panel = self._create_issues_panel()
            self.console.print(issues_panel)
        else:
            success_panel = Panel(
                Align.center(
                    "🎉 No Issues Detected!\nAll operations completed successfully."
                ),
                border_style="green",
                title="✅ Session Status",
            )
            self.console.print(success_panel)

        # Footer
        footer = Panel(
            Align.center(
                f"Session completed at {self.summary.end_time.strftime('%Y - %m - %d %H:%M:%S')}"
            ),
            border_style="blue",
        )
        self.console.print(footer)

    def _create_issues_panel(self) -> Panel:
        """Create issues summary panel"""
        issues_tree = Tree("🚨 Issues Summary")

        if self.summary.critical_issues:
            critical_branch = issues_tree.add(
                f"💥 Critical Issues ({len(self.summary.critical_issues)})"
            )
            for issue in self.summary.critical_issues[:5]:  # Show top 5
                critical_branch.add(
                    f"[red]{issue.message[:60]}...[/red]"
                    if len(issue.message) > 60
                    else f"[red]{issue.message}[/red]"
                )

        if self.summary.errors:
            error_branch = issues_tree.add(f"❌ Errors ({len(self.summary.errors)})")
            for error in self.summary.errors[:5]:  # Show top 5
                error_branch.add(
                    f"[bright_red]{error.message[:60]}...[/bright_red]"
                    if len(error.message) > 60
                    else f"[bright_red]{error.message}[/bright_red]"
                )

        if self.summary.warnings:
            warning_branch = issues_tree.add(
                f"⚠️  Warnings ({len(self.summary.warnings)})"
            )
            for warning in self.summary.warnings[:5]:  # Show top 5
                warning_branch.add(
                    f"[yellow]{warning.message[:60]}...[/yellow]"
                    if len(warning.message) > 60
                    else f"[yellow]{warning.message}[/yellow]"
                )

        return Panel(issues_tree, border_style="red", title="🚨 Issues Detected")

    def _print_basic_summary(self):
        """Print basic formatted summary"""
        print("\n" + " = " * 80)
        print("📊 SESSION SUMMARY")
        print(" = " * 80)

        print(f"⏱️  Duration: {self.summary.total_duration:.2f} seconds")
        print(f"📝 Total Log Entries: {self.summary.total_entries}")

        print("\n📈 Log Level Distribution:")
        for level, count in self.summary.entries_by_level.items():
            print(f"  {level.label}: {count}")

        # Issues summary
        total_issues = (
            len(self.summary.critical_issues)
            + len(self.summary.errors)
            + len(self.summary.warnings)
        )

        if total_issues > 0:
            print(f"\n🚨 ISSUES DETECTED ({total_issues} total)")
            print(" - " * 40)

            if self.summary.critical_issues:
                print(f"💥 Critical Issues: {len(self.summary.critical_issues)}")
                for issue in self.summary.critical_issues[:3]:
                    print(f"   • {issue.message}")

            if self.summary.errors:
                print(f"❌ Errors: {len(self.summary.errors)}")
                for error in self.summary.errors[:3]:
                    print(f"   • {error.message}")

            if self.summary.warnings:
                print(f"⚠️  Warnings: {len(self.summary.warnings)}")
                for warning in self.summary.warnings[:3]:
                    print(f"   • {warning.message}")
        else:
            print("\n🎉 NO ISSUES DETECTED!")
            print("All operations completed successfully.")

        print(
            f"\nSession completed at {self.summary.end_time.strftime('%Y - %m - %d %H:%M:%S')}"
        )
        print(" = " * 80)


# Global logger instance
_global_logger: Optional[AdvancedTerminalLogger] = None


def get_logger(title: str = "NICEGOLD ProjectP") -> AdvancedTerminalLogger:
    """Get or create global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = AdvancedTerminalLogger(title)
    return _global_logger


def demo_logging_system():
    """Demonstrate the advanced logging system"""
    logger = get_logger("Demo Session")

    logger.info("Starting demonstration of advanced logging system")

    # Simulate some operations with progress
    with logger.progress_context():
        # Data loading simulation
        task_id = "data_load"
        logger.start_progress(task_id, "Loading CSV data...", total=100)

        for i in range(0, 101, 10):
            time.sleep(0.1)
            logger.update_progress(task_id, advance=10)

            if i == 30:
                logger.warning(
                    "Missing column detected in CSV",
                    "CSV_LOADER",
                    "Column 'timestamp' not found, using 'Date' instead",
                )
            elif i == 60:
                logger.error(
                    "Invalid data format detected",
                    "DATA_VALIDATOR",
                    "Row 1250 contains invalid price value",
                    ValueError("Invalid price: 'N/A'"),
                )

        logger.complete_progress(task_id, "CSV data loaded successfully")

        # Feature engineering simulation
        logger.info("Starting feature engineering process", "FEATURE_ENG")

        with logger.status_context("Creating technical indicators..."):
            time.sleep(1)
            logger.success("Created 15 technical indicators", "FEATURE_ENG")

        # Model training simulation
        logger.info("Initializing machine learning models", "ML_TRAINER")

        model_task = "model_train"
        logger.start_progress(model_task, "Training RandomForest model...", total=50)

        for epoch in range(50):
            time.sleep(0.05)
            logger.update_progress(model_task, advance=1)

            if epoch == 25:
                logger.warning(
                    "Model convergence slow",
                    "ML_TRAINER",
                    "Consider increasing learning rate",
                )

        logger.complete_progress(model_task, "Model training completed")

        # Simulate critical error
        logger.critical(
            "Memory allocation failed",
            "SYSTEM",
            "Insufficient memory for large dataset",
            MemoryError("Cannot allocate 8GB"),
        )

        # Final operations
        logger.success("Pipeline execution completed", "MAIN")

    # Print comprehensive summary
    logger.print_summary()


if __name__ == "__main__":
    # Run demonstration
    demo_logging_system()
