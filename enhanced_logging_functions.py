# -*- coding: utf - 8 -* -
#!/usr/bin/env python3
import sys
import time
from datetime import datetime
from pathlib import Path

from src.advanced_logger import AdvancedTerminalLogger, LogLevel, get_logger

"""
NICEGOLD ProjectP - Enhanced Logging Integration
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

Modern, beautiful terminal logging system with progress bars, 
color - coded messages, and comprehensive error tracking.

This module provides enhanced logging functions that can be used
throughout the ProjectP.py system to create a modern, clean
terminal experience.

Features:
- Beautiful progress bars with status updates
- Color - coded log levels (INFO, WARNING, ERROR, CRITICAL)
- Real - time status tracking with summary reports
- Clean, uncluttered terminal output
- Comprehensive error collection and reporting
- Session summaries with issue tracking

Author: NICEGOLD Team
Version: 3.0
Created: 2025 - 01 - 05
"""


# Add project root to path if needed
PROJECT_ROOT = Path(__file__).parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Try to import the advanced logger
try:

    LOGGER_AVAILABLE = True
except ImportError:
    LOGGER_AVAILABLE = False
    print("⚠️  Advanced logger not available. Using fallback logging.")

# Global logger instance
_session_logger = None


def init_session_logger(title="NICEGOLD ProjectP"):
    """Initialize the session logger"""
    global _session_logger
    if LOGGER_AVAILABLE and _session_logger is None:
        _session_logger = get_logger(title)
    return _session_logger


def get_session_logger():
    """Get the current session logger"""
    global _session_logger
    if _session_logger is None:
        _session_logger = init_session_logger()
    return _session_logger


def log_info(message, module="MAIN", details=None):
    """Log info message with fallback"""
    logger = get_session_logger()
    if logger:
        logger.info(message, module, details)
    else:
        print(f"ℹ️  [{module}] {message}")
        if details:
            print(f"    💡 {details}")


def log_success(message, module="MAIN", details=None):
    """Log success message with fallback"""
    logger = get_session_logger()
    if logger:
        logger.success(message, module, details)
    else:
        print(f"✅ [{module}] {message}")
        if details:
            print(f"    💡 {details}")


def log_warning(message, module="MAIN", details=None):
    """Log warning message with fallback"""
    logger = get_session_logger()
    if logger:
        logger.warning(message, module, details)
    else:
        print(f"⚠️  [{module}] {message}")
        if details:
            print(f"    💡 {details}")


def log_error(message, module="MAIN", details=None, exception=None):
    """Log error message with fallback"""
    logger = get_session_logger()
    if logger:
        logger.error(message, module, details, exception)
    else:
        print(f"❌ [{module}] {message}")
        if details:
            print(f"    💡 {details}")
        if exception:
            print(f"    🐛 {str(exception)}")


def log_critical(message, module="MAIN", details=None, exception=None):
    """Log critical message with fallback"""
    logger = get_session_logger()
    if logger:
        logger.critical(message, module, details, exception)
    else:
        print(f"💥 [{module}] {message}")
        if details:
            print(f"    💡 {details}")
        if exception:
            print(f"    🐛 {str(exception)}")


def start_progress_task(task_id, description, total=None):
    """Start a progress task with fallback"""
    logger = get_session_logger()
    if logger:
        return logger.start_progress(task_id, description, total)
    else:
        print(f"🔄 Starting: {description}")
        return None


def update_progress_task(task_id, advance=1, description=None):
    """Update progress task with fallback"""
    logger = get_session_logger()
    if logger:
        logger.update_progress(task_id, advance, description)
    else:
        if description:
            print(f"⏳ Progress: {description}")


def complete_progress_task(task_id, message=None):
    """Complete progress task with fallback"""
    logger = get_session_logger()
    if logger:
        logger.complete_progress(task_id, message)
    else:
        if message:
            print(f"✅ Completed: {message}")


def progress_context():
    """Get progress context manager"""
    logger = get_session_logger()
    if logger:
        return logger.progress_context()
    else:
        return _dummy_context()


def status_context(message):
    """Get status context manager"""
    logger = get_session_logger()
    if logger:
        return logger.status_context(message)
    else:
        return _status_context(message)


def print_data_table(title, data, headers=None):
    """Print formatted data table"""
    logger = get_session_logger()
    if logger:
        logger.print_table(title, data, headers)
    else:
        print(f"\n📊 {title}")
        print(" - " * 50)
        for item in data:
            for key, value in item.items():
                print(f"  {key}: {value}")
            print()


def display_session_summary():
    """Display comprehensive session summary"""
    logger = get_session_logger()
    if logger:
        logger.print_summary()
    else:
        print(f"\n📊 SESSION SUMMARY")
        print(" = " * 50)
        print(f"⏱️  Session completed at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"✅ Operations completed")
        print(" = " * 50)


def _dummy_context():
    """Dummy context manager for fallback"""

    class DummyContext:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    return DummyContext()


def _status_context(message):
    """Basic status context for fallback"""

    class BasicStatus:
        def __init__(self, msg):
            self.message = msg

        def __enter__(self):
            print(f"🔄 {self.message}")
            return self

        def __exit__(self, *args):
            pass

    return BasicStatus(message)


# CSV Manager enhanced logging functions
def log_csv_validation_start(module="CSV"):
    """Log start of CSV validation"""
    with status_context("Validating CSV files..."):
        log_info("Starting CSV file validation", module)


def log_csv_file_found(filename, size_mb, module="CSV"):
    """Log CSV file found"""
    log_success(f"CSV file found: {filename}", module, f"Size: {size_mb:.2f} MB")


def log_csv_file_error(filename, error, module="CSV"):
    """Log CSV file error"""
    log_error(f"CSV file error: {filename}", module, str(error))


def log_csv_validation_complete(total_files, valid_files, module="CSV"):
    """Log CSV validation completion"""
    if valid_files == total_files:
        log_success(
            f"CSV validation completed: {valid_files}/{total_files} files valid", module
        )
    elif valid_files > 0:
        log_warning(
            f"CSV validation completed with warnings: {valid_files}/{total_files} files valid",
            module,
        )
    else:
        log_error(f"CSV validation failed: 0/{total_files} files valid", module)


# Pipeline enhanced logging functions
def log_pipeline_start(module="PIPELINE"):
    """Log pipeline start"""
    log_info("Starting pipeline execution", module)


def log_pipeline_step(step_name, module="PIPELINE"):
    """Log pipeline step"""
    log_info(f"Pipeline step: {step_name}", module)


def log_pipeline_success(module="PIPELINE"):
    """Log pipeline success"""
    log_success("Pipeline execution completed successfully", module)


def log_pipeline_error(error, module="PIPELINE"):
    """Log pipeline error"""
    log_error("Pipeline execution failed", module, str(error), error)


# Data processing enhanced logging functions
def log_data_loading_start(filename, module="DATA"):
    """Log data loading start"""
    log_info(f"Loading data from: {filename}", module)


def log_data_loaded(rows, columns, module="DATA"):
    """Log data loaded successfully"""
    log_success(f"Data loaded successfully: {rows} rows, {columns} columns", module)


def log_data_quality_check(missing_values, duplicates, module="DATA"):
    """Log data quality check results"""
    if missing_values == 0 and duplicates == 0:
        log_success("Data quality check passed - no issues found", module)
    else:
        if missing_values > 0:
            log_warning(f"Found {missing_values} missing values", module)
        if duplicates > 0:
            log_warning(f"Found {duplicates} duplicate rows", module)


# System health enhanced logging functions
def log_system_check_start(module="SYSTEM"):
    """Log system health check start"""
    log_info("Starting system health check", module)


def log_package_status(category, installed, total, module="PACKAGES"):
    """Log package status"""
    if installed == total:
        log_success(f"{category}: {installed}/{total} packages installed", module)
    elif installed > total * 0.8:  # 80% threshold
        log_warning(f"{category}: {installed}/{total} packages installed", module)
    else:
        log_error(f"{category}: {installed}/{total} packages installed", module)


def log_missing_package(package_name, module="PACKAGES"):
    """Log missing package"""
    log_warning(f"Missing package: {package_name}", module)


def log_system_check_complete(module="SYSTEM"):
    """Log system health check completion"""
    log_success("System health check completed", module)


# Initialize the logger when this module is imported
if LOGGER_AVAILABLE:
    init_session_logger()
