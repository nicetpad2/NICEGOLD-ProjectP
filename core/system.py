#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System utilities for NICEGOLD ProjectP
Handles system operations, file management, and monitoring
"""

import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil

from utils.colors import Colors, colorize


class SystemManager:
    """System management utilities"""

    def __init__(self):
        self.project_root = Path.cwd()
        self.python_executable = sys.executable

    def check_system_health(self) -> Dict[str, Any]:
        """Check system health and resources"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Python process info
            try:
                process = psutil.Process()
                python_memory = process.memory_info().rss / 1024 / 1024  # MB
                python_cpu = process.cpu_percent()
            except:
                python_memory = 0
                python_cpu = 0

            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3),
                "python_memory_mb": python_memory,
                "python_cpu_percent": python_cpu,
                "status": (
                    "healthy" if cpu_percent < 80 and memory.percent < 85 else "warning"
                ),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def display_system_status(self):
        """Display beautiful system status"""
        health = self.check_system_health()

        print(f"\n{colorize('🖥️  SYSTEM STATUS', Colors.BRIGHT_CYAN)}")
        print(f"{colorize('═' * 50, Colors.BRIGHT_CYAN)}")

        if health["status"] == "error":
            print(
                f"{colorize('❌ Error checking system:', Colors.BRIGHT_RED)} {health.get('error', 'Unknown error')}"
            )
            return

        # CPU Status
        cpu_color = (
            Colors.BRIGHT_GREEN
            if health["cpu_percent"] < 50
            else (
                Colors.BRIGHT_YELLOW
                if health["cpu_percent"] < 80
                else Colors.BRIGHT_RED
            )
        )
        cpu_text = f"{health['cpu_percent']:.1f}%"
        print(f"🔥 CPU Usage: {colorize(cpu_text, cpu_color)}")

        # Memory Status
        mem_color = (
            Colors.BRIGHT_GREEN
            if health["memory_percent"] < 60
            else (
                Colors.BRIGHT_YELLOW
                if health["memory_percent"] < 80
                else Colors.BRIGHT_RED
            )
        )
        mem_text = f"{health['memory_percent']:.1f}%"
        print(
            f"💾 Memory Usage: {colorize(mem_text, mem_color)} ({health['memory_available_gb']:.1f} GB available)"
        )

        # Disk Status
        disk_color = (
            Colors.BRIGHT_GREEN
            if health["disk_percent"] < 70
            else (
                Colors.BRIGHT_YELLOW
                if health["disk_percent"] < 90
                else Colors.BRIGHT_RED
            )
        )
        disk_text = f"{health['disk_percent']:.1f}%"
        print(
            f"💿 Disk Usage: {colorize(disk_text, disk_color)} ({health['disk_free_gb']:.1f} GB free)"
        )

        # Python Process
        print(
            f"🐍 Python Process: {health['python_memory_mb']:.1f} MB, {health['python_cpu_percent']:.1f}% CPU"
        )

        # Overall Status
        status_color = (
            Colors.BRIGHT_GREEN
            if health["status"] == "healthy"
            else Colors.BRIGHT_YELLOW
        )
        status_text = "✅ Healthy" if health["status"] == "healthy" else "⚠️ Under Load"
        print(f"📊 Overall Status: {colorize(status_text, status_color)}")

        print(f"{colorize('═' * 50, Colors.BRIGHT_CYAN)}")

    def run_command(
        self, command: List[str], description: str = "", timeout: int = 300
    ) -> bool:
        """Run system command with timeout and error handling"""
        try:
            print(f"🚀 {description if description else 'Running command'}")
            print(f"📋 Command: {' '.join(command)}")

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.project_root,
            )

            if result.returncode == 0:
                print(f"✅ {description} completed successfully")
                if result.stdout.strip():
                    print(f"📄 Output: {result.stdout.strip()}")
                return True
            else:
                print(f"❌ {description} failed with return code {result.returncode}")
                if result.stderr.strip():
                    print(f"🚨 Error: {result.stderr.strip()}")
                return False

        except subprocess.TimeoutExpired:
            print(f"⏰ {description} timed out after {timeout} seconds")
            return False
        except Exception as e:
            print(f"💥 Error running {description}: {e}")
            return False

    def install_packages(self, packages: List[str]) -> bool:
        """Install Python packages"""
        print(f"📦 Installing {len(packages)} packages...")

        for package in packages:
            print(f"📥 Installing {package}...")
            success = self.run_command(
                [self.python_executable, "-m", "pip", "install", package],
                f"Install {package}",
            )
            if not success:
                print(f"⚠️ Failed to install {package}")

        return True

    def clean_directories(self, directories: List[str]) -> Tuple[int, float]:
        """Clean specified directories and return (files_cleaned, size_mb)"""
        total_cleaned = 0
        total_size_mb = 0.0

        for dir_name in directories:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                try:
                    if dir_name == "__pycache__":
                        # Clean all __pycache__ directories recursively
                        for cache_path in self.project_root.rglob("__pycache__"):
                            size = sum(
                                f.stat().st_size
                                for f in cache_path.rglob("*")
                                if f.is_file()
                            )
                            shutil.rmtree(cache_path)
                            total_size_mb += size / (1024 * 1024)
                            total_cleaned += 1
                            print(f"  🗑️ Removed {cache_path}")
                    else:
                        size = sum(
                            f.stat().st_size for f in dir_path.rglob("*") if f.is_file()
                        )
                        shutil.rmtree(dir_path)
                        total_size_mb += size / (1024 * 1024)
                        total_cleaned += 1
                        print(f"  🗑️ Removed {dir_path}")
                except Exception as e:
                    print(f"  ❌ Error removing {dir_path}: {e}")

        return total_cleaned, total_size_mb

    def get_file_info(self) -> Dict[str, List[Path]]:
        """Get information about project files"""
        search_patterns = {
            "Log Files": ["**/*.log", "**/logs/**/*.txt"],
            "Results": ["**/*results*.csv", "**/*analysis*.csv", "**/*backtest*.csv"],
            "Models": ["**/*.joblib", "**/*.pkl", "**/*.model"],
            "Reports": ["**/*report*.txt", "**/*summary*.txt"],
            "Data": ["**/*.csv", "**/*.parquet"],
            "Plots": ["**/*.png", "**/*.jpg", "**/*.svg"],
        }

        all_files = {}

        for category, patterns in search_patterns.items():
            category_files = []
            for pattern in patterns:
                found_files = list(self.project_root.glob(pattern))
                category_files.extend(found_files)

            # Remove duplicates and sort by modification time
            category_files = list(set(category_files))
            category_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            all_files[category] = category_files

        return all_files

    def format_file_size(self, size_bytes: int) -> str:
        """Convert bytes to human readable format"""
        if size_bytes == 0:
            return "0 B"
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"


# Global system manager instance
system_manager = SystemManager()


def get_system():
    """Get the global system manager instance"""
    return system_manager


def check_system_health():
    """Check system health using global instance"""
    return system_manager.check_system_health()


def run_command(command, timeout=300):
    """Run command using global instance"""
    return system_manager.run_command(command, timeout)


def cleanup_files(patterns):
    """Cleanup files using global instance"""
    return system_manager.cleanup_files(patterns)
