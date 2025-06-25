# -*- coding: utf-8 -*-
"""
NICEGOLD ProjectP - System Health Monitor Module
════════════════════════════════════════════════════════════════════════════════

Comprehensive system health checking and monitoring functionality.

Author: NICEGOLD Team
Version: 3.0
Created: 2025-01-05
"""

import os
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# แก้ไข import path
try:
    from src.core.colors import Colors, ColorThemes, colorize
except ImportError:
    # Fallback สำหรับกรณีที่ run จาก directory อื่น
    sys.path.append(".")
    from src.core.colors import Colors, ColorThemes, colorize


class SystemHealthMonitor:
    """System health monitoring and checking"""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.health_data = {}

    def check_system_health(self) -> Dict[str, Any]:
        """Comprehensive system health check"""
        health = {
            "python_version": platform.python_version(),
            "platform": f"{platform.system()} {platform.release()}",
            "working_dir": str(self.project_root),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "packages": {},
            "data_files": {},
            "directories": {},
            "missing_packages": [],
            "system_resources": {},
            "disk_space": {},
        }

        # Check system resources
        health["system_resources"] = self._check_system_resources()

        # Check disk space
        health["disk_space"] = self._check_disk_space()

        # Check packages
        health["packages"], health["missing_packages"] = self._check_packages()

        # Check directories
        health["directories"] = self._check_directories()

        # Check data files
        health["data_files"] = self._check_data_files()

        self.health_data = health
        return health

    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resources (CPU, Memory, etc.)"""
        try:
            import psutil

            # CPU information
            cpu_info = {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "cpu_usage": psutil.cpu_percent(interval=1),
                "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            }

            # Memory information
            memory = psutil.virtual_memory()
            memory_info = {
                "total": self._format_bytes(memory.total),
                "available": self._format_bytes(memory.available),
                "used": self._format_bytes(memory.used),
                "percentage": memory.percent,
            }

            return {"cpu": cpu_info, "memory": memory_info, "status": "available"}

        except ImportError:
            return {"status": "psutil_not_available"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space"""
        try:
            total, used, free = shutil.disk_usage(self.project_root)

            return {
                "total": self._format_bytes(total),
                "used": self._format_bytes(used),
                "free": self._format_bytes(free),
                "usage_percent": (used / total) * 100,
            }
        except Exception as e:
            return {"error": str(e)}

    def _check_packages(self) -> tuple[Dict[str, Dict[str, Any]], List[str]]:
        """Check essential packages"""
        package_categories = {
            "essential": [
                "pandas",
                "numpy",
                "sklearn",
                "matplotlib",
                "seaborn",
                "joblib",
                "yaml",
                "tqdm",
                "requests",
            ],
            "ml": [
                "catboost",
                "xgboost",
                "lightgbm",
                "optuna",
                "shap",
                "ta",
                "imblearn",
                "featuretools",
            ],
            "production": ["streamlit", "fastapi", "uvicorn", "pydantic", "pyarrow"],
            "gpu_optional": ["torch", "tensorflow"],
        }

        packages_status = {}
        missing_packages = []

        for category, package_list in package_categories.items():
            packages_status[category] = {}

            for pkg in package_list:
                try:
                    package_info = self._check_single_package(pkg)
                    packages_status[category][pkg] = package_info

                    if (
                        package_info["status"] == "missing"
                        and category != "gpu_optional"
                    ):
                        missing_packages.append(pkg)

                except Exception as e:
                    packages_status[category][pkg] = {
                        "status": "error",
                        "version": None,
                        "error": str(e),
                    }

        return packages_status, missing_packages

    def _check_single_package(self, pkg: str) -> Dict[str, Any]:
        """Check a single package"""
        try:
            if pkg == "tensorflow":
                # Special handling for TensorFlow
                os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
                import tensorflow as tf

                return {"status": "installed", "version": tf.__version__}
            elif pkg == "sklearn":
                import sklearn

                return {"status": "installed", "version": sklearn.__version__}
            elif pkg == "yaml":
                import yaml

                return {
                    "status": "installed",
                    "version": getattr(yaml, "__version__", "unknown"),
                }
            else:
                module = __import__(pkg)
                return {
                    "status": "installed",
                    "version": getattr(module, "__version__", "unknown"),
                }

        except ImportError:
            return {"status": "missing", "version": None}

    def _check_directories(self) -> Dict[str, Any]:
        """Check important directories"""
        important_dirs = [
            "datacsv",
            "output_default",
            "models",
            "logs",
            "config",
            "src",
            "tests",
            "docs",
        ]

        directories = {}
        for dir_name in important_dirs:
            dir_path = self.project_root / dir_name
            directories[dir_name] = {
                "exists": dir_path.exists(),
                "path": str(dir_path),
                "is_dir": dir_path.is_dir() if dir_path.exists() else False,
            }

            if dir_path.exists() and dir_path.is_dir():
                try:
                    file_count = len(list(dir_path.iterdir()))
                    directories[dir_name]["file_count"] = file_count
                except Exception:
                    directories[dir_name]["file_count"] = "unknown"

        return directories

    def _check_data_files(self) -> Dict[str, Any]:
        """Check important data files"""
        important_files = [
            "config.yaml",
            "requirements.txt",
            "ProjectP.py",
            "main.py",
            "README.md",
        ]

        files = {}
        for file_name in important_files:
            file_path = self.project_root / file_name
            files[file_name] = {"exists": file_path.exists(), "path": str(file_path)}

            if file_path.exists():
                try:
                    stat = file_path.stat()
                    files[file_name].update(
                        {
                            "size": self._format_bytes(stat.st_size),
                            "modified": datetime.fromtimestamp(stat.st_mtime).strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                        }
                    )
                except Exception:
                    pass

        return files

    def _format_bytes(self, bytes_value: int) -> str:
        """Format bytes to human readable format"""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"

    def print_health_report(self, health_data: Optional[Dict[str, Any]] = None) -> None:
        """Print comprehensive health report"""
        if health_data is None:
            health_data = self.health_data

        if not health_data:
            print(
                ColorThemes.error(
                    "No health data available. Run check_system_health() first."
                )
            )
            return

        print(ColorThemes.header("🏥 SYSTEM HEALTH REPORT"))
        print("=" * 80)

        # Basic system info
        print(
            ColorThemes.info(
                f"🐍 Python Version: {health_data.get('python_version', 'Unknown')}"
            )
        )
        print(
            ColorThemes.info(f"💻 Platform: {health_data.get('platform', 'Unknown')}")
        )
        print(
            ColorThemes.info(
                f"📁 Working Directory: {health_data.get('working_dir', 'Unknown')}"
            )
        )
        print(
            ColorThemes.info(f"⏰ Timestamp: {health_data.get('timestamp', 'Unknown')}")
        )
        print()

        # System resources
        if "system_resources" in health_data:
            self._print_system_resources(health_data["system_resources"])

        # Disk space
        if "disk_space" in health_data:
            self._print_disk_space(health_data["disk_space"])

        # Package status
        if "packages" in health_data:
            self._print_package_status(health_data["packages"])

        # Missing packages
        if health_data.get("missing_packages"):
            print(ColorThemes.error("❌ MISSING PACKAGES:"))
            for pkg in health_data["missing_packages"]:
                print(f"   • {pkg}")
            print()

        # Directories
        if "directories" in health_data:
            self._print_directory_status(health_data["directories"])

        print("=" * 80)

    def _print_system_resources(self, resources: Dict[str, Any]) -> None:
        """Print system resources information"""
        if resources.get("status") == "available":
            print(ColorThemes.header("💾 SYSTEM RESOURCES"))

            # CPU info
            cpu = resources.get("cpu", {})
            print(
                f"   🖥️ CPU Cores: {cpu.get('physical_cores', 'N/A')} physical, {cpu.get('logical_cores', 'N/A')} logical"
            )
            print(f"   📊 CPU Usage: {cpu.get('cpu_usage', 'N/A')}%")

            # Memory info
            memory = resources.get("memory", {})
            print(
                f"   🧠 Memory: {memory.get('used', 'N/A')} / {memory.get('total', 'N/A')} ({memory.get('percentage', 'N/A')}%)"
            )
            print(f"   💡 Available: {memory.get('available', 'N/A')}")
            print()
        else:
            print(ColorThemes.warning("⚠️ System resources information not available"))
            print()

    def _print_disk_space(self, disk: Dict[str, Any]) -> None:
        """Print disk space information"""
        if "error" not in disk:
            usage_percent = disk.get("usage_percent", 0)
            status_color = (
                Colors.BRIGHT_RED
                if usage_percent > 90
                else Colors.BRIGHT_YELLOW if usage_percent > 75 else Colors.BRIGHT_GREEN
            )

            print(ColorThemes.header("💽 DISK SPACE"))
            print(f"   📊 Usage: {colorize(f'{usage_percent:.1f}%', status_color)}")
            print(f"   📁 Total: {disk.get('total', 'N/A')}")
            print(f"   ✅ Free: {disk.get('free', 'N/A')}")
            print()
        else:
            print(ColorThemes.warning(f"⚠️ Disk space check failed: {disk['error']}"))
            print()

    def _print_package_status(self, packages: Dict[str, Dict[str, Any]]) -> None:
        """Print package status information"""
        print(ColorThemes.header("📦 PACKAGE STATUS"))

        for category, pkg_dict in packages.items():
            category_display = category.replace("_", " ").title()
            print(f"   📋 {category_display}:")

            for pkg, info in pkg_dict.items():
                status = info.get("status", "unknown")
                version = info.get("version", "N/A")

                if status == "installed":
                    print(f"      ✅ {pkg} ({version})")
                elif status == "missing":
                    print(f"      ❌ {pkg} - Missing")
                else:
                    print(f"      ⚠️ {pkg} - {status}")
            print()

    def _print_directory_status(self, directories: Dict[str, Any]) -> None:
        """Print directory status information"""
        print(ColorThemes.header("📁 DIRECTORY STATUS"))

        for dir_name, info in directories.items():
            exists = info.get("exists", False)
            file_count = info.get("file_count", "N/A")

            if exists:
                print(f"   ✅ {dir_name}/ ({file_count} items)")
            else:
                print(f"   ❌ {dir_name}/ - Missing")
        print()


# Global health monitor instance
health_monitor = SystemHealthMonitor()


def check_system_health() -> Dict[str, Any]:
    """Check system health (compatibility function)"""
    return health_monitor.check_system_health()


def print_health_report() -> None:
    """Print health report (compatibility function)"""
    health_data = health_monitor.check_system_health()
    health_monitor.print_health_report(health_data)
