import os
import shutil
from datetime import datetime
from typing import Any, Dict, List

import psutil

"""
Health Checker for Smart Monitoring
ตรวจสอบสุขภาพของระบบและโปรเจกต์
"""


class HealthChecker:
    """ตรวจสอบสุขภาพระบบ"""

    def __init__(self, project_root: str):
        self.project_root = project_root
        self.health_status = {}

    def check_system_health(self) -> Dict[str, Any]:
        """ตรวจสอบสุขภาพระบบทั้งหมด"""
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "unknown",
            "checks": {
                "file_system": self._check_file_system(),
                "project_structure": self._check_project_structure(),
                "dependencies": self._check_dependencies(),
                "performance": self._check_performance(),
            },
        }

        # Calculate overall status
        health_report["overall_status"] = self._calculate_overall_status(
            health_report["checks"]
        )

        self.health_status = health_report
        return health_report

    def _check_file_system(self) -> Dict[str, Any]:
        """ตรวจสอบ file system"""
        try:
            # Check if project root exists and is accessible
            accessible = os.path.exists(self.project_root) and os.access(
                self.project_root, os.R_OK
            )

            # Check disk space (simplified)
            free_space = "unknown"
            try:
                total, used, free = shutil.disk_usage(self.project_root)
                free_space = f"{free // (1024**3)} GB"
            except:
                pass

            return {
                "status": "healthy" if accessible else "unhealthy",
                "accessible": accessible,
                "free_space": free_space,
                "issues": [] if accessible else ["Project root not accessible"],
            }
        except Exception as e:
            return {
                "status": "error",
                "accessible": False,
                "error": str(e),
                "issues": [f"File system check failed: {e}"],
            }

    def _check_project_structure(self) -> Dict[str, Any]:
        """ตรวจสอบโครงสร้างโปรเจกต์"""
        required_files = [
            "ProjectP.py",
            "agent/agent_controller.py",
            "agent/agent_config.yaml",
        ]

        missing_files = []
        existing_files = []

        for file_path in required_files:
            full_path = os.path.join(self.project_root, file_path)
            if os.path.exists(full_path):
                existing_files.append(file_path)
            else:
                missing_files.append(file_path)

        status = (
            "healthy"
            if not missing_files
            else "warning" if len(missing_files) < len(required_files) else "unhealthy"
        )

        return {
            "status": status,
            "total_required": len(required_files),
            "existing": len(existing_files),
            "missing": len(missing_files),
            "missing_files": missing_files,
            "issues": [f"Missing files: {missing_files}"] if missing_files else [],
        }

    def _check_dependencies(self) -> Dict[str, Any]:
        """ตรวจสอบ dependencies"""
        try:
            # Check essential imports
            essential_modules = ["pandas", "numpy", "sklearn", "lightgbm"]
            available_modules = []
            missing_modules = []

            for module in essential_modules:
                try:
                    __import__(module)
                    available_modules.append(module)
                except ImportError:
                    missing_modules.append(module)

            status = (
                "healthy"
                if not missing_modules
                else (
                    "warning"
                    if len(missing_modules) < len(essential_modules)
                    else "unhealthy"
                )
            )

            return {
                "status": status,
                "available": available_modules,
                "missing": missing_modules,
                "issues": (
                    [f"Missing modules: {missing_modules}"] if missing_modules else []
                ),
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "issues": [f"Dependency check failed: {e}"],
            }

    def _check_performance(self) -> Dict[str, Any]:
        """ตรวจสอบประสิทธิภาพ"""
        try:

            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            # Simple performance assessment
            performance_issues = []

            if cpu_usage > 90:
                performance_issues.append(f"High CPU usage: {cpu_usage:.1f}%")

            if memory.percent > 85:
                performance_issues.append(f"High memory usage: {memory.percent:.1f}%")

            status = "healthy" if not performance_issues else "warning"

            return {
                "status": status,
                "cpu_usage": cpu_usage,
                "memory_usage": memory.percent,
                "available_memory_gb": memory.available / (1024**3),
                "issues": performance_issues,
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "issues": [f"Performance check failed: {e}"],
            }

    def _calculate_overall_status(self, checks: Dict[str, Dict[str, Any]]) -> str:
        """คำนวณสถานะโดยรวม"""
        statuses = [check.get("status", "unknown") for check in checks.values()]

        if "error" in statuses or "unhealthy" in statuses:
            return "unhealthy"
        elif "warning" in statuses:
            return "warning"
        elif all(status == "healthy" for status in statuses):
            return "healthy"
        else:
            return "unknown"

    def get_health_summary(self) -> Dict[str, Any]:
        """สรุปสุขภาพระบบ"""
        if not self.health_status:
            return {
                "status": "not_checked",
                "message": "Health check not performed yet",
            }

        overall_status = self.health_status.get("overall_status", "unknown")
        checks = self.health_status.get("checks", {})

        # Count issues
        total_issues = sum(len(check.get("issues", [])) for check in checks.values())

        return {
            "overall_status": overall_status,
            "total_issues": total_issues,
            "last_check": self.health_status.get("timestamp"),
            "components_checked": len(checks),
            "healthy_components": sum(
                1 for check in checks.values() if check.get("status") == "healthy"
            ),
        }

    def get_critical_issues(self) -> List[str]:
        """ดึงปัญหาที่ร้ายแรง"""
        if not self.health_status:
            return []

        critical_issues = []
        checks = self.health_status.get("checks", {})

        for component, check in checks.items():
            if check.get("status") in ["error", "unhealthy"]:
                issues = check.get("issues", [])
                for issue in issues:
                    critical_issues.append(f"[{component}] {issue}")

        return critical_issues
