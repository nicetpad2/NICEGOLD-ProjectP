# -*- coding: utf-8 -*-
"""
NICEGOLD ProjectP - API Endpoints
════════════════════════════════════════════════════════════════════════════════

Additional API endpoints and utilities.

Author: NICEGOLD Team
Version: 3.0
Created: 2025-01-05
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import from parent modules
sys.path.append(str(Path(__file__).parent.parent))
from core.colors import Colors, colorize


class APIEndpoints:
    """Additional API endpoints and utilities"""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()

    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        return {
            "project": {
                "name": "NICEGOLD ProjectP",
                "version": "3.0",
                "root": str(self.project_root),
            },
            "python": {"version": sys.version, "executable": sys.executable},
            "system": {"platform": sys.platform, "path": sys.path[:5]},  # First 5 paths
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        models_dir = self.project_root / "models"
        output_models = self.project_root / "output_default" / "models"

        model_files = []

        # Search for model files
        for models_path in [models_dir, output_models]:
            if models_path.exists():
                for pattern in ["*.joblib", "*.pkl", "*.model"]:
                    model_files.extend(models_path.glob(pattern))

        return {
            "total_models": len(model_files),
            "models": [
                {
                    "name": model.name,
                    "path": str(model),
                    "size": model.stat().st_size if model.exists() else 0,
                }
                for model in model_files[:10]  # Limit to 10 models
            ],
        }

    def get_data_info(self) -> Dict[str, Any]:
        """Get data files information"""
        data_dir = self.project_root / "datacsv"

        data_files = []
        total_size = 0

        if data_dir.exists():
            for csv_file in data_dir.glob("*.csv"):
                size = csv_file.stat().st_size
                total_size += size
                data_files.append(
                    {
                        "name": csv_file.name,
                        "path": str(csv_file),
                        "size": size,
                        "modified": csv_file.stat().st_mtime,
                    }
                )

        return {
            "total_files": len(data_files),
            "total_size": total_size,
            "data_directory": str(data_dir),
            "files": sorted(data_files, key=lambda x: x["modified"], reverse=True)[:5],
        }

    def get_logs_info(self) -> Dict[str, Any]:
        """Get logs information"""
        logs_dirs = [
            self.project_root / "logs",
            self.project_root / "output_default" / "logs",
        ]

        log_files = []

        for logs_dir in logs_dirs:
            if logs_dir.exists():
                for pattern in ["*.log", "*.txt"]:
                    log_files.extend(logs_dir.glob(pattern))

        return {
            "total_logs": len(log_files),
            "recent_logs": [
                {
                    "name": log.name,
                    "path": str(log),
                    "size": log.stat().st_size,
                    "modified": log.stat().st_mtime,
                }
                for log in sorted(
                    log_files, key=lambda x: x.stat().st_mtime, reverse=True
                )[:5]
            ],
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from results files"""
        output_dir = self.project_root / "output_default"

        # Look for performance/results files
        result_patterns = ["*results*.csv", "*performance*.csv", "*metrics*.csv"]
        result_files = []

        if output_dir.exists():
            for pattern in result_patterns:
                result_files.extend(output_dir.rglob(pattern))

        # Placeholder performance data
        return {
            "total_result_files": len(result_files),
            "latest_results": [
                {
                    "file": result.name,
                    "path": str(result),
                    "modified": result.stat().st_mtime,
                }
                for result in sorted(
                    result_files, key=lambda x: x.stat().st_mtime, reverse=True
                )[:3]
            ],
            "summary": {
                "status": "Available" if result_files else "No results",
                "last_run": result_files[0].stat().st_mtime if result_files else None,
            },
        }

    def create_status_endpoint_data(self) -> Dict[str, Any]:
        """Create comprehensive status data for API endpoints"""
        try:
            return {
                "status": "healthy",
                "timestamp": str(Path.cwd()),  # Using as timestamp placeholder
                "system": self.get_system_info(),
                "models": self.get_model_info(),
                "data": self.get_data_info(),
                "logs": self.get_logs_info(),
                "performance": self.get_performance_summary(),
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "timestamp": str(Path.cwd())}
