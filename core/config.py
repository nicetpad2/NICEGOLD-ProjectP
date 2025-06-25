#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration manager for NICEGOLD ProjectP
Handles loading and validating configuration files
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigManager:
    """Configuration manager for NICEGOLD ProjectP"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = {}
        self._load_config()

    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f)
                print(f"✅ Configuration loaded from {self.config_path}")
            else:
                print(f"⚠️ Configuration file not found: {self.config_path}")
                self._create_default_config()
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
            self._create_default_config()

    def _create_default_config(self):
        """Create default configuration"""
        self.config = {
            "project": {
                "name": "NICEGOLD ProjectP",
                "version": "2.0.0",
                "description": "Professional AI Trading System",
            },
            "data": {
                "input_folder": "datacsv",
                "output_folder": "output_default",
                "models_folder": "models",
                "logs_folder": "logs",
            },
            "trading": {
                "initial_balance": 10000,
                "max_position_size": 0.1,
                "stop_loss": 0.02,
                "take_profit": 0.04,
            },
            "ml": {
                "models": ["RandomForest", "XGBoost", "LightGBM"],
                "test_size": 0.2,
                "cv_folds": 5,
                "random_state": 42,
            },
            "api": {"dashboard_port": 8501, "api_port": 8000, "host": "localhost"},
        }

        self._save_config()

    def _save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            print(f"💾 Configuration saved to {self.config_path}")
        except Exception as e:
            print(f"❌ Error saving configuration: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)"""
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """Set configuration value by key (supports dot notation)"""
        keys = key.split(".")
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value
        self._save_config()

    def get_folders(self) -> Dict[str, Path]:
        """Get all folder paths"""
        return {
            "input": Path(self.get("data.input_folder", "datacsv")),
            "output": Path(self.get("data.output_folder", "output_default")),
            "models": Path(self.get("data.models_folder", "models")),
            "logs": Path(self.get("data.logs_folder", "logs")),
        }

    def ensure_folders(self):
        """Create necessary folders if they don't exist"""
        folders = self.get_folders()
        for name, path in folders.items():
            path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Ensured folder exists: {path}")

    def validate_paths(self):
        """Validate and create necessary paths - alias for ensure_folders"""
        self.ensure_folders()


# Global configuration instance
config_manager = ConfigManager()


def get_config():
    """Get the global configuration manager instance"""
    return config_manager


def get_setting(key: str, default: Any = None) -> Any:
    """Get a configuration setting"""
    return config_manager.get(key, default)


def set_setting(key: str, value: Any) -> None:
    """Set a configuration setting"""
    config_manager.set(key, value)


def ensure_folders():
    """Ensure all necessary folders exist"""
    config_manager.ensure_folders()


def get_folders():
    """Get all folder paths"""
    return config_manager.get_folders()
