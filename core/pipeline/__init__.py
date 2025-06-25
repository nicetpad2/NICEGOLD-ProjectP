#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NICEGOLD ProjectP - Trading Pipeline Module
Enterprise-grade trading pipeline system
"""

from .backtester import Backtester
from .data_loader import DataLoader
from .data_validator import DataValidator
from .feature_engineer import FeatureEngineer
from .model_trainer import ModelTrainer
from .performance_analyzer import PerformanceAnalyzer
from .pipeline_orchestrator import PipelineOrchestrator

__all__ = [
    "DataLoader",
    "DataValidator",
    "FeatureEngineer",
    "ModelTrainer",
    "Backtester",
    "PerformanceAnalyzer",
    "PipelineOrchestrator",
]

__version__ = "2.0.0"
__author__ = "NICEGOLD Enterprise"
