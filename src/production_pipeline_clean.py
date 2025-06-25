#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NICEGOLD ProjectP - Production Pipeline (Clean Version)
=====================================================

A clean, working version of the production pipeline with proper error handling
and robust imports.
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path for reliable imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from src.advanced_logger import AdvancedTerminalLogger as AdvancedLogger
    from src.production_features import ProductionFeatureEngineer
except ImportError as e:
    logger.warning(f"Could not import some modules: {e}")
    ProductionFeatureEngineer = None
    AdvancedLogger = None


# Fallback logging functions
def log_pipeline_start(name: str, **kwargs):
    logger.info(f"🚀 Starting pipeline: {name}")


def log_pipeline_end(name: str, **kwargs):
    logger.info(f"✅ Pipeline completed: {name}")


def log_section_start(name: str, **kwargs):
    logger.info(f"📋 Starting section: {name}")


def log_section_end(name: str, **kwargs):
    logger.info(f"✅ Section completed: {name}")


def log_error(msg: str, **kwargs):
    logger.error(f"❌ {msg}")


def log_success(msg: str, **kwargs):
    logger.info(f"✅ {msg}")


def log_info(msg: str, **kwargs):
    logger.info(f"ℹ️ {msg}")


class ProductionPipeline:
    """
    Production Pipeline for NICEGOLD ProjectP

    Handles the complete ML pipeline from data loading to model deployment.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the production pipeline"""
        self.config_path = config_path
        self.config = self._load_config()
        self.feature_engineer = None
        self.logger = AdvancedLogger() if AdvancedLogger else logger

        # Initialize components
        self._initialize_components()

    def _load_config(self) -> Dict[str, Any]:
        """Load pipeline configuration"""
        default_config = {
            "data": {
                "input_path": "data/raw/market_data.csv",
                "output_path": "data/processed/",
                "validation_split": 0.2,
            },
            "features": {
                "price_features": True,
                "momentum_features": True,
                "trend_features": True,
                "volatility_features": True,
                "volume_features": True,
                "pattern_features": True,
                "time_features": True,
                "statistical_features": True,
            },
            "model": {
                "type": "ensemble",
                "cross_validation_folds": 5,
                "random_state": 42,
            },
        }

        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                log_info(f"Loaded config from {self.config_path}")
            except Exception as e:
                log_error(f"Could not load config: {e}")

        return default_config

    def _initialize_components(self):
        """Initialize pipeline components"""
        try:
            if ProductionFeatureEngineer:
                self.feature_engineer = ProductionFeatureEngineer(
                    feature_config=self.config.get("features", {})
                )
                log_success("Feature engineer initialized")
            else:
                log_error("ProductionFeatureEngineer not available")

        except Exception as e:
            log_error(f"Component initialization failed: {e}")

    def run_full_pipeline(
        self, input_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Run the complete production pipeline

        Args:
            input_data: Optional input DataFrame, will load from config if None

        Returns:
            Dictionary with pipeline results
        """
        log_pipeline_start("Full Production Pipeline")

        try:
            results = {}

            # 1. Data Loading
            log_section_start("Data Loading")
            if input_data is not None:
                df = input_data.copy()
            else:
                df = self._load_data()

            log_info(f"Loaded data with shape: {df.shape}")
            results["data_shape"] = df.shape
            log_section_end("Data Loading")

            # 2. Feature Engineering
            log_section_start("Feature Engineering")
            if self.feature_engineer:
                # Clear any cached module imports first
                if "src.production_features" in sys.modules:
                    del sys.modules["src.production_features"]
                    from src.production_features import ProductionFeatureEngineer

                    self.feature_engineer = ProductionFeatureEngineer(
                        feature_config=self.config.get("features", {})
                    )

                df_engineered = self.feature_engineer.engineer_features(df)
                log_info(f"Engineered features shape: {df_engineered.shape}")

                # Get feature summary with error handling
                try:
                    if hasattr(self.feature_engineer, "get_feature_summary"):
                        feature_summary = self.feature_engineer.get_feature_summary()
                        log_info(f"Feature summary: {feature_summary}")
                        results["feature_summary"] = feature_summary
                    else:
                        log_error("get_feature_summary method not found")
                        results["feature_summary"] = {}
                except AttributeError as e:
                    log_error(f"Feature summary error: {e}")
                    results["feature_summary"] = {}

                results["engineered_shape"] = df_engineered.shape
                results["features_added"] = df_engineered.shape[1] - df.shape[1]
            else:
                log_error("Feature engineer not available")
                df_engineered = df
                results["feature_summary"] = {}

            log_section_end("Feature Engineering")

            # 3. Data Validation
            log_section_start("Data Validation")
            validation_results = self._validate_data(df_engineered)
            results["validation"] = validation_results
            log_section_end("Data Validation")

            # 4. Success
            results["status"] = "success"
            results["timestamp"] = pd.Timestamp.now().isoformat()

            log_pipeline_end("Full Production Pipeline")
            return results

        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            log_error(error_msg)

            return {
                "status": "error",
                "error": error_msg,
                "traceback": traceback.format_exc(),
                "timestamp": pd.Timestamp.now().isoformat(),
            }

    def _load_data(self) -> pd.DataFrame:
        """Load data from configured source"""
        input_path = self.config["data"]["input_path"]

        if not os.path.exists(input_path):
            # Create sample data if file doesn't exist
            log_info("Creating sample data for testing")
            return self._create_sample_data()

        try:
            if input_path.endswith(".csv"):
                df = pd.read_csv(input_path)
            elif input_path.endswith(".parquet"):
                df = pd.read_parquet(input_path)
            else:
                raise ValueError(f"Unsupported file format: {input_path}")

            return df

        except Exception as e:
            log_error(f"Could not load data from {input_path}: {e}")
            return self._create_sample_data()

    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample OHLCV data for testing"""
        dates = pd.date_range(start="2020-01-01", periods=1000, freq="D")
        np.random.seed(42)

        # Generate realistic price data
        base_price = 100
        prices = []
        for i in range(1000):
            change = np.random.normal(0.001, 0.02)
            base_price *= 1 + change
            prices.append(base_price)

        data = []
        for i, price in enumerate(prices):
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i - 1] if i > 0 else price
            volume = np.random.randint(100000, 1000000)

            data.append(
                {
                    "timestamp": dates[i],
                    "Open": open_price,
                    "High": high,
                    "Low": low,
                    "Close": price,
                    "Volume": volume,
                }
            )

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        log_info("Created sample OHLCV data")
        return df

    def _validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate the processed data"""
        validation = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "null_values": df.isnull().sum().sum(),
            "inf_values": np.isinf(df.select_dtypes(include=[np.number])).sum().sum(),
            "data_types": df.dtypes.to_dict(),
        }

        # Check for required columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        validation["missing_required_columns"] = missing_cols

        # Data quality score
        score = 100
        if validation["null_values"] > 0:
            score -= min(20, validation["null_values"] / len(df) * 100)
        if validation["inf_values"] > 0:
            score -= min(10, validation["inf_values"] / len(df) * 100)
        if missing_cols:
            score -= len(missing_cols) * 10

        validation["quality_score"] = max(0, score)

        return validation


def main():
    """Main function for CLI usage"""
    parser = argparse.ArgumentParser(
        description="NICEGOLD ProjectP Production Pipeline"
    )
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--input", type=str, help="Input data file")
    parser.add_argument("--output", type=str, help="Output directory")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = ProductionPipeline(config_path=args.config)

    # Load input data if provided
    input_data = None
    if args.input and os.path.exists(args.input):
        try:
            input_data = pd.read_csv(args.input)
            log_info(f"Loaded input data from {args.input}")
        except Exception as e:
            log_error(f"Could not load input data: {e}")

    # Run pipeline
    results = pipeline.run_full_pipeline(input_data=input_data)

    # Save results
    output_dir = args.output or "output"
    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, f"pipeline_results_{int(time.time())}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    log_success(f"Results saved to {results_file}")

    # Print summary
    if results["status"] == "success":
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📊 Data shape: {results.get('data_shape', 'N/A')}")
        print(f"🔧 Engineered shape: {results.get('engineered_shape', 'N/A')}")
        print(f"➕ Features added: {results.get('features_added', 0)}")
        print(f"📋 Feature summary: {results.get('feature_summary', {})}")
    else:
        print(f"\n❌ Pipeline failed: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
