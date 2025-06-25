# -*- coding: utf-8 -*-
#!/usr/bin/env python3

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

from .advanced_logger import AdvancedTerminalLogger as AdvancedLogger
from .production_features import ProductionFeatureEngineer
from .production_ml_training import ProductionMLPipeline
from .robust_csv_manager import RobustCSVManager

# -*- coding: utf-8 -*-
#!/usr/bin/env python3


# Try to import project modules
try:
    from .advanced_logger import AdvancedTerminalLogger as AdvancedLogger
    from .production_features import ProductionFeatureEngineer
    from .production_ml_training import ProductionMLPipeline
    from .robust_csv_manager import RobustCSVManager
except ImportError:
    # Fallback imports when running as script
    from src.advanced_logger import AdvancedTerminalLogger as AdvancedLogger
    from src.production_features import ProductionFeatureEngineer
    from src.production_ml_training import ProductionMLPipeline
    from src.robust_csv_manager import RobustCSVManager

# Try to import enhanced logging functions
try:
    from enhanced_logging_functions import (
        create_summary_report,
        log_backtesting_results,
        log_cross_validation_results,
        log_dataframe_info,
        log_error,
        log_feature_importance,
        log_info,
        log_model_performance,
        log_pipeline_end,
        log_pipeline_start,
        log_section_end,
        log_section_start,
        log_success,
        log_warning,
    )
except ImportError:
    # Fallback logging functions if enhanced logging is not available
    def log_pipeline_start(name: str, **kwargs):
        print(f"🚀 Starting pipeline: {name}")

    def log_pipeline_end(name: str, **kwargs):
        print(f"✅ Pipeline completed: {name}")

    def log_section_start(name: str, **kwargs):
        print(f"📋 Starting section: {name}")

    def log_section_end(name: str, **kwargs):
        print(f"✅ Section completed: {name}")

    def log_dataframe_info(*args, **kwargs):
        pass

    def log_model_performance(*args, **kwargs):
        pass

    def log_feature_importance(*args, **kwargs):
        pass

    def log_cross_validation_results(*args, **kwargs):
        pass

    def log_backtesting_results(*args, **kwargs):
        pass

    def log_error(*args, **kwargs):
        pass

    def log_warning(*args, **kwargs):
        pass

    def log_success(*args, **kwargs):
        pass

    def log_info(*args, **kwargs):
        pass

    def create_summary_report(*args, **kwargs):
        pass


"""
NICEGOLD ProjectP - Production Pipeline Integration
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

Complete, production - ready pipeline orchestrator that integrates:
- Advanced logging and progress tracking
- Robust CSV data management
- Production feature engineering
- Production ML training and validation
- Comprehensive error handling and monitoring

Author: NICEGOLD ProjectP Team
"""


# Import our production modules
try:
    from .advanced_logger import AdvancedTerminalLogger as AdvancedLogger
    from .production_features import ProductionFeatureEngineer
    from .production_ml_training import ProductionMLPipeline
    from .robust_csv_manager import RobustCSVManager
except ImportError:
    # Handle relative import issues when running as script
    sys.path.append(str(Path(__file__).parent.parent))
    from src.advanced_logger import AdvancedTerminalLogger as AdvancedLogger
    from src.production_features import ProductionFeatureEngineer
    from src.production_ml_training import ProductionMLPipeline
    from src.robust_csv_manager import RobustCSVManager

# Import enhanced logging functions
try:
    from enhanced_logging_functions import (
        log_critical,
        log_error,
        log_info,
        log_pipeline_end,
        log_pipeline_start,
        log_progress,
        log_section_end,
        log_section_start,
        log_success,
        log_warning,
    )
except ImportError:
    # Fallback logging if enhanced functions not available
    logger = logging.getLogger(__name__)

    def log_pipeline_start(name: str, **kwargs):
        logger.info(f"Starting pipeline: {name}")

    def log_pipeline_end(name: str, **kwargs):
        logger.info(f"Completed pipeline: {name}")

    def log_section_start(name: str, **kwargs):
        logger.info(f"Starting section: {name}")

    def log_section_end(name: str, **kwargs):
        logger.info(f"Completed section: {name}")

    def log_error(msg: str, **kwargs):
        logger.error(msg)

    def log_warning(msg: str, **kwargs):
        logger.warning(msg)

    def log_success(msg: str, **kwargs):
        logger.info(f"✅ {msg}")

    def log_info(msg: str, **kwargs):
        logger.info(msg)

    def log_critical(msg: str, **kwargs):
        logger.critical(msg)

    def log_progress(msg: str, **kwargs):
        logger.info(msg)

    def log_section_start(name: str, **kwargs):
        logger.info(f"Starting section: {name}")

    def log_section_end(name: str, **kwargs):
        logger.info(f"Completed section: {name}")

    def log_progress(value: float, **kwargs):
        logger.info(f"Progress: {value:.1%}")

    def log_error(msg: str, **kwargs):
        logger.error(msg)

    def log_warning(msg: str, **kwargs):
        logger.warning(msg)

    def log_critical(msg: str, **kwargs):
        logger.critical(msg)

    def log_success(msg: str, **kwargs):
        logger.info(f"✓ {msg}")

    def log_info(msg: str, **kwargs):
        logger.info(msg)


class ProductionPipeline:
    """
    Complete production pipeline orchestrator.

    Handles the full end - to - end pipeline:
    1. Data validation and loading
    2. Feature engineering
    3. Model training and validation
    4. Evaluation and reporting
    5. Model persistence and deployment readiness
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the production pipeline."""
        self.config = config or {}
        self.csv_manager = RobustCSVManager()
        self.feature_engineer = ProductionFeatureEngineer()
        self.ml_pipeline = ProductionMLPipeline()
        self.logger = AdvancedLogger("PRODUCTION_PIPELINE")

        # Pipeline state
        self.data_loaded = False
        self.features_created = False
        self.model_trained = False
        self.results = {}

        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.features_data = None
        self.model_results = None

    def run_full_pipeline(
        self, data_path: str, output_dir: str = "output", test_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Execute the complete production pipeline.

        Args:
            data_path: Path to the CSV data file
            output_dir: Directory for saving outputs
            test_mode: If True, use smaller data subset for testing

        Returns:
            Dictionary containing pipeline results and metrics
        """
        pipeline_start_time = time.time()

        try:
            log_pipeline_start(
                "NICEGOLD Production Pipeline",
                data_path=data_path,
                output_dir=output_dir,
                test_mode=test_mode,
            )

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Step 1: Data Loading and Validation
            self._run_data_loading(data_path, test_mode)

            # Step 2: Feature Engineering
            self._run_feature_engineering()

            # Step 3: Model Training and Validation
            self._run_model_training(output_dir)

            # Step 4: Final Evaluation and Reporting
            self._run_final_evaluation(output_dir)

            # Calculate total pipeline time
            total_time = time.time() - pipeline_start_time

            # Compile final results
            final_results = {
                "pipeline_status": "SUCCESS",
                "total_execution_time": total_time,
                "data_shape": (
                    self.raw_data.shape if self.raw_data is not None else None
                ),
                "features_created": (
                    len(self.features_data.columns)
                    if self.features_data is not None
                    else 0
                ),
                "model_metrics": (
                    self.model_results.get("test_metrics", {})
                    if self.model_results
                    else {}
                ),
                "output_directory": output_dir,
                "timestamp": time.strftime("%Y - %m - %d %H:%M:%S"),
            }

            log_pipeline_end(
                "NICEGOLD Production Pipeline",
                execution_time=f"{total_time:.2f}s",
                status="SUCCESS",
            )

            return final_results

        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            log_critical(error_msg, exception=e)
            self.logger.critical(error_msg, details=traceback.format_exc())

            return {
                "pipeline_status": "FAILED",
                "error_message": error_msg,
                "total_execution_time": time.time() - pipeline_start_time,
                "timestamp": time.strftime("%Y - %m - %d %H:%M:%S"),
            }

    def _run_data_loading(self, data_path: str, test_mode: bool = False) -> None:
        """Load and validate data."""
        log_section_start("Data Loading & Validation")

        try:
            # Load and validate CSV file using validate_and_standardize_csv
            log_info(f"Loading and validating CSV file: {data_path}")
            self.raw_data = self.csv_manager.validate_and_standardize_csv(data_path)

            if test_mode and len(self.raw_data) > 10000:
                log_warning("Test mode: Using subset of data (10, 000 records)")
                self.raw_data = self.raw_data.head(10000)

            log_success(f"Data loaded successfully: {self.raw_data.shape}")
            log_info(f"Columns: {list(self.raw_data.columns)}")

            # Check if we have time index
            if hasattr(self.raw_data.index, "min"):
                log_info(
                    f"Date range: {self.raw_data.index.min()} to {self.raw_data.index.max()}"
                )

            self.data_loaded = True
            log_progress(0.25, message="Data loading completed")

        except Exception as e:
            log_error(f"Data loading failed: {str(e)}")
            raise

        log_section_end("Data Loading & Validation")

    def _run_feature_engineering(self) -> None:
        """Create features for modeling."""
        log_section_start("Feature Engineering")

        try:
            if not self.data_loaded:
                raise ValueError("Data must be loaded before feature engineering")

            log_info("Starting feature engineering...")

            # Create features using production feature engineer
            self.features_data = self.feature_engineer.engineer_features(self.raw_data)

            log_success(f"Features created: {len(self.features_data.columns)} columns")
            log_info(f"Feature data shape: {self.features_data.shape}")

            # Log feature categories
            feature_counts = self.feature_engineer.get_feature_summary()
            for category, count in feature_counts.items():
                log_info(f"  {category}: {count} features")

            self.features_created = True
            log_progress(0.50, message="Feature engineering completed")

        except Exception as e:
            log_error(f"Feature engineering failed: {str(e)}")
            raise

        log_section_end("Feature Engineering")

    def _run_model_training(self, output_dir: str) -> None:
        """Train and validate models."""
        log_section_start("Model Training & Validation")

        try:
            if not self.features_created:
                raise ValueError("Features must be created before model training")

            log_info("Starting model training pipeline...")

            # Prepare target variable (assuming 'target' column exists or create one)
            if "target" not in self.features_data.columns:
                log_warning("No 'target' column found, creating price direction target")
                if "Close" in self.features_data.columns:
                    target_col = "Close"
                elif "close" in self.features_data.columns:
                    target_col = "close"
                else:
                    # Use first price - like column
                    price_cols = [
                        col
                        for col in self.features_data.columns
                        if any(word in col.lower() for word in ["close", "price"])
                    ]
                    if price_cols:
                        target_col = price_cols[0]
                    else:
                        raise ValueError(
                            "Cannot find suitable price column for target creation"
                        )

                self.features_data["target"] = (
                    self.features_data[target_col].shift(-1)
                    > self.features_data[target_col]
                ).astype(int)

            # Separate features and target
            X = self.features_data.drop(["target"], axis=1)
            y = self.features_data["target"]

            # Train models
            self.model_results = self.ml_pipeline.train_ensemble(X, y)

            # Log training results
            if "training_metrics" in self.model_results:
                metrics = self.model_results["training_metrics"]
                log_success(
                    f"Model training completed - Accuracy: {metrics.get('accuracy', 'N/A'):.3f}"
                )
                log_info(f"  Precision: {metrics.get('precision', 'N/A'):.3f}")
                log_info(f"  Recall: {metrics.get('recall', 'N/A'):.3f}")
                log_info(f"  F1 - Score: {metrics.get('f1', 'N/A'):.3f}")

            self.model_trained = True
            log_progress(0.75, message="Model training completed")

        except Exception as e:
            log_error(f"Model training failed: {str(e)}")
            raise

        log_section_end("Model Training & Validation")

    def _run_final_evaluation(self, output_dir: str) -> None:
        """Final evaluation and reporting."""
        log_section_start("Final Evaluation & Reporting")

        try:
            if not self.model_trained:
                raise ValueError("Model must be trained before final evaluation")

            log_info("Generating final evaluation report...")

            # Save pipeline summary
            summary_path = os.path.join(output_dir, "pipeline_summary.json")
            summary = {
                "data_info": {
                    "original_shape": self.raw_data.shape,
                    "features_shape": self.features_data.shape,
                    "date_range": {
                        "start": (
                            str(self.raw_data.index.min())
                            if hasattr(self.raw_data.index, "min")
                            else "N/A"
                        ),
                        "end": (
                            str(self.raw_data.index.max())
                            if hasattr(self.raw_data.index, "max")
                            else "N/A"
                        ),
                    },
                },
                "feature_info": self.feature_engineer.get_feature_summary(),
                "model_info": self.model_results,
                "pipeline_timestamp": time.strftime("%Y - %m - %d %H:%M:%S"),
            }

            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)

            log_success(f"Pipeline summary saved: {summary_path}")

            # Save processed data
            data_path = os.path.join(output_dir, "processed_data.csv")
            self.features_data.to_csv(data_path, index=False)
            log_success(f"Processed data saved: {data_path}")

            log_progress(1.0, message="Pipeline completed successfully")

        except Exception as e:
            log_error(f"Final evaluation failed: {str(e)}")
            raise

        log_section_end("Final Evaluation & Reporting")

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            "data_loaded": self.data_loaded,
            "features_created": self.features_created,
            "model_trained": self.model_trained,
            "data_shape": self.raw_data.shape if self.raw_data is not None else None,
            "features_count": (
                len(self.features_data.columns) if self.features_data is not None else 0
            ),
        }


def run_production_pipeline(
    data_path: str,
    output_dir: str = "output",
    config: Optional[Dict[str, Any]] = None,
    test_mode: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function to run the complete production pipeline.

    Args:
        data_path: Path to the CSV data file
        output_dir: Directory for saving outputs
        config: Optional pipeline configuration
        test_mode: If True, use smaller data subset for testing

    Returns:
        Dictionary containing pipeline results and metrics
    """
    pipeline = ProductionPipeline(config)
    return pipeline.run_full_pipeline(data_path, output_dir, test_mode)


if __name__ == "__main__":
    # Demo/test execution

    parser = argparse.ArgumentParser(description="NICEGOLD Production Pipeline")
    parser.add_argument(" -  - data", required=True, help="Path to CSV data file")
    parser.add_argument(" -  - output", default="output", help="Output directory")
    parser.add_argument(
        " -  - test", action="store_true", help="Test mode with smaller dataset"
    )

    args = parser.parse_args()

    results = run_production_pipeline(
        data_path=args.data, output_dir=args.output, test_mode=args.test
    )

    print("\n" + " = " * 60)
    print("PIPELINE EXECUTION COMPLETED")
    print(" = " * 60)
    print(f"Status: {results['pipeline_status']}")
    if results["pipeline_status"] == "SUCCESS":
        print(f"Execution Time: {results['total_execution_time']:.2f}s")
        print(f"Data Shape: {results['data_shape']}")
        print(f"Features Created: {results['features_created']}")
        print(f"Output Directory: {results['output_directory']}")
    else:
        print(f"Error: {results.get('error_message', 'Unknown error')}")
    print(" = " * 60)
