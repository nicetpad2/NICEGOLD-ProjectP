#!/usr/bin/env python3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import numpy as np
import os
import pandas as pd
import pickle
import warnings
"""
🔍 ADVANCED VALIDATION SYSTEM
Comprehensive validation framework for NICEGOLD Pipeline
"""


@dataclass
class ValidationResult:
    """Result of a validation check"""

    passed: bool
    message: str
    score: Optional[float] = None
    details: Optional[Dict] = None
    level: str = "INFO"  # INFO, WARNING, ERROR, CRITICAL


class DataValidation:
    """Advanced data validation for market data"""

    def __init__(self):
        self.min_rows = 1000
        self.required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        self.optional_columns = ["tick_volume", "spread", "real_volume"]

    def validate_basic_structure(self, df: pd.DataFrame) -> ValidationResult:
        """Validate basic data structure"""
        try:
            # Check if DataFrame is not empty
            if df.empty:
                return ValidationResult(False, "DataFrame is empty", level = "CRITICAL")

            # Check minimum rows
            if len(df) < self.min_rows:
                return ValidationResult(
                    False, 
                    f"Insufficient data: {len(df)} rows < {self.min_rows} required", 
                    level = "ERROR", 
                )

            # Check required columns
            missing_cols = []
            df_cols_lower = [col.lower() for col in df.columns]

            for required_col in self.required_columns:
                if required_col.lower() not in df_cols_lower:
                    missing_cols.append(required_col)

            if missing_cols:
                return ValidationResult(
                    False, f"Missing required columns: {missing_cols}", level = "CRITICAL"
                )

            return ValidationResult(
                True, f"Basic structure OK: {len(df):, } rows, {len(df.columns)} columns"
            )

        except Exception as e:
            return ValidationResult(
                False, f"Structure validation error: {str(e)}", level = "ERROR"
            )

    def validate_data_quality(self, df: pd.DataFrame) -> ValidationResult:
        """Validate data quality and consistency"""
        try:
            issues = []
            score = 100.0

            # Check for NaN values
            nan_counts = df.isnull().sum()
            total_cells = len(df) * len(df.columns)
            nan_percentage = (nan_counts.sum() / total_cells) * 100

            if nan_percentage > 10:
                issues.append(f"High NaN percentage: {nan_percentage:.1f}%")
                score -= 20
            elif nan_percentage > 5:
                issues.append(f"Moderate NaN percentage: {nan_percentage:.1f}%")
                score -= 10

            # Check for duplicate timestamps
            if "timestamp" in df.columns:
                duplicate_timestamps = df["timestamp"].duplicated().sum()
                if duplicate_timestamps > 0:
                    issues.append(f"Duplicate timestamps: {duplicate_timestamps}")
                    score -= 15

            # Check price data consistency
            price_cols = ["open", "high", "low", "close"]
            available_price_cols = [col for col in price_cols if col in df.columns]

            if len(available_price_cols) >= 4:
                # Check OHLC logic
                invalid_ohlc = 0
                for idx in df.index:
                    try:
                        o, h, l, c = df.loc[idx, ["open", "high", "low", "close"]]
                        if pd.isna([o, h, l, c]).any():
                            continue
                        if not (l <= o <= h and l <= c <= h):
                            invalid_ohlc += 1
                    except:
                        continue

                if invalid_ohlc > 0:
                    invalid_percentage = (invalid_ohlc / len(df)) * 100
                    issues.append(
                        f"Invalid OHLC relationships: {invalid_percentage:.1f}%"
                    )
                    score -= min(30, invalid_percentage * 2)

            # Check for outliers in price data
            for col in available_price_cols:
                if col in df.columns:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    outliers = (
                        (df[col] < (q1 - 3 * iqr)) | (df[col] > (q3 + 3 * iqr))
                    ).sum()
                    outlier_percentage = (outliers / len(df)) * 100

                    if outlier_percentage > 1:
                        issues.append(
                            f"High outliers in {col}: {outlier_percentage:.1f}%"
                        )
                        score -= 5

            # Check volume data
            if "volume" in df.columns:
                zero_volume = (df["volume"] == 0).sum()
                zero_volume_percentage = (zero_volume / len(df)) * 100

                if zero_volume_percentage > 20:
                    issues.append(
                        f"High zero volume percentage: {zero_volume_percentage:.1f}%"
                    )
                    score -= 10

            level = "INFO"
            if score < 70:
                level = "ERROR"
            elif score < 85:
                level = "WARNING"

            message = f"Data quality score: {score:.1f}/100"
            if issues:
                message += f" - Issues: {'; '.join(issues)}"

            return ValidationResult(
                score >= 70, 
                message, 
                score = score, 
                details = {"issues": issues}, 
                level = level, 
            )

        except Exception as e:
            return ValidationResult(
                False, f"Quality validation error: {str(e)}", level = "ERROR"
            )

    def validate_time_series(self, df: pd.DataFrame) -> ValidationResult:
        """Validate time series properties"""
        try:
            if "timestamp" not in df.columns:
                return ValidationResult(
                    False, "No timestamp column found", level = "WARNING"
                )

            # Convert timestamp to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                try:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                except:
                    return ValidationResult(
                        False, "Cannot parse timestamp column", level = "ERROR"
                    )

            # Check time ordering
            is_sorted = df["timestamp"].is_monotonic_increasing
            if not is_sorted:
                return ValidationResult(
                    False, "Timestamps are not in chronological order", level = "WARNING"
                )

            # Check time gaps
            time_diffs = df["timestamp"].diff().dropna()
            most_common_interval = time_diffs.mode()

            if len(most_common_interval) > 0:
                expected_interval = most_common_interval.iloc[0]
                large_gaps = (
                    time_diffs > expected_interval * 3
                )  # Gaps 3x larger than expected
                gap_count = large_gaps.sum()

                if gap_count > 0:
                    gap_percentage = (gap_count / len(time_diffs)) * 100
                    level = "WARNING" if gap_percentage < 5 else "ERROR"
                    return ValidationResult(
                        gap_percentage < 10, 
                        f"Time series gaps detected: {gap_count} gaps ({gap_percentage:.1f}%)", 
                        level = level, 
                    )

            # Calculate data coverage
            total_span = df["timestamp"].max() - df["timestamp"].min()
            expected_points = (
                total_span / time_diffs.median() if len(time_diffs) > 0 else len(df)
            )
            coverage = len(df) / expected_points * 100 if expected_points > 0 else 100

            return ValidationResult(
                True, 
                f"Time series OK: {coverage:.1f}% coverage, {len(df):, } points over {total_span.days} days", 
            )

        except Exception as e:
            return ValidationResult(
                False, f"Time series validation error: {str(e)}", level = "ERROR"
            )


class ModelValidation:
    """Validation for machine learning models and predictions"""

    def validate_model_file(self, model_path: str) -> ValidationResult:
        """Validate model file integrity"""
        try:
            if not os.path.exists(model_path):
                return ValidationResult(
                    False, f"Model file not found: {model_path}", level = "CRITICAL"
                )

            # Check file size
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            if file_size < 0.1:
                return ValidationResult(
                    False, f"Model file too small: {file_size:.2f}MB", level = "WARNING"
                )

            # Try to load the model
            try:
                if model_path.endswith(".pkl"):
                    with open(model_path, "rb") as f:
                        model = pickle.load(f)

                    # Check if model has required methods
                    required_methods = ["predict", "predict_proba"]
                    missing_methods = [
                        method
                        for method in required_methods
                        if not hasattr(model, method)
                    ]

                    if missing_methods:
                        return ValidationResult(
                            False, 
                            f"Model missing methods: {missing_methods}", 
                            level = "ERROR", 
                        )

                elif model_path.endswith(".json"):
                    with open(model_path, "r") as f:
                        model_info = json.load(f)

                    required_keys = ["model_type", "features", "performance"]
                    missing_keys = [
                        key for key in required_keys if key not in model_info
                    ]

                    if missing_keys:
                        return ValidationResult(
                            False, 
                            f"Model info missing keys: {missing_keys}", 
                            level = "WARNING", 
                        )

                return ValidationResult(True, f"Model file valid: {file_size:.2f}MB")

            except Exception as e:
                return ValidationResult(
                    False, f"Cannot load model: {str(e)}", level = "ERROR"
                )

        except Exception as e:
            return ValidationResult(
                False, f"Model validation error: {str(e)}", level = "ERROR"
            )

    def validate_predictions(
        self, predictions: Union[np.ndarray, pd.Series, List]
    ) -> ValidationResult:
        """Validate prediction outputs"""
        try:
            if isinstance(predictions, (pd.Series, list)):
                predictions = np.array(predictions)

            if len(predictions) == 0:
                return ValidationResult(False, "No predictions found", level = "CRITICAL")

            # Check for invalid values
            invalid_count = np.sum(np.isnan(predictions) | np.isinf(predictions))
            if invalid_count > 0:
                invalid_percentage = (invalid_count / len(predictions)) * 100
                level = "ERROR" if invalid_percentage > 5 else "WARNING"
                return ValidationResult(
                    invalid_percentage <= 5, 
                    f"Invalid predictions: {invalid_count} ({invalid_percentage:.1f}%)", 
                    level = level, 
                )

            # Check prediction range (assuming binary classification)
            if np.all((predictions >= 0) & (predictions <= 1)):
                # Looks like probabilities
                prediction_type = "probabilities"

                # Check distribution
                very_certain = np.sum((predictions < 0.1) | (predictions > 0.9))
                very_certain_percentage = (very_certain / len(predictions)) * 100

                details = {
                    "type": prediction_type, 
                    "mean": float(np.mean(predictions)), 
                    "std": float(np.std(predictions)), 
                    "very_certain_percentage": very_certain_percentage, 
                }

            else:
                # Looks like class predictions
                prediction_type = "classes"
                unique_values = np.unique(predictions)

                details = {
                    "type": prediction_type, 
                    "unique_values": unique_values.tolist(), 
                    "class_distribution": {
                        str(val): int(np.sum(predictions == val))
                        for val in unique_values
                    }, 
                }

            return ValidationResult(
                True, 
                f"Predictions valid: {len(predictions):, } {prediction_type}", 
                details = details, 
            )

        except Exception as e:
            return ValidationResult(
                False, f"Prediction validation error: {str(e)}", level = "ERROR"
            )

    def validate_performance_metrics(self, metrics: Dict) -> ValidationResult:
        """Validate model performance metrics"""
        try:
            required_metrics = ["auc", "accuracy"]
            missing_metrics = [
                metric for metric in required_metrics if metric not in metrics
            ]

            if missing_metrics:
                return ValidationResult(
                    False, 
                    f"Missing performance metrics: {missing_metrics}", 
                    level = "WARNING", 
                )

            issues = []
            score = 100.0

            # Check AUC
            auc = metrics.get("auc", 0)
            if auc < 0.5:
                issues.append(f"Poor AUC: {auc:.3f} < 0.5 (worse than random)")
                score = 0
            elif auc < 0.6:
                issues.append(f"Low AUC: {auc:.3f} < 0.6")
                score = 30
            elif auc < 0.7:
                issues.append(f"Moderate AUC: {auc:.3f} < 0.7")
                score = 60
            elif auc >= 0.8:
                score = 100
            else:
                score = 80

            # Check accuracy
            accuracy = metrics.get("accuracy", 0)
            if accuracy < 0.5:
                issues.append(f"Poor accuracy: {accuracy:.3f} < 0.5")
                score = min(score, 20)
            elif accuracy < 0.6:
                issues.append(f"Low accuracy: {accuracy:.3f} < 0.6")
                score = min(score, 40)

            level = "INFO"
            if score < 30:
                level = "CRITICAL"
            elif score < 60:
                level = "ERROR"
            elif score < 80:
                level = "WARNING"

            message = f"Performance metrics: AUC = {auc:.3f}, Accuracy = {accuracy:.3f}"
            if issues:
                message += f" - Issues: {'; '.join(issues)}"

            return ValidationResult(
                score >= 30,  # Minimum acceptable score
                message, 
                score = score, 
                details = metrics, 
                level = level, 
            )

        except Exception as e:
            return ValidationResult(
                False, f"Metrics validation error: {str(e)}", level = "ERROR"
            )


class PipelineValidation:
    """End - to - end pipeline validation"""

    def __init__(self):
        self.data_validator = DataValidation()
        self.model_validator = ModelValidation()

    def validate_pipeline_stage(
        self, stage_name: str, stage_outputs: Dict[str, str]
    ) -> List[ValidationResult]:
        """Validate outputs of a specific pipeline stage"""
        results = []

        try:
            if stage_name.lower() == "preprocess":
                # Validate preprocessed data
                for output_name, output_path in stage_outputs.items():
                    if output_path.endswith(".parquet"):
                        try:
                            df = pd.read_parquet(output_path)
                            results.append(
                                self.data_validator.validate_basic_structure(df)
                            )
                            results.append(
                                self.data_validator.validate_data_quality(df)
                            )
                            results.append(self.data_validator.validate_time_series(df))
                        except Exception as e:
                            results.append(
                                ValidationResult(
                                    False, 
                                    f"Cannot load {output_name}: {str(e)}", 
                                    level = "ERROR", 
                                )
                            )

            elif stage_name.lower() in ["train", "walkforward"]:
                # Validate model outputs
                for output_name, output_path in stage_outputs.items():
                    if output_path.endswith(".pkl"):
                        results.append(
                            self.model_validator.validate_model_file(output_path)
                        )
                    elif (
                        output_path.endswith(".csv")
                        and "metrics" in output_name.lower()
                    ):
                        try:
                            metrics_df = pd.read_csv(output_path)
                            if not metrics_df.empty:
                                # Convert to dict for validation
                                metrics_dict = {}
                                if "auc" in metrics_df.columns:
                                    metrics_dict["auc"] = metrics_df["auc"].mean()
                                if "accuracy" in metrics_df.columns:
                                    metrics_dict["accuracy"] = metrics_df[
                                        "accuracy"
                                    ].mean()
                                if "acc_test" in metrics_df.columns:
                                    metrics_dict["accuracy"] = metrics_df[
                                        "acc_test"
                                    ].mean()

                                results.append(
                                    self.model_validator.validate_performance_metrics(
                                        metrics_dict
                                    )
                                )
                        except Exception as e:
                            results.append(
                                ValidationResult(
                                    False, 
                                    f"Cannot load metrics {output_name}: {str(e)}", 
                                    level = "ERROR", 
                                )
                            )

            elif stage_name.lower() in ["predict", "backtest"]:
                # Validate prediction outputs
                for output_name, output_path in stage_outputs.items():
                    if (
                        output_path.endswith(".csv")
                        and "prediction" in output_name.lower()
                    ):
                        try:
                            pred_df = pd.read_csv(output_path)
                            if "prediction" in pred_df.columns:
                                results.append(
                                    self.model_validator.validate_predictions(
                                        pred_df["prediction"]
                                    )
                                )
                            elif "pred_proba" in pred_df.columns:
                                results.append(
                                    self.model_validator.validate_predictions(
                                        pred_df["pred_proba"]
                                    )
                                )
                        except Exception as e:
                            results.append(
                                ValidationResult(
                                    False, 
                                    f"Cannot load predictions {output_name}: {str(e)}", 
                                    level = "ERROR", 
                                )
                            )

            # Generic file existence check
            for output_name, output_path in stage_outputs.items():
                if not os.path.exists(output_path):
                    results.append(
                        ValidationResult(
                            False, 
                            f"Expected output not found: {output_name} at {output_path}", 
                            level = "ERROR", 
                        )
                    )
                else:
                    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                    if file_size < 0.001:  # Less than 1KB
                        results.append(
                            ValidationResult(
                                False, 
                                f"Output file too small: {output_name} ({file_size:.3f}MB)", 
                                level = "WARNING", 
                            )
                        )
                    else:
                        results.append(
                            ValidationResult(
                                True, 
                                f"Output file OK: {output_name} ({file_size:.2f}MB)", 
                            )
                        )

        except Exception as e:
            results.append(
                ValidationResult(
                    False, 
                    f"Stage validation error for {stage_name}: {str(e)}", 
                    level = "ERROR", 
                )
            )

        return results

    def validate_full_pipeline(
        self, output_dir: str = "output_default"
    ) -> Dict[str, List[ValidationResult]]:
        """Validate complete pipeline outputs"""
        validation_results = {}

        # Define expected outputs for each stage
        stage_outputs = {
            "preprocess": {
                "preprocessed_data": os.path.join(
                    output_dir, "preprocessed_super.parquet"
                )
            }, 
            "train": {
                "model": os.path.join(output_dir, "catboost_model_best.pkl"), 
                "features": os.path.join(output_dir, "train_features.txt"), 
            }, 
            "walkforward": {
                "metrics": os.path.join(output_dir, "walkforward_metrics.csv"), 
                "model_cv": os.path.join(output_dir, "catboost_model_best_cv.pkl"), 
            }, 
            "predict": {"predictions": os.path.join(output_dir, "predictions.csv")}, 
            "backtest": {
                "backtest_results": os.path.join(output_dir, "backtest_results.csv")
            }, 
        }

        # Validate each stage
        for stage_name, outputs in stage_outputs.items():
            validation_results[stage_name] = self.validate_pipeline_stage(
                stage_name, outputs
            )

        return validation_results

    def generate_validation_report(
        self, validation_results: Dict[str, List[ValidationResult]], output_file: str
    ):
        """Generate comprehensive validation report"""
        report = {
            "timestamp": datetime.now().isoformat(), 
            "overall_status": "UNKNOWN", 
            "stage_summaries": {}, 
            "critical_issues": [], 
            "warnings": [], 
            "recommendations": [], 
        }

        total_checks = 0
        passed_checks = 0
        critical_issues = []
        warnings = []

        for stage_name, results in validation_results.items():
            stage_summary = {
                "total_checks": len(results), 
                "passed_checks": sum(1 for r in results if r.passed), 
                "failed_checks": sum(1 for r in results if not r.passed), 
                "critical_count": sum(1 for r in results if r.level == "CRITICAL"), 
                "error_count": sum(1 for r in results if r.level == "ERROR"), 
                "warning_count": sum(1 for r in results if r.level == "WARNING"), 
                "details": [
                    {
                        "passed": r.passed, 
                        "message": r.message, 
                        "level": r.level, 
                        "score": r.score, 
                    }
                    for r in results
                ], 
            }

            report["stage_summaries"][stage_name] = stage_summary

            total_checks += len(results)
            passed_checks += sum(1 for r in results if r.passed)

            # Collect issues
            for result in results:
                if result.level == "CRITICAL":
                    critical_issues.append(f"{stage_name}: {result.message}")
                elif result.level in ["ERROR", "WARNING"]:
                    warnings.append(f"{stage_name}: {result.message}")

        # Determine overall status
        if critical_issues:
            report["overall_status"] = "CRITICAL"
        elif passed_checks / total_checks < 0.8:
            report["overall_status"] = "FAILED"
        elif warnings:
            report["overall_status"] = "WARNING"
        else:
            report["overall_status"] = "PASSED"

        report["critical_issues"] = critical_issues
        report["warnings"] = warnings

        # Generate recommendations
        recommendations = []
        if critical_issues:
            recommendations.append(
                "🔴 Address critical issues before production deployment"
            )
        if len(warnings) > 5:
            recommendations.append(
                "⚠️ Review and address warnings to improve pipeline reliability"
            )
        if passed_checks / total_checks < 0.9:
            recommendations.append(
                "🔧 Consider improving pipeline robustness - validation success rate is below 90%"
            )

        report["recommendations"] = recommendations
        report["summary_stats"] = {
            "total_checks": total_checks, 
            "passed_checks": passed_checks, 
            "success_rate": passed_checks / total_checks if total_checks > 0 else 0, 
        }

        # Save report
        with open(output_file, "w") as f:
            json.dump(report, f, indent = 2)

        return report


if __name__ == "__main__":
    # Test validation system
    validator = PipelineValidation()

    print("🔍 Testing validation system...")

    # Test with sample data
    test_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024 - 01 - 01", periods = 1000, freq = "1min"), 
            "open": np.random.normal(2000, 10, 1000), 
            "high": np.random.normal(2005, 10, 1000), 
            "low": np.random.normal(1995, 10, 1000), 
            "close": np.random.normal(2000, 10, 1000), 
            "volume": np.random.randint(100, 1000, 1000), 
        }
    )

    # Ensure OHLC logic
    test_data["high"] = np.maximum.reduce(
        [test_data["open"], test_data["high"], test_data["low"], test_data["close"]]
    )
    test_data["low"] = np.minimum.reduce(
        [test_data["open"], test_data["high"], test_data["low"], test_data["close"]]
    )

    data_validator = DataValidation()

    print("✅ Basic structure validation:")
    result = data_validator.validate_basic_structure(test_data)
    print(f"  {result.message}")

    print("✅ Data quality validation:")
    result = data_validator.validate_data_quality(test_data)
    print(f"  {result.message}")

    print("✅ Time series validation:")
    result = data_validator.validate_time_series(test_data)
    print(f"  {result.message}")

    print("\n🎯 Validation system test completed successfully!")