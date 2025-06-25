#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NICEGOLD ProjectP - Data Validator Module
Validates and cleans trading data for analysis
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """Enterprise-grade data validator for trading data"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize DataValidator with optional configuration"""
        self.config = config or {}

        # Setup logger
        import logging

        self.logger = logging.getLogger(__name__)

        # Default required columns for trading data
        self.required_columns = self.config.get(
            "required_columns", ["open", "high", "low", "close"]
        )
        self.optional_columns = self.config.get(
            "optional_columns", ["volume", "timestamp"]
        )

        # Validation thresholds
        self.max_price_ratio = self.config.get(
            "max_price_ratio", 10.0
        )  # High/Low ratio threshold
        self.min_volume = self.config.get("min_volume", 0)
        self.outlier_zscore_threshold = self.config.get("outlier_zscore_threshold", 5.0)

        # Column mapping for case-insensitive matching
        self.column_mapping = {
            'open': ['open', 'Open', 'OPEN'],
            'high': ['high', 'High', 'HIGH'], 
            'low': ['low', 'Low', 'LOW'],
            'close': ['close', 'Close', 'CLOSE'],
            'volume': ['volume', 'Volume', 'VOLUME'],
            'timestamp': ['timestamp', 'Timestamp', 'TIMESTAMP', 'time', 'Time', 'TIME'],
        }

    def normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names to lowercase standard format
        
        Args:
            df: DataFrame with potentially mixed case columns
            
        Returns:
            DataFrame with normalized column names
        """
        df_normalized = df.copy()
        
        # Create reverse mapping from actual columns to standard names
        column_rename_map = {}
        
        for standard_name, possible_names in self.column_mapping.items():
            for possible_name in possible_names:
                if possible_name in df_normalized.columns:
                    column_rename_map[possible_name] = standard_name
                    break
        
        # Rename columns
        if column_rename_map:
            df_normalized = df_normalized.rename(columns=column_rename_map)
            self.logger.info(f"Normalized column names: {column_rename_map}")
        
        return df_normalized

    def validate_data_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate basic data structure

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with validation results
        """
        results = {"is_valid": True, "errors": [], "warnings": [], "info": {}}

        # Check if DataFrame is empty
        if df.empty:
            results["is_valid"] = False
            results["errors"].append("DataFrame is empty")
            return results

        # Check for required columns
        missing_required = [
            col for col in self.required_columns if col not in df.columns
        ]
        if missing_required:
            results["is_valid"] = False
            results["errors"].append(f"Missing required columns: {missing_required}")

        # Check for recommended columns
        missing_optional = [
            col for col in self.optional_columns if col not in df.columns
        ]
        if missing_optional:
            results["warnings"].append(f"Missing optional columns: {missing_optional}")

        # Check data types
        for col in self.required_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                results["warnings"].append(f"Column '{col}' is not numeric")

        results["info"]["shape"] = df.shape
        results["info"]["columns"] = list(df.columns)

        return results

    def validate_ohlc_logic(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate OHLC price logic

        Args:
            df: DataFrame with OHLC data

        Returns:
            Dictionary with validation results
        """
        results = {"is_valid": True, "errors": [], "warnings": [], "violations": 0}

        if not all(col in df.columns for col in self.required_columns):
            results["is_valid"] = False
            results["errors"].append("Missing OHLC columns for validation")
            return results

        # Check High >= max(Open, Close) and High >= Low
        high_violations = (
            (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["high"] < df["low"])
        ).sum()

        # Check Low <= min(Open, Close) and Low <= High
        low_violations = (
            (df["low"] > df["open"])
            | (df["low"] > df["close"])
            | (df["low"] > df["high"])
        ).sum()

        total_violations = high_violations + low_violations
        results["violations"] = total_violations

        if total_violations > 0:
            violation_rate = (total_violations / len(df) * 100) if len(df) > 0 else 0.0
            if violation_rate > 5:  # More than 5% violations
                results["is_valid"] = False
                results["errors"].append(
                    f"High OHLC logic violation rate: {violation_rate:.2f}%"
                )
            else:
                results["warnings"].append(
                    f"OHLC logic violations: {total_violations} ({violation_rate:.2f}%)"
                )

        return results

    def detect_outliers(
        self, df: pd.DataFrame, method: str = "iqr", factor: float = 1.5
    ) -> Dict[str, Any]:
        """
        Detect outliers in price data

        Args:
            df: DataFrame with price data
            method: Outlier detection method ('iqr', 'zscore')
            factor: Outlier factor threshold

        Returns:
            Dictionary with outlier information
        """
        results = {"outliers": {}, "outlier_count": 0, "outlier_rate": 0.0}

        # Check if DataFrame is empty
        if df.empty or len(df) == 0:
            self.logger.warning("DataFrame is empty, skipping outlier detection")
            return results

        price_columns = [
            col for col in ["open", "high", "low", "close"] if col in df.columns
        ]

        # Check if no price columns found
        if not price_columns:
            self.logger.warning("No price columns found, skipping outlier detection")
            return results

        for col in price_columns:
            if method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)

            elif method == "zscore":
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val == 0 or pd.isna(std_val):
                    # If standard deviation is 0, no outliers can be detected
                    outliers = pd.Series([False] * len(df), index=df.index)
                else:
                    z_scores = np.abs((df[col] - mean_val) / std_val)
                    outliers = z_scores > factor

            else:
                raise ValueError(f"Unknown outlier detection method: {method}")

            results["outliers"][col] = {
                "count": outliers.sum(),
                "rate": (outliers.sum() / len(df) * 100) if len(df) > 0 else 0.0,
                "indices": outliers[outliers].index.tolist(),
            }

        total_outliers = sum(info["count"] for info in results["outliers"].values())
        results["outlier_count"] = total_outliers
        total_cells = len(df) * len(price_columns)
        results["outlier_rate"] = (
            (total_outliers / total_cells * 100) if total_cells > 0 else 0.0
        )

        return results

    def check_data_gaps(
        self, df: pd.DataFrame, timestamp_col: str = "timestamp"
    ) -> Dict[str, Any]:
        """
        Check for gaps in time series data

        Args:
            df: DataFrame with timestamp
            timestamp_col: Name of timestamp column

        Returns:
            Dictionary with gap information
        """
        results = {"has_gaps": False, "gap_count": 0, "gaps": [], "largest_gap": None}

        if timestamp_col not in df.columns:
            results["error"] = f"Timestamp column '{timestamp_col}' not found"
            return results

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            try:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            except:
                results["error"] = f"Cannot convert '{timestamp_col}' to datetime"
                return results

        # Sort by timestamp
        df_sorted = df.sort_values(timestamp_col)

        # Calculate time differences
        time_diffs = df_sorted[timestamp_col].diff()

        # Detect expected frequency
        mode_diff = (
            time_diffs.mode().iloc[0]
            if not time_diffs.mode().empty
            else pd.Timedelta(minutes=1)
        )

        # Find gaps (differences significantly larger than expected)
        gap_threshold = mode_diff * 2
        gaps = time_diffs[time_diffs > gap_threshold]

        if not gaps.empty:
            results["has_gaps"] = True
            results["gap_count"] = len(gaps)
            results["largest_gap"] = gaps.max()

            # Store gap details
            for idx, gap in gaps.items():
                gap_start = df_sorted.loc[idx - 1, timestamp_col] if idx > 0 else None
                gap_end = df_sorted.loc[idx, timestamp_col]
                results["gaps"].append(
                    {"start": gap_start, "end": gap_end, "duration": gap}
                )

        return results

    def clean_data(
        self,
        df: pd.DataFrame,
        remove_outliers: bool = True,
        fix_ohlc_violations: bool = True,
        fill_missing: bool = True,
    ) -> pd.DataFrame:
        """
        Clean data based on validation results

        Args:
            df: DataFrame to clean
            remove_outliers: Whether to remove outliers
            fix_ohlc_violations: Whether to fix OHLC violations
            fill_missing: Whether to fill missing values

        Returns:
            Cleaned DataFrame
        """
        # First normalize column names
        df_clean = self.normalize_column_names(df.copy())

        logger.info(f"Starting data cleaning. Initial shape: {df_clean.shape}")

        # Check if DataFrame is empty
        if df_clean.empty or len(df_clean) == 0:
            logger.warning("DataFrame is empty, returning as-is")
            return df_clean

        # Remove rows with all NaN values in OHLC columns
        ohlc_cols = [col for col in self.required_columns if col in df_clean.columns]
        if ohlc_cols:
            df_clean = df_clean.dropna(subset=ohlc_cols, how="all")
            logger.info(f"After removing all-NaN rows: {df_clean.shape}")

            # Check again after dropna
            if df_clean.empty or len(df_clean) == 0:
                logger.warning("DataFrame became empty after removing NaN rows")
                return df_clean

        # Fix OHLC violations
        if fix_ohlc_violations and all(
            col in df_clean.columns for col in self.required_columns
        ):
            # Ensure High is the maximum of OHLC
            df_clean["high"] = df_clean[["open", "high", "low", "close"]].max(axis=1)
            # Ensure Low is the minimum of OHLC
            df_clean["low"] = df_clean[["open", "high", "low", "close"]].min(axis=1)
            logger.info("Fixed OHLC logic violations")

        # Fill missing values
        if fill_missing:
            for col in ohlc_cols:
                if df_clean[col].isnull().any():
                    # Forward fill first, then backward fill
                    df_clean[col] = (
                        df_clean[col].fillna(method="ffill").fillna(method="bfill")
                    )
            logger.info("Filled missing values")

        # Remove outliers
        if remove_outliers:
            outlier_info = self.detect_outliers(df_clean)
            if outlier_info["outlier_rate"] > 0:
                # Create a mask for rows without outliers
                mask = pd.Series([True] * len(df_clean), index=df_clean.index)

                for col, info in outlier_info["outliers"].items():
                    if info["count"] > 0:
                        outlier_indices = info["indices"]
                        mask[outlier_indices] = False

                df_clean = df_clean[mask]
                logger.info(
                    f"Removed {(~mask).sum()} outlier rows. New shape: {df_clean.shape}"
                )

        logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")
        return df_clean

    def generate_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with quality report
        """
        report = {
            "timestamp": pd.Timestamp.now(),
            "data_shape": df.shape,
            "structure_validation": self.validate_data_structure(df),
            "ohlc_validation": {},
            "outlier_analysis": {},
            "gap_analysis": {},
            "summary": {},
        }

        # OHLC validation
        if all(col in df.columns for col in self.required_columns):
            report["ohlc_validation"] = self.validate_ohlc_logic(df)

        # Outlier analysis
        report["outlier_analysis"] = self.detect_outliers(df)

        # Gap analysis
        if "timestamp" in df.columns:
            report["gap_analysis"] = self.check_data_gaps(df)

        # Summary
        report["summary"] = {
            "overall_quality": "Good",  # Will be determined by validation results
            "data_completeness": (1 - df.isnull().sum().sum() / df.size) * 100,
            "recommendations": [],
        }

        # Determine overall quality
        if not report["structure_validation"]["is_valid"]:
            report["summary"]["overall_quality"] = "Poor"
            report["summary"]["recommendations"].extend(
                report["structure_validation"]["errors"]
            )
        elif report["outlier_analysis"]["outlier_rate"] > 10:
            report["summary"]["overall_quality"] = "Fair"
            report["summary"]["recommendations"].append(
                "High outlier rate detected - consider data cleaning"
            )
        elif report["gap_analysis"].get("has_gaps", False):
            report["summary"]["overall_quality"] = "Fair"
            report["summary"]["recommendations"].append(
                "Data gaps detected - consider filling missing periods"
            )

        return report

    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Main validation method that performs comprehensive data validation

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with validation results
        """
        logger.info(f"Starting data validation for DataFrame with shape {df.shape}")

        # First normalize column names
        df_normalized = self.normalize_column_names(df)

        validation_results = {
            "validation_timestamp": pd.Timestamp.now().isoformat(),
            "data_shape": df_normalized.shape,
            "structure_validation": {},
            "ohlc_validation": {},
            "outlier_analysis": {},
            "gap_analysis": {},
            "overall_status": "passed",
            "errors": [],
            "warnings": [],
        }

        try:
            # Structure validation
            structure_results = self.validate_data_structure(df_normalized)
            validation_results["structure_validation"] = structure_results

            if not structure_results["is_valid"]:
                validation_results["overall_status"] = "failed"
                validation_results["errors"].extend(structure_results["errors"])

            validation_results["warnings"].extend(structure_results["warnings"])

            # OHLC validation (if applicable)
            if all(
                col in df_normalized.columns
                for col in self.required_columns
            ):
                ohlc_results = self.validate_ohlc_logic(df_normalized)
                validation_results["ohlc_validation"] = ohlc_results

                if not ohlc_results["is_valid"]:
                    validation_results["warnings"].extend(ohlc_results["errors"])

            # Outlier detection
            outlier_results = self.detect_outliers(df_normalized)
            validation_results["outlier_analysis"] = outlier_results

            # Configurable outlier rate threshold (default 20% instead of 15%)
            outlier_threshold = self.config.get("outlier_rate_threshold", 20.0)
            if outlier_results["outlier_rate"] > outlier_threshold:
                validation_results["warnings"].append(
                    f"High outlier rate detected: {outlier_results['outlier_rate']:.1f}% (threshold: {outlier_threshold}%)"
                )
            else:
                # Log as info if outliers are within acceptable range
                self.logger.info(
                    f"Outlier rate within acceptable range: {outlier_results['outlier_rate']:.1f}% (threshold: {outlier_threshold}%)"
                )

            # Gap analysis with improved handling
            timestamp_cols = [
                col
                for col in df_normalized.columns
                if "time" in col.lower() or "date" in col.lower()
            ]
            if timestamp_cols:
                gap_results = self.check_data_gaps(df_normalized, timestamp_col=timestamp_cols[0])
                validation_results["gap_analysis"] = gap_results

                # Configurable gap threshold
                max_acceptable_gaps = self.config.get("max_acceptable_gaps", 1000)
                gap_count = gap_results.get("gap_count", 0)
                
                if gap_results.get("has_gaps", False):
                    if gap_count > max_acceptable_gaps:
                        validation_results["warnings"].append(
                            f"High number of data gaps: {gap_count} gaps found (threshold: {max_acceptable_gaps})"
                        )
                    else:
                        # Log as info if gaps are within acceptable range
                        self.logger.info(
                            f"Data gaps within acceptable range: {gap_count} gaps found (threshold: {max_acceptable_gaps})"
                        )
                        
                    # Additional gap analysis
                    largest_gap = gap_results.get("largest_gap")
                    if largest_gap:
                        gap_hours = largest_gap.total_seconds() / 3600
                        max_gap_hours = self.config.get("max_gap_hours", 24)
                        if gap_hours > max_gap_hours:
                            validation_results["warnings"].append(
                                f"Large data gap detected: {gap_hours:.1f} hours (max allowed: {max_gap_hours}h)"
                            )

            # Log results
            if validation_results["overall_status"] == "passed":
                self.logger.info("Data validation completed successfully")
            else:
                self.logger.warning(
                    f"Data validation completed with errors: {validation_results['errors']}"
                )

            if validation_results["warnings"]:
                self.logger.warning(
                    f"Data validation warnings: {validation_results['warnings']}"
                )

        except Exception as e:
            validation_results["overall_status"] = "failed"
            validation_results["errors"].append(f"Validation error: {str(e)}")
            self.logger.error(f"Data validation failed: {str(e)}")

        return validation_results
