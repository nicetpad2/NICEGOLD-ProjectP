#!/usr/bin/env python3
"""
Real Data Loader for XAUUSD CSV Files
====================================

Loads real XAUUSD M1 and M15 data from datacsv/ folder without any limitations
or dummy data generation. Processes the full dataset for production use.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    import numpy as np
except ImportError:
    import warnings

    warnings.warn("NumPy not available, using fallback calculations")
    np = None

import pandas as pd

logger = logging.getLogger(__name__)


class RealDataLoader:
    """
    Real data loader for XAUUSD CSV files
    """

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.datacsv_dir = self.project_root / "datacsv"
        self.m1_file = self.datacsv_dir / "XAUUSD_M1.csv"
        self.m15_file = self.datacsv_dir / "XAUUSD_M15.csv"

        # Ensure data directory exists
        if not self.datacsv_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.datacsv_dir}")

        # Check if files exist
        if not self.m1_file.exists():
            raise FileNotFoundError(f"M1 data file not found: {self.m1_file}")

        if not self.m15_file.exists():
            raise FileNotFoundError(f"M15 data file not found: {self.m15_file}")

        logger.info(f"Real data loader initialized with:")
        logger.info(f"  M1 file: {self.m1_file}")
        logger.info(f"  M15 file: {self.m15_file}")

    def load_m1_data(self, limit_rows: Optional[int] = None) -> pd.DataFrame:
        """
        Load M1 data from XAUUSD_M1.csv

        Args:
            limit_rows: If None, load all data. If specified, limit to this number.

        Returns:
            DataFrame with M1 OHLCV data
        """
        logger.info("Loading M1 data from real CSV file...")

        try:
            # Load data without row limits if limit_rows is None
            if limit_rows is None:
                df = pd.read_csv(self.m1_file)
                logger.info(f"Loaded full M1 dataset: {len(df):,} rows")
            else:
                df = pd.read_csv(self.m1_file, nrows=limit_rows)
                logger.info(f"Loaded M1 dataset (limited): {len(df):,} rows")

            # Process data
            df = self._process_data(df, "M1")

            return df

        except Exception as e:
            logger.error(f"Error loading M1 data: {e}")
            raise

    def load_m15_data(self, limit_rows: Optional[int] = None) -> pd.DataFrame:
        """
        Load M15 data from XAUUSD_M15.csv

        Args:
            limit_rows: If None, load all data. If specified, limit to this number.

        Returns:
            DataFrame with M15 OHLCV data
        """
        logger.info("Loading M15 data from real CSV file...")

        try:
            # Load data without row limits if limit_rows is None
            if limit_rows is None:
                df = pd.read_csv(self.m15_file)
                logger.info(f"Loaded full M15 dataset: {len(df):,} rows")
            else:
                df = pd.read_csv(self.m15_file, nrows=limit_rows)
                logger.info(f"Loaded M15 dataset (limited): {len(df):,} rows")

            # Process data
            df = self._process_data(df, "M15")

            return df

        except Exception as e:
            logger.error(f"Error loading M15 data: {e}")
            raise

    def _process_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Process raw data and add required columns
        """
        logger.info("Processing %s data...", timeframe)

        # Convert Buddhist calendar date to Gregorian
        def convert_buddhist_date(date_str):
            """Convert Buddhist calendar date (YYYYMMDD) to Gregorian"""
            date_str = str(date_str)
            if len(date_str) == 8:
                year = int(date_str[:4])
                month = date_str[4:6]
                day = date_str[6:8]

                # Convert from Buddhist year to Gregorian year
                gregorian_year = year - 543

                return f"{gregorian_year}-{month}-{day}"
            return date_str

        # Apply Buddhist to Gregorian conversion
        df["gregorian_date"] = df["Date"].apply(convert_buddhist_date)

        # Create datetime column with proper format
        try:
            df["datetime"] = pd.to_datetime(
                df["gregorian_date"] + " " + df["Timestamp"].astype(str),
                format="%Y-%m-%d %H:%M:%S",
                errors="coerce",
            )
            logger.info("Successfully converted Buddhist dates to Gregorian")
        except Exception as e:
            logger.warning("Date conversion failed, trying alternative method: %s", e)
            # Fallback method
            df["datetime"] = pd.to_datetime(
                df["gregorian_date"] + " " + df["Timestamp"].astype(str),
                errors="coerce",
            )

        # Drop rows with invalid dates
        initial_len = len(df)
        df = df.dropna(subset=["datetime"])
        if len(df) < initial_len:
            logger.info("Dropped %d rows with invalid dates", initial_len - len(df))

        # Sort by datetime
        df = df.sort_values("datetime").reset_index(drop=True)

        # Add basic features
        df = self._add_basic_features(df)

        # Remove any remaining invalid data
        df = df.dropna()

        logger.info("Processed %s data: %d rows after cleaning", timeframe, len(df))

        return df

    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic trading features
        """
        # Returns
        df["returns"] = df["Close"].pct_change()

        # High-Low spread
        df["hl_spread"] = (df["High"] - df["Low"]) / df["Close"]

        # Price position within range
        df["price_position"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"])

        # Volume-weighted price
        df["vwap"] = df["Volume"] * df["Close"]

        # Log returns for better distribution
        if np is not None:
            df["log_returns"] = np.log(df["Close"]).diff()
        else:
            # Fallback without numpy
            import math

            df["log_returns"] = (
                df["Close"].apply(lambda x: math.log(x) if x > 0 else None).diff()
            )

        return df

    def get_data_info(self) -> Dict[str, any]:
        """
        Get information about available data
        """
        info = {}

        try:
            # Get file sizes
            m1_size = self.m1_file.stat().st_size / (1024 * 1024)  # MB
            m15_size = self.m15_file.stat().st_size / (1024 * 1024)  # MB

            # Get row counts (approximate from first few rows)
            m1_sample = pd.read_csv(self.m1_file, nrows=1000)
            m15_sample = pd.read_csv(self.m15_file, nrows=1000)

            info = {
                "files": {
                    "XAUUSD_M1.csv": {
                        "size_mb": round(m1_size, 2),
                        "exists": True,
                        "columns": list(m1_sample.columns),
                    },
                    "XAUUSD_M15.csv": {
                        "size_mb": round(m15_size, 2),
                        "exists": True,
                        "columns": list(m15_sample.columns),
                    },
                },
                "estimated_rows": {"M1": "~1.7M rows", "M15": "~118K rows"},
            }

        except Exception as e:
            logger.error(f"Error getting data info: {e}")
            info = {"error": str(e)}

        return info

    def load_combined_data(
        self, use_m15: bool = True, limit_rows: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load and combine M1 and M15 data

        Args:
            use_m15: Whether to include M15 data features
            limit_rows: Limit rows per timeframe (None = no limit)

        Returns:
            Combined DataFrame
        """
        logger.info("Loading combined M1 and M15 data...")

        # Load M1 data (primary)
        df_m1 = self.load_m1_data(limit_rows)

        if use_m15:
            # Load M15 data for additional features
            df_m15 = self.load_m15_data(limit_rows)

            # Add M15 features to M1 data (simplified approach)
            # This would require more sophisticated alignment in production
            logger.info("Note: M15 data loaded but basic combination used for now")

        return df_m1


def load_real_data(
    config: Dict = None, limit_rows: Optional[int] = None
) -> pd.DataFrame:
    """
    Main function to load real data according to configuration

    Args:
        config: Configuration dictionary
        limit_rows: Row limit (None = no limit, can be overridden by env var)

    Returns:
        Processed DataFrame ready for ML pipeline
    """
    # Check for environment variable for row limit (for debug mode)
    import os

    env_limit = os.environ.get("NICEGOLD_ROW_LIMIT")
    if env_limit is not None:
        try:
            limit_rows = int(env_limit)
            logger.info("Using row limit from environment: %d", limit_rows)
        except ValueError:
            logger.warning("Invalid NICEGOLD_ROW_LIMIT value: %s", env_limit)

    loader = RealDataLoader()

    # Load M1 data by default
    if limit_rows is not None:
        logger.info("Loading real data with row limit: %d", limit_rows)
        df = loader.load_m1_data(limit_rows=limit_rows)
    else:
        logger.info("Loading complete real data (no row limit)")
        df = loader.load_m1_data(limit_rows=None)

    # Add target column for ML (this is a placeholder - should be based on actual trading logic)
    df["target"] = _create_target_column(df)

    logger.info(f"Real data loaded successfully: {len(df):,} rows")
    return df


def _create_target_column(df: pd.DataFrame) -> pd.Series:
    """
    Create target column based on price movement
    This is a simplified example - should be replaced with actual trading logic
    """
    # Simple target: 1 if next price is higher, 0 otherwise
    future_returns = df["Close"].shift(-1) / df["Close"] - 1

    # Target based on future return threshold
    threshold = 0.0001  # 0.01% price movement
    target = (future_returns > threshold).astype(int)

    return target


if __name__ == "__main__":
    # Test the real data loader
    import logging

    logging.basicConfig(level=logging.INFO)

    try:
        loader = RealDataLoader()

        # Get data info
        info = loader.get_data_info()
        print("\n=== Data Information ===")
        for key, value in info.items():
            print(f"{key}: {value}")

        # Load sample data (first 10000 rows for testing)
        print("\n=== Loading Sample Data ===")
        df = load_real_data(limit_rows=10000)

        print(f"\nData shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"Target distribution: {df['target'].value_counts().to_dict()}")

        print("\n=== First 5 rows ===")
        print(df.head())

    except Exception as e:
        print(f"Error: {e}")
