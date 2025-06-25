#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NICEGOLD ProjectP - Data Loader Module
Handles loading and initial processing of trading data
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Enterprise-grade data loader for trading data"""

    def __init__(self, data_folder: Path):
        """
        Initialize DataLoader

        Args:
            data_folder: Path to data directory
        """
        self.data_folder = Path(data_folder)
        self.supported_formats = [".csv", ".parquet", ".json"]

    def get_available_files(self) -> List[Path]:
        """Get list of available data files"""
        files = []
        for ext in self.supported_formats:
            files.extend(list(self.data_folder.glob(f"*{ext}")))
        return sorted(files)

    def load_csv(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """
        Load CSV file with automatic format detection

        Args:
            file_path: Path to CSV file
            **kwargs: Additional pandas.read_csv arguments

        Returns:
            DataFrame with loaded data
        """
        try:
            # Try to detect common CSV formats
            common_configs = [{"sep": ","}, {"sep": ";"}, {"sep": "\t"}, {"sep": "|"}]

            for config in common_configs:
                try:
                    df = pd.read_csv(file_path, **config, **kwargs, nrows=5)
                    if len(df.columns) > 1:  # Multi-column data found
                        df_full = pd.read_csv(file_path, **config, **kwargs)
                        logger.info(
                            f"Successfully loaded {file_path} with separator '{config['sep']}'"
                        )
                        return df_full
                except:
                    continue

            # Default fallback
            df = pd.read_csv(file_path, **kwargs)
            logger.info(f"Loaded {file_path} with default settings")
            return df

        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise

    def load_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load trading data automatically

        Args:
            file_path: Specific file to load, or None to auto-select

        Returns:
            DataFrame with loaded trading data
        """
        if file_path is None:
            # Auto-select best file
            available_files = self.get_available_files()
            if not available_files:
                raise FileNotFoundError(f"No data files found in {self.data_folder}")
            file_path = available_files[0]  # Take first available file

        logger.info(f"Loading data from {file_path}")

        # Load based on file extension
        if file_path.suffix.lower() == ".csv":
            df = self.load_csv(file_path)
        elif file_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(file_path)
        elif file_path.suffix.lower() == ".json":
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df

    def auto_detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Auto-detect common trading data columns

        Args:
            df: Input DataFrame

        Returns:
            Dictionary mapping standard names to actual column names
        """
        column_mapping = {}
        columns_lower = [col.lower() for col in df.columns]

        # Common patterns for OHLCV data
        patterns = {
            "timestamp": ["time", "timestamp", "date", "datetime"],
            "open": ["open", "o"],
            "high": ["high", "h"],
            "low": ["low", "l"],
            "close": ["close", "c", "price"],
            "volume": ["volume", "vol", "v"],
            "symbol": ["symbol", "pair", "instrument"],
        }

        for standard_name, possible_names in patterns.items():
            for possible_name in possible_names:
                matches = [col for col in df.columns if possible_name in col.lower()]
                if matches:
                    column_mapping[standard_name] = matches[0]
                    break

        return column_mapping

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data with standard formatting

        Args:
            df: Raw DataFrame

        Returns:
            Prepared DataFrame with standard column names
        """
        df = df.copy()

        # Auto-detect columns
        column_mapping = self.auto_detect_columns(df)

        # Rename columns to standard names
        if column_mapping:
            df = df.rename(columns={v: k for k, v in column_mapping.items()})

        # Convert timestamp if present
        if "timestamp" in df.columns:
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp")
            except:
                logger.warning("Could not convert timestamp column")

        # Ensure numeric columns are properly typed
        numeric_columns = ["open", "high", "low", "close", "volume"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Remove rows with all NaN values in OHLC columns
        ohlc_cols = [
            col for col in ["open", "high", "low", "close"] if col in df.columns
        ]
        if ohlc_cols:
            df = df.dropna(subset=ohlc_cols, how="all")

        logger.info(f"Data prepared: {len(df)} rows remaining")
        return df

    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive information about loaded data

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with data information
        """
        info = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
        }

        # Time range if timestamp available
        if "timestamp" in df.columns:
            info["time_range"] = {
                "start": df["timestamp"].min(),
                "end": df["timestamp"].max(),
                "duration": df["timestamp"].max() - df["timestamp"].min(),
            }

        # Price statistics if available
        if "close" in df.columns:
            info["price_stats"] = {
                "min": df["close"].min(),
                "max": df["close"].max(),
                "mean": df["close"].mean(),
                "std": df["close"].std(),
            }

        return info
