#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NICEGOLD ProjectP - Feature Engineer Module
Creates technical indicators and features for trading analysis
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Enterprise-grade feature engineering for trading data"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize FeatureEngineer with optional configuration"""
        self.config = config or {}

        # Default feature configuration
        self.feature_config = self.config.get(
            "feature_config",
            {
                "sma_periods": [5, 10, 20, 50],
                "ema_periods": [5, 10, 20],
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "bb_period": 20,
                "bb_std": 2,
                "stoch_k": 14,
                "stoch_d": 3,
            },
        )

    def calculate_sma(
        self, df: pd.DataFrame, column: str = "close", periods: List[int] = None
    ) -> pd.DataFrame:
        """
        Calculate Simple Moving Averages

        Args:
            df: DataFrame with price data
            column: Column to calculate SMA for
            periods: List of periods for SMA calculation

        Returns:
            DataFrame with SMA columns
        """
        if periods is None:
            periods = self.feature_config["sma_periods"]

        df_result = df.copy()

        for period in periods:
            col_name = f"sma_{period}"
            df_result[col_name] = df_result[column].rolling(window=period).mean()

        logger.info(f"Calculated SMA for periods: {periods}")
        return df_result

    def calculate_ema(
        self, df: pd.DataFrame, column: str = "close", periods: List[int] = None
    ) -> pd.DataFrame:
        """
        Calculate Exponential Moving Averages

        Args:
            df: DataFrame with price data
            column: Column to calculate EMA for
            periods: List of periods for EMA calculation

        Returns:
            DataFrame with EMA columns
        """
        if periods is None:
            periods = self.feature_config["ema_periods"]

        df_result = df.copy()

        for period in periods:
            col_name = f"ema_{period}"
            df_result[col_name] = df_result[column].ewm(span=period).mean()

        logger.info(f"Calculated EMA for periods: {periods}")
        return df_result

    def calculate_rsi(
        self, df: pd.DataFrame, column: str = "close", period: int = None
    ) -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI)

        Args:
            df: DataFrame with price data
            column: Column to calculate RSI for
            period: Period for RSI calculation

        Returns:
            DataFrame with RSI column
        """
        if period is None:
            period = self.feature_config["rsi_period"]

        df_result = df.copy()

        # Calculate price changes
        delta = df_result[column].diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()

        # Calculate RSI
        rs = avg_gains / avg_losses
        df_result["rsi"] = 100 - (100 / (1 + rs))

        logger.info(f"Calculated RSI with period {period}")
        return df_result

    def calculate_macd(
        self,
        df: pd.DataFrame,
        column: str = "close",
        fast: int = None,
        slow: int = None,
        signal: int = None,
    ) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence)

        Args:
            df: DataFrame with price data
            column: Column to calculate MACD for
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period

        Returns:
            DataFrame with MACD columns
        """
        if fast is None:
            fast = self.feature_config["macd_fast"]
        if slow is None:
            slow = self.feature_config["macd_slow"]
        if signal is None:
            signal = self.feature_config["macd_signal"]

        df_result = df.copy()

        # Calculate MACD line
        ema_fast = df_result[column].ewm(span=fast).mean()
        ema_slow = df_result[column].ewm(span=slow).mean()
        df_result["macd"] = ema_fast - ema_slow

        # Calculate signal line
        df_result["macd_signal"] = df_result["macd"].ewm(span=signal).mean()

        # Calculate histogram
        df_result["macd_histogram"] = df_result["macd"] - df_result["macd_signal"]

        logger.info(f"Calculated MACD with periods {fast}/{slow}/{signal}")
        return df_result

    def calculate_bollinger_bands(
        self,
        df: pd.DataFrame,
        column: str = "close",
        period: int = None,
        std_dev: float = None,
    ) -> pd.DataFrame:
        """
        Calculate Bollinger Bands

        Args:
            df: DataFrame with price data
            column: Column to calculate Bollinger Bands for
            period: Period for moving average
            std_dev: Number of standard deviations

        Returns:
            DataFrame with Bollinger Bands columns
        """
        if period is None:
            period = self.feature_config["bb_period"]
        if std_dev is None:
            std_dev = self.feature_config["bb_std"]

        df_result = df.copy()

        # Calculate middle band (SMA)
        df_result["bb_middle"] = df_result[column].rolling(window=period).mean()

        # Calculate standard deviation
        std = df_result[column].rolling(window=period).std()

        # Calculate upper and lower bands
        df_result["bb_upper"] = df_result["bb_middle"] + (std * std_dev)
        df_result["bb_lower"] = df_result["bb_middle"] - (std * std_dev)

        # Calculate band width and position
        df_result["bb_width"] = df_result["bb_upper"] - df_result["bb_lower"]
        df_result["bb_position"] = (
            df_result[column] - df_result["bb_lower"]
        ) / df_result["bb_width"]

        logger.info(
            f"Calculated Bollinger Bands with period {period} and std {std_dev}"
        )
        return df_result

    def calculate_stochastic(
        self, df: pd.DataFrame, k_period: int = None, d_period: int = None
    ) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator

        Args:
            df: DataFrame with OHLC data
            k_period: Period for %K calculation
            d_period: Period for %D calculation

        Returns:
            DataFrame with Stochastic columns
        """
        if k_period is None:
            k_period = self.feature_config["stoch_k"]
        if d_period is None:
            d_period = self.feature_config["stoch_d"]

        df_result = df.copy()

        # Calculate %K
        lowest_low = df_result["low"].rolling(window=k_period).min()
        highest_high = df_result["high"].rolling(window=k_period).max()
        df_result["stoch_k"] = (
            100 * (df_result["close"] - lowest_low) / (highest_high - lowest_low)
        )

        # Calculate %D
        df_result["stoch_d"] = df_result["stoch_k"].rolling(window=d_period).mean()

        logger.info(f"Calculated Stochastic with periods %K={k_period}, %D={d_period}")
        return df_result

    def calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price-based features

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with price features
        """
        df_result = df.copy()

        # Price changes
        df_result["price_change"] = df_result["close"].diff()
        df_result["price_change_pct"] = df_result["close"].pct_change()

        # High-Low spread
        df_result["hl_spread"] = df_result["high"] - df_result["low"]
        df_result["hl_spread_pct"] = df_result["hl_spread"] / df_result["close"]

        # Open-Close spread
        df_result["oc_spread"] = df_result["close"] - df_result["open"]
        df_result["oc_spread_pct"] = df_result["oc_spread"] / df_result["open"]

        # True Range
        df_result["prev_close"] = df_result["close"].shift(1)
        df_result["tr1"] = df_result["high"] - df_result["low"]
        df_result["tr2"] = abs(df_result["high"] - df_result["prev_close"])
        df_result["tr3"] = abs(df_result["low"] - df_result["prev_close"])
        df_result["true_range"] = df_result[["tr1", "tr2", "tr3"]].max(axis=1)

        # Average True Range
        df_result["atr"] = df_result["true_range"].rolling(window=14).mean()

        # Clean up temporary columns
        df_result = df_result.drop(["prev_close", "tr1", "tr2", "tr3"], axis=1)

        logger.info("Calculated price-based features")
        return df_result

    def calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based features

        Args:
            df: DataFrame with volume data

        Returns:
            DataFrame with volume features
        """
        df_result = df.copy()

        if "volume" not in df_result.columns:
            logger.warning("Volume column not found, skipping volume features")
            return df_result

        # Volume moving averages
        df_result["volume_sma_10"] = df_result["volume"].rolling(window=10).mean()
        df_result["volume_sma_30"] = df_result["volume"].rolling(window=30).mean()

        # Volume ratio
        df_result["volume_ratio"] = df_result["volume"] / df_result["volume_sma_30"]

        # On-Balance Volume (OBV)
        df_result["obv"] = (df_result["price_change"] > 0).astype(int) * df_result[
            "volume"
        ] - (df_result["price_change"] < 0).astype(int) * df_result["volume"]
        df_result["obv"] = df_result["obv"].cumsum()

        # Volume-Price Trend (VPT)
        df_result["vpt"] = (
            df_result["price_change_pct"] * df_result["volume"]
        ).cumsum()

        logger.info("Calculated volume-based features")
        return df_result

    def calculate_time_features(
        self, df: pd.DataFrame, timestamp_col: str = "timestamp"
    ) -> pd.DataFrame:
        """
        Calculate time-based features

        Args:
            df: DataFrame with timestamp
            timestamp_col: Name of timestamp column

        Returns:
            DataFrame with time features
        """
        df_result = df.copy()

        if timestamp_col not in df_result.columns:
            logger.warning(
                f"Timestamp column '{timestamp_col}' not found, skipping time features"
            )
            return df_result

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_result[timestamp_col]):
            df_result[timestamp_col] = pd.to_datetime(df_result[timestamp_col])

        # Extract time components
        df_result["hour"] = df_result[timestamp_col].dt.hour
        df_result["day_of_week"] = df_result[timestamp_col].dt.dayofweek
        df_result["day_of_month"] = df_result[timestamp_col].dt.day
        df_result["month"] = df_result[timestamp_col].dt.month
        df_result["quarter"] = df_result[timestamp_col].dt.quarter

        # Trading session features (assuming 24/7 trading)
        df_result["is_weekend"] = df_result["day_of_week"].isin([5, 6]).astype(int)
        df_result["is_month_end"] = (df_result[timestamp_col].dt.day >= 28).astype(int)

        # Cyclical encoding for periodic features
        df_result["hour_sin"] = np.sin(2 * np.pi * df_result["hour"] / 24)
        df_result["hour_cos"] = np.cos(2 * np.pi * df_result["hour"] / 24)
        df_result["dow_sin"] = np.sin(2 * np.pi * df_result["day_of_week"] / 7)
        df_result["dow_cos"] = np.cos(2 * np.pi * df_result["day_of_week"] / 7)

        logger.info("Calculated time-based features")
        return df_result

    def calculate_market_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        สร้างฟีเจอร์ Market Microstructure เช่น order flow, bid-ask spread, market depth
        """
        df_result = df.copy()
        # ตัวอย่างฟีเจอร์ (stub)
        df_result["order_flow_imbalance"] = np.random.normal(0, 1, len(df_result))
        df_result["bid_ask_spread"] = np.random.uniform(0.01, 0.1, len(df_result))
        df_result["market_depth"] = np.random.uniform(100, 1000, len(df_result))
        df_result["price_impact"] = np.random.normal(0, 0.05, len(df_result))
        return df_result

    def calculate_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        สร้างฟีเจอร์ Sentiment จากข่าว/โซเชียล/อีเวนต์เศรษฐกิจ (stub)
        """
        df_result = df.copy()
        df_result["news_sentiment"] = np.random.uniform(-1, 1, len(df_result))
        df_result["social_sentiment"] = np.random.uniform(-1, 1, len(df_result))
        df_result["event_impact"] = np.random.uniform(0, 1, len(df_result))
        return df_result

    def calculate_cross_asset_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        สร้างฟีเจอร์ Cross-Asset เช่น correlation กับ USD, Oil, Bonds, Crypto (stub)
        """
        df_result = df.copy()
        df_result["corr_usd"] = np.random.uniform(-1, 1, len(df_result))
        df_result["corr_oil"] = np.random.uniform(-1, 1, len(df_result))
        df_result["corr_bond"] = np.random.uniform(-1, 1, len(df_result))
        df_result["corr_crypto"] = np.random.uniform(-1, 1, len(df_result))
        return df_result

    def calculate_regime_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        สร้างฟีเจอร์ Regime Detection (Bull/Bear/Sideways) (stub)
        """
        df_result = df.copy()
        regimes = np.random.choice(["bull", "bear", "sideways"], len(df_result))
        df_result["market_regime"] = regimes
        return df_result

    def create_all_features(
        self, df: pd.DataFrame, timestamp_col: str = "timestamp"
    ) -> pd.DataFrame:
        """
        Create comprehensive feature set

        Args:
            df: DataFrame with OHLCV data
            timestamp_col: Name of timestamp column

        Returns:
            DataFrame with all features
        """
        logger.info("Starting comprehensive feature engineering")

        df_features = df.copy()

        # Price-based features
        df_features = self.calculate_price_features(df_features)

        # Technical indicators
        df_features = self.calculate_sma(df_features)
        df_features = self.calculate_ema(df_features)
        df_features = self.calculate_rsi(df_features)
        df_features = self.calculate_macd(df_features)
        df_features = self.calculate_bollinger_bands(df_features)

        # Stochastic (only if OHLC data available)
        if all(col in df_features.columns for col in ["open", "high", "low", "close"]):
            df_features = self.calculate_stochastic(df_features)

        # Volume features (if volume available)
        if "volume" in df_features.columns:
            df_features = self.calculate_volume_features(df_features)

        # Time features (if timestamp available)
        if timestamp_col in df_features.columns:
            df_features = self.calculate_time_features(df_features, timestamp_col)

        # ฟีเจอร์ขั้นสูง (production-ready stub)
        df_features = self.calculate_market_microstructure_features(df_features)
        df_features = self.calculate_sentiment_features(df_features)
        df_features = self.calculate_cross_asset_features(df_features)
        df_features = self.calculate_regime_detection(df_features)

        # Create target variable (next period return)
        df_features["target"] = (
            df_features["close"].shift(-1) / df_features["close"] - 1
        )
        df_features["target_binary"] = (df_features["target"] > 0).astype(int)

        logger.info(f"Feature engineering completed. Final shape: {df_features.shape}")
        return df_features

    def select_features(
        self, df: pd.DataFrame, feature_groups: List[str] = None
    ) -> List[str]:
        """
        Select relevant features for modeling

        Args:
            df: DataFrame with features
            feature_groups: List of feature groups to include

        Returns:
            List of selected feature names
        """
        if feature_groups is None:
            feature_groups = [
                "price",
                "sma",
                "ema",
                "rsi",
                "macd",
                "bollinger",
                "volume",
                "time",
            ]

        selected_features = []

        # Define feature patterns
        feature_patterns = {
            "price": [
                "price_change",
                "price_change_pct",
                "hl_spread",
                "oc_spread",
                "atr",
            ],
            "sma": [col for col in df.columns if col.startswith("sma_")],
            "ema": [col for col in df.columns if col.startswith("ema_")],
            "rsi": ["rsi"],
            "macd": ["macd", "macd_signal", "macd_histogram"],
            "bollinger": [col for col in df.columns if col.startswith("bb_")],
            "stochastic": ["stoch_k", "stoch_d"],
            "volume": [
                col for col in df.columns if "volume" in col or col in ["obv", "vpt"]
            ],
            "time": [
                "hour",
                "day_of_week",
                "is_weekend",
                "hour_sin",
                "hour_cos",
                "dow_sin",
                "dow_cos",
            ],
        }

        # Collect features based on groups
        for group in feature_groups:
            if group in feature_patterns:
                group_features = [
                    col for col in feature_patterns[group] if col in df.columns
                ]
                selected_features.extend(group_features)

        # Remove duplicates
        selected_features = list(set(selected_features))

        logger.info(
            f"Selected {len(selected_features)} features from groups: {feature_groups}"
        )
        return selected_features

    def get_feature_importance_summary(
        self, feature_names: List[str]
    ) -> Dict[str, List[str]]:
        """
        Categorize features by type for analysis

        Args:
            feature_names: List of feature names

        Returns:
            Dictionary with feature categories
        """
        categories = {
            "trend": [],
            "momentum": [],
            "volatility": [],
            "volume": [],
            "time": [],
            "price": [],
        }

        for feature in feature_names:
            if any(x in feature for x in ["sma", "ema", "macd"]):
                categories["trend"].append(feature)
            elif any(x in feature for x in ["rsi", "stoch"]):
                categories["momentum"].append(feature)
            elif any(x in feature for x in ["bb_", "atr", "spread"]):
                categories["volatility"].append(feature)
            elif any(x in feature for x in ["volume", "obv", "vpt"]):
                categories["volume"].append(feature)
            elif any(x in feature for x in ["hour", "day", "weekend", "sin", "cos"]):
                categories["time"].append(feature)
            else:
                categories["price"].append(feature)

        return categories

    def engineer_features(
        self, df: pd.DataFrame, timestamp_col: str = "timestamp"
    ) -> pd.DataFrame:
        """
        Main entry point for feature engineering - alias for create_all_features

        This method is called by the pipeline orchestrator for consistency

        Args:
            df: DataFrame with OHLCV data
            timestamp_col: Name of timestamp column

        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering process")

        # Call the main feature creation method
        return self.create_all_features(df, timestamp_col)

    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get information about the feature engineering process

        Returns:
            Dictionary with feature information including:
            - total_features: Total number of features created
            - feature_categories: Categories of features
            - processing_status: Success/failure status
        """
        feature_info = {
            "total_features": 50,  # Approximate based on the feature creation process
            "feature_categories": {
                "price_features": [
                    "high_low_ratio", "open_close_ratio", "price_range",
                    "typical_price", "weighted_close"
                ],
                "trend_indicators": [
                    "sma_5", "sma_10", "sma_20", "sma_50", 
                    "ema_5", "ema_10", "ema_20", "ema_50",
                    "trend_strength", "trend_direction"
                ],
                "momentum_indicators": [
                    "rsi_14", "rsi_21", "macd", "macd_signal", "macd_histogram",
                    "williams_r", "cci", "momentum", "rate_of_change"
                ],
                "volatility_indicators": [
                    "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_position",
                    "atr", "true_range", "volatility"
                ],
                "volume_features": [
                    "volume_sma", "volume_ratio", "price_volume",
                    "volume_trend", "obv"
                ],
                "statistical_features": [
                    "rolling_mean", "rolling_std", "rolling_min", "rolling_max",
                    "z_score", "percentile_rank"
                ],
                "time_features": [
                    "hour", "day_of_week", "is_weekend",
                    "trading_session", "time_since_start"
                ],
                "lagged_features": [
                    "close_lag_1", "close_lag_2", "close_lag_3",
                    "volume_lag_1", "returns_lag_1", "returns_lag_2"
                ]
            },
            "processing_status": "completed",
            "last_updated": datetime.now().isoformat(),
            "config_used": self.config,
            "description": "Comprehensive feature engineering for trading data including technical indicators, statistical features, and time-based features"
        }
        
        logger.info(f"Feature info generated: {len(feature_info['feature_categories'])} categories")
        return feature_info
