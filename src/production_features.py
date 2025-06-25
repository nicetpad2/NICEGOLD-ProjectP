# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import logging
import os
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import ta
from ta.momentum import (
    ROCIndicator,
    RSIIndicator,
    StochasticOscillator,
    WilliamsRIndicator,
)
from ta.trend import (
    MACD,
    ADXIndicator,
    CCIIndicator,
    EMAIndicator,
    PSARIndicator,
    SMAIndicator,
)
from ta.volatility import BollingerBands, KeltnerChannel
from ta.volume import (
    AccDistIndexIndicator,
    MFIIndicator,
    OnBalanceVolumeIndicator,
    VolumePriceTrendIndicator,
)

"""
Production-Ready Feature Engineering for NICEGOLD ProjectP
===========================================================

Complete technical analysis feature engineering system with robust error handling
and production-ready performance optimizations.

Author: NICEGOLD Team
Version: 3.0 Production
Created: 2025-06-24
"""

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class ProductionFeatureEngineer:
    """
    Production - ready feature engineering system for XAUUSD trading data.

    Features:
    - 50+ Technical Indicators
    - Robust error handling
    - Performance optimized
    - Data validation
    - Memory efficient
    """

    def __init__(
        self,
        validate_data: bool = True,
        optimize_memory: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize the feature engineering system.

        Args:
            validate_data: Enable data validation
            optimize_memory: Enable memory optimization
            verbose: Enable verbose logging
        """
        self.validate_data = validate_data
        self.optimize_memory = optimize_memory
        self.verbose = verbose

        # Feature configuration
        self.feature_config = {
            "price_features": True,
            "momentum_features": True,
            "trend_features": True,
            "volatility_features": True,
            "volume_features": True,
            "pattern_features": True,
            "time_features": True,
            "statistical_features": True,
        }

        # Technical indicator parameters
        self.params = {
            "rsi_periods": [14, 21],
            "sma_periods": [10, 20, 50, 200],
            "ema_periods": [12, 26, 50, 200],
            "macd_params": [(12, 26, 9)],
            "bb_periods": [20],
            "bb_std": [2.0],
            "stoch_periods": [14],
            "adx_periods": [14],
            "cci_periods": [20],
            "williams_periods": [14],
        }

    def validate_input_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and prepare input data for feature engineering.

        Args:
            df: Input DataFrame with OHLCV data

        Returns:
            Validated and prepared DataFrame

        Raises:
            ValueError: If data validation fails
        """
        if not self.validate_data:
            return df

        logger.info("Validating input data...")

        # Check required columns
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check data types
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"Converting {col} to numeric")
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Check for sufficient data
        if len(df) < 200:
            raise ValueError(
                f"Insufficient data: {len(df)} rows. Minimum 200 required."
            )

        # Check for data quality
        null_counts = df[required_columns].isnull().sum()
        if null_counts.sum() > 0:
            logger.warning(f"Found null values: {null_counts.to_dict()}")
            # Forward fill then backward fill
            df[required_columns] = (
                df[required_columns].fillna(method="ffill").fillna(method="bfill")
            )

        # Check for data consistency
        invalid_rows = (
            (df["High"] < df["Low"])
            | (df["High"] < df["Open"])
            | (df["High"] < df["Close"])
            | (df["Low"] > df["Open"])
            | (df["Low"] > df["Close"])
            | (df["Volume"] < 0)
        )

        if invalid_rows.sum() > 0:
            logger.warning(f"Found {invalid_rows.sum()} invalid rows. Fixing...")
            # Fix invalid OHLC data
            df.loc[invalid_rows, "High"] = df.loc[
                invalid_rows, ["Open", "High", "Low", "Close"]
            ].max(axis=1)
            df.loc[invalid_rows, "Low"] = df.loc[
                invalid_rows, ["Open", "High", "Low", "Close"]
            ].min(axis=1)
            df.loc[invalid_rows, "Volume"] = df.loc[invalid_rows, "Volume"].abs()

        logger.info(f"✅ Data validation completed. Shape: {df.shape}")
        return df

    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price - based features."""
        if not self.feature_config["price_features"]:
            return df

        logger.info("Creating price features...")

        try:
            # Basic price features
            df["hl_ratio"] = df["High"] / df["Low"]
            df["oc_ratio"] = df["Open"] / df["Close"]
            df["price_range"] = (df["High"] - df["Low"]) / df["Close"]
            df["body_size"] = abs(df["Close"] - df["Open"]) / df["Close"]
            df["upper_shadow"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / df[
                "Close"
            ]
            df["lower_shadow"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / df[
                "Close"
            ]

            # Price position features
            df["close_position"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"])
            df["open_position"] = (df["Open"] - df["Low"]) / (df["High"] - df["Low"])

            # Gap features
            df["gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
            df["gap_up"] = (df["gap"] > 0).astype(int)
            df["gap_down"] = (df["gap"] < 0).astype(int)

            # Typical price
            df["typical_price"] = (df["High"] + df["Low"] + df["Close"]) / 3
            df["weighted_price"] = (df["High"] + df["Low"] + 2 * df["Close"]) / 4

            logger.info("✅ Price features created")

        except Exception as e:
            logger.error(f"Error creating price features: {e}")

        return df

    def create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create momentum - based technical indicators."""
        if not self.feature_config["momentum_features"]:
            return df

        logger.info("Creating momentum features...")

        try:
            # RSI indicators
            for period in self.params["rsi_periods"]:
                rsi = RSIIndicator(close=df["Close"], window=period)
                df[f"rsi_{period}"] = rsi.rsi()

                # RSI levels
                df[f"rsi_{period}_overbought"] = (df[f"rsi_{period}"] > 70).astype(int)
                df[f"rsi_{period}_oversold"] = (df[f"rsi_{period}"] < 30).astype(int)

            # MACD
            for fast, slow, signal in self.params["macd_params"]:
                macd = MACD(
                    close=df["Close"],
                    window_slow=slow,
                    window_fast=fast,
                    window_sign=signal,
                )
                df[f"macd_{fast}_{slow}"] = macd.macd()
                df[f"macd_signal_{fast}_{slow}"] = macd.macd_signal()
                df[f"macd_histogram_{fast}_{slow}"] = macd.macd_diff()

                # MACD crossovers
                df[f"macd_bullish_{fast}_{slow}"] = (
                    (df[f"macd_{fast}_{slow}"] > df[f"macd_signal_{fast}_{slow}"])
                    & (
                        df[f"macd_{fast}_{slow}"].shift(1)
                        <= df[f"macd_signal_{fast}_{slow}"].shift(1)
                    )
                ).astype(int)

                df[f"macd_bearish_{fast}_{slow}"] = (
                    (df[f"macd_{fast}_{slow}"] < df[f"macd_signal_{fast}_{slow}"])
                    & (
                        df[f"macd_{fast}_{slow}"].shift(1)
                        >= df[f"macd_signal_{fast}_{slow}"].shift(1)
                    )
                ).astype(int)

            # Stochastic Oscillator
            for period in self.params["stoch_periods"]:
                stoch = StochasticOscillator(
                    high=df["High"], low=df["Low"], close=df["Close"], window=period
                )
                df[f"stoch_k_{period}"] = stoch.stoch()
                df[f"stoch_d_{period}"] = stoch.stoch_signal()

                # Stochastic levels
                df[f"stoch_overbought_{period}"] = (
                    df[f"stoch_k_{period}"] > 80
                ).astype(int)
                df[f"stoch_oversold_{period}"] = (df[f"stoch_k_{period}"] < 20).astype(
                    int
                )

            # Williams %R
            for period in self.params["williams_periods"]:
                williams = WilliamsRIndicator(
                    high=df["High"], low=df["Low"], close=df["Close"], lbp=period
                )
                df[f"williams_r_{period}"] = williams.williams_r()

            # Momentum
            df["momentum_1"] = df["Close"] / df["Close"].shift(1) - 1
            df["momentum_5"] = df["Close"] / df["Close"].shift(5) - 1
            df["momentum_10"] = df["Close"] / df["Close"].shift(10) - 1

            # Rate of Change
            df["roc_5"] = df["Close"].pct_change(5)
            df["roc_10"] = df["Close"].pct_change(10)
            df["roc_20"] = df["Close"].pct_change(20)

            logger.info("✅ Momentum features created")

        except Exception as e:
            logger.error(f"Error creating momentum features: {e}")

        return df

    def create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create trend - based technical indicators."""
        if not self.feature_config["trend_features"]:
            return df

        logger.info("Creating trend features...")

        try:
            # Simple Moving Averages
            for period in self.params["sma_periods"]:
                sma = SMAIndicator(close=df["Close"], window=period)
                df[f"sma_{period}"] = sma.sma_indicator()

                # Price vs SMA
                df[f"close_vs_sma_{period}"] = df["Close"] / df[f"sma_{period}"] - 1
                df[f"above_sma_{period}"] = (df["Close"] > df[f"sma_{period}"]).astype(
                    int
                )

            # Exponential Moving Averages
            for period in self.params["ema_periods"]:
                ema = EMAIndicator(close=df["Close"], window=period)
                df[f"ema_{period}"] = ema.ema_indicator()

                # Price vs EMA
                df[f"close_vs_ema_{period}"] = df["Close"] / df[f"ema_{period}"] - 1
                df[f"above_ema_{period}"] = (df["Close"] > df[f"ema_{period}"]).astype(
                    int
                )

            # Moving Average Crossovers
            df["golden_cross"] = (
                (df["ema_50"] > df["ema_200"])
                & (df["ema_50"].shift(1) <= df["ema_200"].shift(1))
            ).astype(int)

            df["death_cross"] = (
                (df["ema_50"] < df["ema_200"])
                & (df["ema_50"].shift(1) >= df["ema_200"].shift(1))
            ).astype(int)

            # ADX (Average Directional Index)
            for period in self.params["adx_periods"]:
                adx = ADXIndicator(
                    high=df["High"], low=df["Low"], close=df["Close"], window=period
                )
                df[f"adx_{period}"] = adx.adx()
                df[f"adx_pos_{period}"] = adx.adx_pos()
                df[f"adx_neg_{period}"] = adx.adx_neg()

                # ADX trend strength
                df[f"adx_strong_trend_{period}"] = (df[f"adx_{period}"] > 25).astype(
                    int
                )
                df[f"adx_weak_trend_{period}"] = (df[f"adx_{period}"] < 20).astype(int)

            # Commodity Channel Index
            for period in self.params["cci_periods"]:
                cci = CCIIndicator(
                    high=df["High"], low=df["Low"], close=df["Close"], window=period
                )
                df[f"cci_{period}"] = cci.cci()

                # CCI levels
                df[f"cci_overbought_{period}"] = (df[f"cci_{period}"] > 100).astype(int)
                df[f"cci_oversold_{period}"] = (df[f"cci_{period}"] < -100).astype(int)

            # Parabolic SAR
            psar = PSARIndicator(high=df["High"], low=df["Low"], close=df["Close"])
            df["psar"] = psar.psar()
            df["psar_bullish"] = (df["Close"] > df["psar"]).astype(int)

            logger.info("✅ Trend features created")

        except Exception as e:
            logger.error(f"Error creating trend features: {e}")

        return df

    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility - based technical indicators."""
        if not self.feature_config["volatility_features"]:
            return df

        logger.info("Creating volatility features...")

        try:
            # Bollinger Bands
            for period in self.params["bb_periods"]:
                for std in self.params["bb_std"]:
                    bb = BollingerBands(
                        close=df["Close"], window=period, window_dev=std
                    )
                    df[f"bb_upper_{period}_{std}"] = bb.bollinger_hband()
                    df[f"bb_middle_{period}_{std}"] = bb.bollinger_mavg()
                    df[f"bb_lower_{period}_{std}"] = bb.bollinger_lband()
                    df[f"bb_width_{period}_{std}"] = bb.bollinger_wband()
                    df[f"bb_percent_{period}_{std}"] = bb.bollinger_pband()

                    # BB position
                    df[f"bb_squeeze_{period}_{std}"] = (
                        df[f"bb_width_{period}_{std}"]
                        < df[f"bb_width_{period}_{std}"].rolling(20).mean()
                    ).astype(int)
                    df[f"bb_expansion_{period}_{std}"] = (
                        df[f"bb_width_{period}_{std}"]
                        > df[f"bb_width_{period}_{std}"].rolling(20).mean()
                    ).astype(int)

            # Keltner Channels
            kc = KeltnerChannel(high=df["High"], low=df["Low"], close=df["Close"])
            df["kc_upper"] = kc.keltner_channel_hband()
            df["kc_middle"] = kc.keltner_channel_mband()
            df["kc_lower"] = kc.keltner_channel_lband()
            df["kc_width"] = kc.keltner_channel_wband()
            df["kc_percent"] = kc.keltner_channel_pband()

            # Average True Range
            df["atr_14"] = ta.volatility.AverageTrueRange(
                high=df["High"], low=df["Low"], close=df["Close"], window=14
            ).average_true_range()
            df["atr_21"] = ta.volatility.AverageTrueRange(
                high=df["High"], low=df["Low"], close=df["Close"], window=21
            ).average_true_range()

            # Volatility measures
            df["volatility_5"] = df["Close"].rolling(5).std()
            df["volatility_10"] = df["Close"].rolling(10).std()
            df["volatility_20"] = df["Close"].rolling(20).std()

            # Historical volatility
            df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
            df["hist_vol_20"] = df["log_return"].rolling(20).std() * np.sqrt(252)
            df["hist_vol_50"] = df["log_return"].rolling(50).std() * np.sqrt(252)

            logger.info("✅ Volatility features created")

        except Exception as e:
            logger.error(f"Error creating volatility features: {e}")

        return df

    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume - based technical indicators."""
        if not self.feature_config["volume_features"]:
            return df

        logger.info("Creating volume features...")

        try:
            # Volume moving averages
            df["volume_sma_10"] = df["Volume"].rolling(10).mean()
            df["volume_sma_20"] = df["Volume"].rolling(20).mean()
            df["volume_ratio"] = df["Volume"] / df["volume_sma_20"]

            # Volume indicators
            df["high_volume"] = (df["Volume"] > df["volume_sma_20"] * 1.5).astype(int)
            df["low_volume"] = (df["Volume"] < df["volume_sma_20"] * 0.5).astype(int)

            # On Balance Volume
            obv = OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"])
            df["obv"] = obv.on_balance_volume()
            df["obv_sma"] = df["obv"].rolling(20).mean()

            # Accumulation/Distribution Index
            ad = AccDistIndexIndicator(
                high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"]
            )
            df["ad_index"] = ad.acc_dist_index()

            # Chaikin Money Flow
            cmf = ChaikinMoneyFlowIndicator(
                high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"]
            )
            df["cmf"] = cmf.chaikin_money_flow()

            # Price Volume Trend
            df["pvt"] = ta.volume.VolumePriceTrendIndicator(
                close=df["Close"], volume=df["Volume"]
            ).volume_price_trend()

            # Volume Rate of Change
            df["volume_roc"] = df["Volume"].pct_change(5)

            logger.info("✅ Volume features created")

        except Exception as e:
            logger.error(f"Error creating volume features: {e}")

        return df

    def create_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create candlestick pattern features."""
        if not self.feature_config["pattern_features"]:
            return df

        logger.info("Creating pattern features...")

        try:
            # Basic candlestick patterns
            df["doji"] = (
                abs(df["Close"] - df["Open"]) <= (df["High"] - df["Low"]) * 0.1
            ).astype(int)
            df["hammer"] = (
                (df["lower_shadow"] > 2 * df["body_size"])
                & (df["upper_shadow"] < df["body_size"])
            ).astype(int)

            df["shooting_star"] = (
                (df["upper_shadow"] > 2 * df["body_size"])
                & (df["lower_shadow"] < df["body_size"])
            ).astype(int)

            # Bullish/Bearish candles
            df["bullish_candle"] = (df["Close"] > df["Open"]).astype(int)
            df["bearish_candle"] = (df["Close"] < df["Open"]).astype(int)

            # Consecutive patterns
            df["consecutive_up"] = df["bullish_candle"] * (
                df["bullish_candle"]
                .groupby((df["bullish_candle"] == 0).cumsum())
                .cumcount()
                + 1
            )
            df["consecutive_down"] = df["bearish_candle"] * (
                df["bearish_candle"]
                .groupby((df["bearish_candle"] == 0).cumsum())
                .cumcount()
                + 1
            )

            # Price action patterns
            df["higher_high"] = (df["High"] > df["High"].shift(1)).astype(int)
            df["lower_low"] = (df["Low"] < df["Low"].shift(1)).astype(int)
            df["higher_low"] = (df["Low"] > df["Low"].shift(1)).astype(int)
            df["lower_high"] = (df["High"] < df["High"].shift(1)).astype(int)

            # Inside/Outside bars
            df["inside_bar"] = (
                (df["High"] < df["High"].shift(1)) & (df["Low"] > df["Low"].shift(1))
            ).astype(int)

            df["outside_bar"] = (
                (df["High"] > df["High"].shift(1)) & (df["Low"] < df["Low"].shift(1))
            ).astype(int)

            logger.info("✅ Pattern features created")

        except Exception as e:
            logger.error(f"Error creating pattern features: {e}")

        return df

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time - based features."""
        if not self.feature_config["time_features"]:
            return df

        logger.info("Creating time features...")

        try:
            # Ensure we have a datetime column
            if "Time" in df.columns:
                df["datetime"] = pd.to_datetime(df["Time"])
            elif df.index.name == "datetime" or pd.api.types.is_datetime64_any_dtype(
                df.index
            ):
                df["datetime"] = df.index
            else:
                logger.warning("No datetime column found. Skipping time features.")
                return df

            # Extract time components
            df["hour"] = df["datetime"].dt.hour
            df["day_of_week"] = df["datetime"].dt.dayofweek
            df["day_of_month"] = df["datetime"].dt.day
            df["month"] = df["datetime"].dt.month
            df["quarter"] = df["datetime"].dt.quarter

            # Trading session features (assuming UTC timezone)
            df["asian_session"] = ((df["hour"] >= 0) & (df["hour"] < 8)).astype(int)
            df["london_session"] = ((df["hour"] >= 8) & (df["hour"] < 16)).astype(int)
            df["new_york_session"] = ((df["hour"] >= 13) & (df["hour"] < 21)).astype(
                int
            )
            df["overlap_london_ny"] = ((df["hour"] >= 13) & (df["hour"] < 16)).astype(
                int
            )

            # Weekend/Weekday
            df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
            df["is_monday"] = (df["day_of_week"] == 0).astype(int)
            df["is_friday"] = (df["day_of_week"] == 4).astype(int)

            # Month end/start
            df["month_start"] = (df["day_of_month"] <= 5).astype(int)
            df["month_end"] = (df["day_of_month"] >= 25).astype(int)

            # Remove the datetime column to prevent issues with ML training
            if "datetime" in df.columns:
                df = df.drop(columns=["datetime"])

            logger.info("✅ Time features created")

        except Exception as e:
            logger.error(f"Error creating time features: {e}")

        return df

    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features."""
        if not self.feature_config["statistical_features"]:
            return df

        logger.info("Creating statistical features...")

        try:
            # Rolling statistics
            for window in [5, 10, 20, 50]:
                df[f"mean_{window}"] = df["Close"].rolling(window).mean()
                df[f"std_{window}"] = df["Close"].rolling(window).std()
                df[f"min_{window}"] = df["Close"].rolling(window).min()
                df[f"max_{window}"] = df["Close"].rolling(window).max()
                df[f"median_{window}"] = df["Close"].rolling(window).median()

                # Percentile features
                df[f"price_percentile_{window}"] = (
                    df["Close"].rolling(window).rank(pct=True)
                )

                # Z - score
                df[f"zscore_{window}"] = (df["Close"] - df[f"mean_{window}"]) / df[
                    f"std_{window}"
                ]

            # Skewness and Kurtosis
            df["skewness_20"] = df["Close"].rolling(20).skew()
            df["kurtosis_20"] = df["Close"].rolling(20).kurt()

            # Distance from extremes
            df["distance_from_high_20"] = (df["max_20"] - df["Close"]) / df["Close"]
            df["distance_from_low_20"] = (df["Close"] - df["min_20"]) / df["Close"]

            # Linear regression features
            for window in [10, 20]:
                x = np.arange(window)
                slopes = []
                r_squared = []

                for i in range(window, len(df)):
                    y = df["Close"].iloc[i - window : i].values
                    if len(y) == window and not np.isnan(y).any():
                        try:
                            coeffs = np.polyfit(x, y, 1)
                            slope = coeffs[0]
                            y_pred = np.polyval(coeffs, x)
                            ss_res = np.sum((y - y_pred) ** 2)
                            ss_tot = np.sum((y - np.mean(y)) ** 2)
                            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                        except:
                            slope = np.nan
                            r2 = np.nan
                    else:
                        slope = np.nan
                        r2 = np.nan

                    slopes.append(slope)
                    r_squared.append(r2)

                # Pad with NaN for initial values
                slopes = [np.nan] * window + slopes
                r_squared = [np.nan] * window + r_squared

                df[f"slope_{window}"] = slopes
                df[f"r_squared_{window}"] = r_squared

            logger.info("✅ Statistical features created")

        except Exception as e:
            logger.error(f"Error creating statistical features: {e}")

        return df

    def create_target_variable(
        self, df: pd.DataFrame, method: str = "future_return"
    ) -> pd.DataFrame:
        """
        Create target variable for machine learning.

        Args:
            df: DataFrame with features
            method: Method for target creation ('future_return', 'threshold', 'trend')

        Returns:
            DataFrame with target variable
        """
        logger.info(f"Creating target variable using method: {method}")

        try:
            if method == "future_return":
                # Future return target
                df["future_return"] = df["Close"].shift(-1) / df["Close"] - 1
                df["target"] = (df["future_return"] > 0).astype(int)

            elif method == "threshold":
                # Threshold - based target (0.1% movement)
                threshold = 0.001
                df["future_return"] = df["Close"].shift(-1) / df["Close"] - 1
                df["target"] = np.where(
                    df["future_return"] > threshold,
                    1,
                    np.where(df["future_return"] < -threshold, 0, np.nan),
                )

            elif method == "trend":
                # Multi - period trend target
                df["future_5"] = df["Close"].shift(-5) / df["Close"] - 1
                df["target"] = (df["future_5"] > 0).astype(int)

            else:
                raise ValueError(f"Unknown target method: {method}")

            # Remove rows with NaN targets (last few rows)
            df = df.dropna(subset=["target"])

            logger.info(
                f"✅ Target variable created. Target distribution: {df['target'].value_counts().to_dict()}"
            )

        except Exception as e:
            logger.error(f"Error creating target variable: {e}")

        return df

    def optimize_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize memory usage by converting data types."""
        if not self.optimize_memory:
            return df

        logger.info("Optimizing memory usage...")

        initial_memory = df.memory_usage(deep=True).sum() / 1024**2

        # Convert float64 to float32 where possible
        float_cols = df.select_dtypes(include=["float64"]).columns
        for col in float_cols:
            if df[col].dtype == "float64":
                df[col] = pd.to_numeric(df[col], downcast="float")

        # Convert int64 to smaller int types where possible
        int_cols = df.select_dtypes(include=["int64"]).columns
        for col in int_cols:
            if df[col].dtype == "int64":
                df[col] = pd.to_numeric(df[col], downcast="integer")

        final_memory = df.memory_usage(deep=True).sum() / 1024**2
        reduction = (initial_memory - final_memory) / initial_memory * 100

        logger.info(
            f"✅ Memory optimization completed. Reduced by {reduction:.1f}% ({initial_memory:.1f}MB → {final_memory:.1f}MB)"
        )

        return df

    def engineer_features(
        self, df: pd.DataFrame, target_method: str = "future_return"
    ) -> pd.DataFrame:
        """
        Main method to engineer all features.

        Args:
            df: Input DataFrame with OHLCV data
            target_method: Method for target variable creation

        Returns:
            DataFrame with engineered features
        """
        logger.info("🚀 Starting feature engineering...")

        # Validate input data
        df = self.validate_input_data(df)

        # Create features
        df = self.create_price_features(df)
        df = self.create_momentum_features(df)
        df = self.create_trend_features(df)
        df = self.create_volatility_features(df)
        df = self.create_volume_features(df)
        df = self.create_pattern_features(df)
        df = self.create_time_features(df)
        df = self.create_statistical_features(df)

        # Create target variable
        df = self.create_target_variable(df, method=target_method)

        # Store column names for feature summary
        self.last_engineered_columns = list(df.columns)

        # Optimize memory
        df = self.optimize_memory_usage(df)

        # Final cleanup
        df = df.replace([np.inf, -np.inf], np.nan)

        # Get feature statistics
        feature_count = len(
            [
                col
                for col in df.columns
                if col
                not in [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                    "Time",
                    "datetime",
                    "target",
                ]
            ]
        )

        logger.info(f"✅ Feature engineering completed!")
        logger.info(f"📊 Final dataset shape: {df.shape}")
        logger.info(f"🎯 Features created: {feature_count}")
        logger.info(f"🎯 Target balance: {df['target'].value_counts().to_dict()}")

        return df

    def get_feature_summary(self) -> Dict[str, int]:
        """
        Get a summary of feature categories and counts.

        Returns:
            Dictionary with feature category counts
        """
        # Define feature patterns for categorization
        feature_patterns = {
            "price": [
                "hl_ratio",
                "oc_ratio",
                "price_range",
                "body_size",
                "shadow",
                "position",
                "gap",
            ],
            "momentum": ["rsi", "stoch", "williams", "momentum", "rate_of_change"],
            "trend": ["sma", "ema", "macd", "adx", "cci", "slope", "trend"],
            "volatility": ["bb", "atr", "volatility", "std", "range"],
            "volume": ["volume", "vwap", "obv", "mfi", "vpt"],
            "pattern": [
                "doji",
                "hammer",
                "engulfing",
                "star",
                "harami",
                "piercing",
                "pattern",
            ],
            "time": ["hour", "day", "week", "month", "quarter", "year", "season"],
            "statistical": [
                "skew",
                "kurt",
                "correlation",
                "z_score",
                "percentile",
                "lag",
                "diff",
            ],
        }

        # Get all current columns (if features have been created)
        if hasattr(self, "last_engineered_columns"):
            columns = list(self.last_engineered_columns)
        else:
            # Return default counts based on configuration if no features created yet
            return {
                "price": 8 if self.feature_config.get("price_features", True) else 0,
                "momentum": (
                    6 if self.feature_config.get("momentum_features", True) else 0
                ),
                "trend": 12 if self.feature_config.get("trend_features", True) else 0,
                "volatility": (
                    8 if self.feature_config.get("volatility_features", True) else 0
                ),
                "volume": 6 if self.feature_config.get("volume_features", True) else 0,
                "pattern": (
                    8 if self.feature_config.get("pattern_features", True) else 0
                ),
                "time": 6 if self.feature_config.get("time_features", True) else 0,
                "statistical": (
                    10 if self.feature_config.get("statistical_features", True) else 0
                ),
                "other": 0,
            }

        # Initialize counts
        feature_counts = {category: 0 for category in feature_patterns.keys()}
        feature_counts["other"] = 0

        # Categorize features
        base_columns = {
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Time",
            "datetime",
            "target",
        }

        for col in columns:
            if col in base_columns:
                continue

            categorized = False
            for category, patterns in feature_patterns.items():
                if any(pattern in col.lower() for pattern in patterns):
                    feature_counts[category] += 1
                    categorized = True
                    break

            if not categorized:
                feature_counts["other"] += 1

        return feature_counts

    # ...existing code...


def engineer_features_production(
    df: pd.DataFrame,
    config: Optional[Dict] = None,
    target_method: str = "future_return",
) -> pd.DataFrame:
    """
    Production wrapper function for feature engineering.

    Args:
        df: Input DataFrame with OHLCV data
        config: Configuration dictionary
        target_method: Method for target variable creation

    Returns:
        DataFrame with engineered features
    """
    # Initialize feature engineer
    engineer = ProductionFeatureEngineer(
        validate_data=config.get("validate_data", True) if config else True,
        optimize_memory=config.get("optimize_memory", True) if config else True,
        verbose=config.get("verbose", True) if config else True,
    )

    # Update configuration if provided
    if config:
        if "feature_config" in config:
            engineer.feature_config.update(config["feature_config"])
        if "params" in config:
            engineer.params.update(config["params"])

    # Engineer features
    return engineer.engineer_features(df, target_method=target_method)


if __name__ == "__main__":
    # Test with sample data

    logger.info("Testing Production Feature Engineering...")

    # Load sample data
    if os.path.exists("datacsv/XAUUSD_M1.csv"):
        logger.info("Loading real data for testing...")
        df = pd.read_csv("datacsv/XAUUSD_M1.csv", nrows=10000)
    else:
        logger.info("Creating sample data for testing...")
        # Create sample OHLCV data
        np.random.seed(42)
        dates = pd.date_range("2023 - 01 - 01", periods=1000, freq="1H")

        df = pd.DataFrame(
            {
                "Time": dates,
                "Open": np.random.uniform(1800, 2000, 1000),
                "High": np.random.uniform(1810, 2010, 1000),
                "Low": np.random.uniform(1790, 1990, 1000),
                "Close": np.random.uniform(1800, 2000, 1000),
                "Volume": np.random.uniform(100, 1000, 1000),
            }
        )

        # Ensure OHLC consistency
        df["High"] = df[["Open", "High", "Low", "Close"]].max(axis=1)
        df["Low"] = df[["Open", "High", "Low", "Close"]].min(axis=1)

    # Engineer features
    df_features = engineer_features_production(df)

    logger.info(f"✅ Test completed. Final shape: {df_features.shape}")
    logger.info(f"📊 Feature columns: {len(df_features.columns)}")
    logger.info(
        f"🎯 Target distribution: {df_features['target'].value_counts().to_dict()}"
    )
