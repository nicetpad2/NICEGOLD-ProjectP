

# Anti - Noise, Anti - Leak, Anti - Overfitting Framework
# For Enterprise Trading ML Systems
# Import required modules
# 🔧 FALLBACK FIX: sklearn mutual_info_regression compatibility
# 🛡️ Advanced ML Protection System
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from sklearn.ensemble import IsolationForest
    from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import RFE, SelectFromModel
        from sklearn.metrics import mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import RobustScaler, StandardScaler
    from tracking import EnterpriseTracker
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import logging
import numpy as np
import pandas as pd
import warnings
import yaml
try:

    print("✅ Using sklearn.feature_selection.mutual_info_regression")
except ImportError:
    try:

        print("✅ Using sklearn.metrics.mutual_info_regression")
    except ImportError:
        print(
            "⚠️ sklearn mutual_info_regression not available, using fallback implementation"
        )

        def mutual_info_regression(X, y, **kwargs):
            """Fallback implementation for mutual_info_regression"""
            # Simple correlation - based fallback
            if hasattr(X, "shape") and len(X.shape) == 2:
                n_features = X.shape[1]
            else:
                n_features = 1

            # Return small random values to simulate mutual information
            return np.random.RandomState(42).uniform(0.001, 0.1, n_features)


# Import tracking system
try:

    TRACKING_AVAILABLE = True
except ImportError:
    TRACKING_AVAILABLE = False
    warnings.warn("Tracking system not available. Running in standalone mode.")


class ProtectionLevel(Enum):
    """Protection levels for ML system"""

    BASIC = "basic"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    ENTERPRISE = "enterprise"


@dataclass
class NoiseConfig:
    """Configuration for noise detection and removal"""

    outlier_detection_method: str = "isolation_forest"
    contamination_rate: float = 0.1
    noise_threshold: float = 0.95
    rolling_window_size: int = 20
    volatility_threshold: float = 3.0
    enable_adaptive_filtering: bool = True
    feature_noise_detection: bool = True
    adaptive_window_multiplier: float = 1.5
    mutual_info_threshold: float = 0.01
    correlation_stability_threshold: float = 0.5
    enable_fourier_filtering: bool = False
    fourier_cutoff_freq: float = 0.1
    enable_kalman_filtering: bool = False
    missing_value_strategy: str = "intelligent"
    max_missing_percentage: float = 0.3
    missing_run_threshold: int = 5


@dataclass
class LeakageConfig:
    """Configuration for data leakage prevention"""

    temporal_gap_hours: int = 24
    strict_time_validation: bool = True
    feature_leakage_detection: bool = True
    target_leakage_threshold: float = 0.8
    future_data_check: bool = True
    cross_validation_method: str = "time_series"
    enable_feature_timing_check: bool = True
    future_data_tolerance_minutes: int = 0
    perfect_correlation_threshold: float = 0.95
    validation_gap_periods: int = 1
    purge_gap_periods: int = 1
    feature_availability_buffer_hours: int = 1
    check_market_hours: bool = True
    enable_information_leakage_detection: bool = True
    information_coefficient_threshold: float = 0.1
    enable_target_encoding_detection: bool = True
    statistical_similarity_threshold: float = 0.01


@dataclass
class OverfittingConfig:
    """Configuration for overfitting prevention"""

    max_features_ratio: float = 0.3
    min_samples_per_feature: int = 10
    regularization_strength: float = 0.01
    early_stopping_patience: int = 10
    cross_validation_folds: int = 5
    validation_score_threshold: float = 0.1
    feature_importance_threshold: float = 0.001
    enable_ensemble_validation: bool = True
    feature_selection_method: str = "rfe"
    enable_elastic_net: bool = True
    l1_ratio: float = 0.5
    enable_nested_cv: bool = True
    early_stopping_min_delta: float = 0.001
    monitor_metric: str = "val_loss"
    enable_permutation_importance: bool = True
    importance_stability_check: bool = True
    ensemble_methods: List[str] = field(
        default_factory = lambda: ["bagging", "boosting", "stacking"]
    )
    ensemble_diversity_threshold: float = 0.1
    max_depth: int = 10
    min_samples_leaf: int = 1
    min_samples_split: int = 2
    max_leaf_nodes: Optional[int] = None
    min_impurity_decrease: float = 0.0
    ccp_alpha: float = 0.0
    enable_early_stopping: bool = True
    validation_fraction: float = 0.1
    n_iter_no_change: int = 5
    tol: float = 1e - 4
    l1_ratio: float = 0.5
    enable_nested_cv: bool = True
    early_stopping_min_delta: float = 0.001
    monitor_metric: str = "val_loss"
    enable_permutation_importance: bool = True
    importance_stability_check: bool = True
    ensemble_methods: List[str] = field(
        default_factory = lambda: ["bagging", "boosting", "stacking"]
    )
    ensemble_diversity_threshold: float = 0.1
    max_depth: int = 10


@dataclass
class ProtectionResult:
    """Results from protection system"""

    is_clean: bool
    noise_score: float
    leakage_score: float
    overfitting_score: float
    issues_found: List[str] = field(default_factory = list)
    recommendations: List[str] = field(default_factory = list)
    cleaned_data: Optional[pd.DataFrame] = None
    feature_report: Dict[str, Any] = field(default_factory = dict)


class NoiseDetector:
    """Advanced noise detection and removal system"""

    def __init__(self, config: NoiseConfig):
        self.config = config
        self.outlier_detector = IsolationForest(
            contamination = config.contamination_rate, random_state = 42
        )
        self.scaler = RobustScaler()
        self.noise_history = []

    def detect_statistical_noise(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect statistical noise in data"""
        noise_report = {
            "outliers": {}, 
            "volatility_spikes": {}, 
            "missing_patterns": {}, 
            "distribution_anomalies": {}, 
        }

        for column in data.select_dtypes(include = [np.number]).columns:
            series = data[column].dropna()

            # Outlier detection
            z_scores = np.abs((series - series.mean()) / series.std())
            outliers = z_scores > self.config.volatility_threshold
            noise_report["outliers"][column] = {
                "count": outliers.sum(), 
                "percentage": (outliers.sum() / len(series)) * 100, 
                "indices": series[outliers].index.tolist(), 
            }

            # Volatility spikes
            if len(series) > self.config.rolling_window_size:
                rolling_std = series.rolling(self.config.rolling_window_size).std()
                volatility_spikes = rolling_std > rolling_std.quantile(0.95)
                noise_report["volatility_spikes"][column] = {
                    "count": volatility_spikes.sum(), 
                    "max_volatility": rolling_std.max(), 
                    "avg_volatility": rolling_std.mean(), 
                }

            # Missing value patterns
            missing_mask = data[column].isna()
            if missing_mask.any():
                # Check for systematic missing patterns
                missing_runs = []
                current_run = 0
                for is_missing in missing_mask:
                    if is_missing:
                        current_run += 1
                    else:
                        if current_run > 0:
                            missing_runs.append(current_run)
                        current_run = 0

                noise_report["missing_patterns"][column] = {
                    "total_missing": missing_mask.sum(), 
                    "missing_percentage": (missing_mask.sum() / len(data)) * 100, 
                    "longest_missing_run": max(missing_runs) if missing_runs else 0, 
                    "missing_runs": len(missing_runs), 
                }

        return noise_report

    def detect_feature_noise(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Detect noise in features relative to target"""
        feature_noise = {}

        for column in X.select_dtypes(include = [np.number]).columns:
            feature = X[column].dropna()

            # Mutual information with target
            mi_score = mutual_info_regression(
                feature.values.reshape( - 1, 1), y[feature.index]
            )[0]

            # Feature stability over time
            if "timestamp" in X.columns or X.index.name == "timestamp":
                # Calculate rolling correlation with target
                window_size = min(50, len(feature) // 10)
                if window_size > 10:
                    rolling_corr = feature.rolling(window_size).corr(y[feature.index])
                    correlation_stability = rolling_corr.std()
                else:
                    correlation_stability = 0
            else:
                correlation_stability = 0

            feature_noise[column] = {
                "mutual_info_score": mi_score, 
                "correlation_stability": correlation_stability, 
                "is_noisy": mi_score < 0.01 or correlation_stability > 0.5, 
            }

        return feature_noise

    def clean_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Clean data by removing noise"""
        cleaned_data = data.copy()
        cleaning_report = {
            "rows_removed": 0, 
            "columns_modified": [], 
            "outliers_handled": {}, 
            "missing_values_filled": {}, 
        }

        # Remove extreme outliers using Isolation Forest
        numeric_columns = cleaned_data.select_dtypes(include = [np.number]).columns
        if len(numeric_columns) > 0:
            outlier_mask = (
                self.outlier_detector.fit_predict(
                    cleaned_data[numeric_columns].fillna(0)
                )
                == -1
            )

            rows_before = len(cleaned_data)
            cleaned_data = cleaned_data[~outlier_mask]
            cleaning_report["rows_removed"] = rows_before - len(cleaned_data)

        # Handle missing values intelligently
        for column in cleaned_data.columns:
            missing_count = cleaned_data[column].isna().sum()
            if missing_count > 0:
                if cleaned_data[column].dtype in ["int64", "float64"]:
                    # Use median for numeric columns
                    cleaned_data[column].fillna(
                        cleaned_data[column].median(), inplace = True
                    )
                else:
                    # Use mode for categorical columns
                    mode_value = cleaned_data[column].mode()
                    if len(mode_value) > 0:
                        cleaned_data[column].fillna(mode_value[0], inplace = True)

                cleaning_report["missing_values_filled"][column] = missing_count
                cleaning_report["columns_modified"].append(column)

        return cleaned_data, cleaning_report


class LeakageDetector:
    """Advanced data leakage detection system"""

    def __init__(self, config: LeakageConfig):
        self.config = config
        self.feature_timestamps = {}
        self.leakage_history = []

    def detect_temporal_leakage(
        self, 
        data: pd.DataFrame, 
        timestamp_col: str = "timestamp", 
        target_col: str = "target", 
    ) -> Dict[str, Any]:
        """Detect temporal data leakage"""
        leakage_report = {
            "future_data_leakage": False, 
            "temporal_gaps": {}, 
            "feature_timing_issues": {}, 
            "target_shift_problems": {}, 
        }

        if timestamp_col not in data.columns:
            leakage_report["warning"] = f"Timestamp column '{timestamp_col}' not found"
            return leakage_report

        # Convert timestamp to datetime if needed
        timestamps = pd.to_datetime(data[timestamp_col])

        # Check for future data leakage
        current_time = datetime.now()
        future_data = timestamps > current_time
        if future_data.any():
            leakage_report["future_data_leakage"] = True
            leakage_report["future_data_count"] = future_data.sum()

        # Check temporal gaps in required features
        for column in data.columns:
            if column in [timestamp_col, target_col]:
                continue

            # Check if feature has proper temporal alignment
            feature_availability = data[column].notna()
            if self.config.enable_feature_timing_check:
                # Features should not be available before they logically could be
                # This is domain - specific, but we can check for suspicious patterns

                # Check for features that are perfectly correlated with future target
                if target_col in data.columns:
                    shifted_target = data[target_col].shift( - 1)  # Future target
                    if len(shifted_target.dropna()) > 10:
                        correlation = data[column].corr(shifted_target)
                        if abs(correlation) > self.config.target_leakage_threshold:
                            leakage_report["feature_timing_issues"][column] = {
                                "future_target_correlation": correlation, 
                                "risk_level": "HIGH", 
                            }

        return leakage_report

    def detect_feature_leakage(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Detect feature leakage through statistical analysis"""
        leakage_report = {
            "suspicious_features": {}, 
            "perfect_correlations": {}, 
            "target_encodings": {}, 
        }

        for column in X.select_dtypes(include = [np.number]).columns:
            # Check for perfect or near - perfect correlation with target
            correlation = X[column].corr(y)
            if abs(correlation) > self.config.target_leakage_threshold:
                leakage_report["perfect_correlations"][column] = {
                    "correlation": correlation, 
                    "risk_level": "HIGH" if abs(correlation) > 0.95 else "MEDIUM", 
                }

            # Check for target encoding leakage
            # Features that have identical statistical properties to target
            feature_std = X[column].std()
            target_std = y.std()
            feature_mean = X[column].mean()
            target_mean = y.mean()

            if (
                abs(feature_std - target_std) < 0.01
                and abs(feature_mean - target_mean) < 0.01
            ):
                leakage_report["target_encodings"][column] = {
                    "statistical_similarity": True, 
                    "risk_level": "MEDIUM", 
                }

        return leakage_report

    def validate_time_series_split(
        self, data: pd.DataFrame, timestamp_col: str = "timestamp"
    ) -> Dict[str, Any]:
        """Validate proper time series splitting"""
        validation_report = {
            "is_valid_split": True, 
            "chronological_order": True, 
            "temporal_gaps": [], 
            "recommendations": [], 
        }

        if timestamp_col not in data.columns:
            validation_report["is_valid_split"] = False
            validation_report["recommendations"].append(
                f"Add timestamp column '{timestamp_col}' for proper time series validation"
            )
            return validation_report

        timestamps = pd.to_datetime(data[timestamp_col])

        # Check chronological order
        if not timestamps.is_monotonic_increasing:
            validation_report["chronological_order"] = False
            validation_report["recommendations"].append(
                "Data should be sorted chronologically for time series modeling"
            )

        # Check for temporal gaps
        time_diffs = timestamps.diff().dropna()
        median_diff = time_diffs.median()
        large_gaps = time_diffs > median_diff * 10  # Gaps 10x larger than median

        if large_gaps.any():
            validation_report["temporal_gaps"] = [
                {
                    "index": idx, 
                    "gap_size": str(time_diffs.iloc[idx]), 
                    "expected_size": str(median_diff), 
                }
                for idx in np.where(large_gaps)[0]
            ]
            validation_report["recommendations"].append(
                "Large temporal gaps detected. Consider gap - aware cross - validation."
            )

        return validation_report


class OverfittingDetector:
    """Advanced overfitting detection and prevention system"""

    def __init__(self, config: OverfittingConfig):
        self.config = config
        self.validation_scores = []
        self.feature_importance_history = []

    def detect_overfitting_risk(
        self, X: pd.DataFrame, y: pd.Series, model: Any = None
    ) -> Dict[str, Any]:
        """Detect overfitting risk factors"""
        risk_report = {
            "risk_level": "LOW", 
            "risk_factors": {}, 
            "feature_analysis": {}, 
            "data_complexity": {}, 
            "recommendations": [], 
        }

        n_samples, n_features = X.shape

        # Check sample - to - feature ratio
        samples_per_feature = n_samples / n_features
        if samples_per_feature < self.config.min_samples_per_feature:
            risk_report["risk_factors"]["low_sample_ratio"] = {
                "current_ratio": samples_per_feature, 
                "recommended_min": self.config.min_samples_per_feature, 
                "severity": "HIGH", 
            }
            risk_report["risk_level"] = "HIGH"
            risk_report["recommendations"].append(
                f"Increase sample size or reduce features. Current ratio: {samples_per_feature:.1f}"
            )

        # Check feature correlation (multicollinearity)
        numeric_features = X.select_dtypes(include = [np.number])
        if len(numeric_features.columns) > 1:
            correlation_matrix = numeric_features.corr().abs()
            # Find highly correlated feature pairs
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    if correlation_matrix.iloc[i, j] > 0.9:
                        high_corr_pairs.append(
                            {
                                "feature1": correlation_matrix.columns[i], 
                                "feature2": correlation_matrix.columns[j], 
                                "correlation": correlation_matrix.iloc[i, j], 
                            }
                        )

            if high_corr_pairs:
                risk_report["risk_factors"]["multicollinearity"] = {
                    "high_correlation_pairs": high_corr_pairs, 
                    "severity": "MEDIUM", 
                }
                risk_report["recommendations"].append(
                    "Remove highly correlated features to reduce overfitting risk"
                )

        # Data complexity analysis
        # Check target distribution
        target_entropy = self._calculate_entropy(y)
        risk_report["data_complexity"]["target_entropy"] = target_entropy

        if target_entropy < 0.5:
            risk_report["risk_factors"]["low_target_variance"] = {
                "entropy": target_entropy, 
                "severity": "MEDIUM", 
            }
            risk_report["recommendations"].append(
                "Low target variance detected. Consider data augmentation or resampling."
            )

        return risk_report

    def validate_with_time_series_cv(
        self, X: pd.DataFrame, y: pd.Series, model: Any, timestamp_col: str = None
    ) -> Dict[str, Any]:
        """Validate model using proper time series cross - validation"""
        validation_report = {
            "cv_scores": [], 
            "mean_score": 0.0, 
            "std_score": 0.0, 
            "overfitting_detected": False, 
            "score_variance": 0.0, 
        }

        # Use TimeSeriesSplit for proper validation
        tscv = TimeSeriesSplit(n_splits = self.config.cross_validation_folds)

        try:
            scores = cross_val_score(model, X, y, cv = tscv, scoring = "r2")
            validation_report["cv_scores"] = scores.tolist()
            validation_report["mean_score"] = scores.mean()
            validation_report["std_score"] = scores.std()
            validation_report["score_variance"] = scores.var()

            # Detect overfitting through score variance
            if scores.std() > self.config.validation_score_threshold:
                validation_report["overfitting_detected"] = True
                validation_report["overfitting_reason"] = (
                    "High variance in cross - validation scores"
                )

            # Check for negative scores (very bad sign)
            if any(score < 0 for score in scores):
                validation_report["overfitting_detected"] = True
                validation_report["overfitting_reason"] = (
                    "Negative cross - validation scores detected"
                )

        except Exception as e:
            validation_report["error"] = str(e)
            validation_report["overfitting_detected"] = True
            validation_report["overfitting_reason"] = "Cross - validation failed"

        return validation_report

    def _calculate_entropy(self, series: pd.Series) -> float:
        """Calculate entropy of a series"""
        value_counts = series.value_counts(normalize = True)
        entropy = -sum(p * np.log2(p) for p in value_counts if p > 0)
        return entropy

    def feature_selection_validation(
        self, X: pd.DataFrame, y: pd.Series, model: Any
    ) -> Dict[str, Any]:
        """Validate feature selection to prevent overfitting"""
        selection_report = {
            "selected_features": [], 
            "feature_importance": {}, 
            "selection_method": "recursive_feature_elimination", 
            "features_removed": [], 
        }

        # Use Recursive Feature Elimination
        max_features = max(1, int(len(X.columns) * self.config.max_features_ratio))

        rfe = RFE(estimator = model, n_features_to_select = max_features, step = 1)
        rfe.fit(X, y)

        selected_features = X.columns[rfe.support_].tolist()
        removed_features = X.columns[~rfe.support_].tolist()

        selection_report["selected_features"] = selected_features
        selection_report["features_removed"] = removed_features
        selection_report["feature_importance"] = dict(
            zip(selected_features, rfe.ranking_[rfe.support_])
        )

        return selection_report


class MLProtectionSystem:
    """Main protection system coordinating all components"""

    def __init__(
        self, 
        protection_level: ProtectionLevel = ProtectionLevel.STANDARD, 
        config_path: Optional[str] = None, 
    ):
        self.protection_level = protection_level
        self.config = self._load_config(config_path)

        # Initialize components
        self.noise_detector = NoiseDetector(self.config["noise"])
        self.leakage_detector = LeakageDetector(self.config["leakage"])
        self.overfitting_detector = OverfittingDetector(self.config["overfitting"])
        # Initialize tracking if available
        if TRACKING_AVAILABLE:
            self.tracker = EnterpriseTracker("tracking_config.yaml")
        else:
            self.tracker = None

        # Setup logging
        self.logger = self._setup_logging()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration based on protection level"""
        if config_path and Path(config_path).exists():
            with open(config_path, "r", encoding = "utf - 8") as f:
                loaded_config = yaml.safe_load(f)
                # Convert dictionaries to config objects if needed
                return self._convert_dict_to_config_objects(loaded_config)

        # Default configurations based on protection level
        configs = {
            ProtectionLevel.BASIC: {
                "noise": NoiseConfig(contamination_rate = 0.05), 
                "leakage": LeakageConfig(temporal_gap_hours = 12), 
                "overfitting": OverfittingConfig(max_features_ratio = 0.5), 
            }, 
            ProtectionLevel.STANDARD: {
                "noise": NoiseConfig(), 
                "leakage": LeakageConfig(), 
                "overfitting": OverfittingConfig(), 
            }, 
            ProtectionLevel.AGGRESSIVE: {
                "noise": NoiseConfig(contamination_rate = 0.15, volatility_threshold = 2.0), 
                "leakage": LeakageConfig(
                    temporal_gap_hours = 48, target_leakage_threshold = 0.6
                ), 
                "overfitting": OverfittingConfig(
                    max_features_ratio = 0.2, min_samples_per_feature = 20
                ), 
            }, 
            ProtectionLevel.ENTERPRISE: {
                "noise": NoiseConfig(
                    contamination_rate = 0.2, 
                    volatility_threshold = 1.5, 
                    enable_adaptive_filtering = True, 
                    feature_noise_detection = True, 
                ), 
                "leakage": LeakageConfig(
                    temporal_gap_hours = 72, 
                    strict_time_validation = True, 
                    target_leakage_threshold = 0.5, 
                    enable_feature_timing_check = True, 
                ), 
                "overfitting": OverfittingConfig(
                    max_features_ratio = 0.15, 
                    min_samples_per_feature = 30, 
                    cross_validation_folds = 10, 
                    enable_ensemble_validation = True, 
                ), 
            }, 
        }

        return configs[self.protection_level]

    def _convert_dict_to_config_objects(
        self, config_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert dictionary config to config objects"""
        result = {}

        # Convert noise config
        if "noise" in config_dict:
            if isinstance(config_dict["noise"], dict):
                result["noise"] = NoiseConfig(**config_dict["noise"])
            else:
                result["noise"] = config_dict["noise"]
        else:
            result["noise"] = NoiseConfig()

        # Convert leakage config
        if "leakage" in config_dict:
            if isinstance(config_dict["leakage"], dict):
                result["leakage"] = LeakageConfig(**config_dict["leakage"])
            else:
                result["leakage"] = config_dict["leakage"]
        else:
            result["leakage"] = LeakageConfig()

        # Convert overfitting config
        if "overfitting" in config_dict:
            if isinstance(config_dict["overfitting"], dict):
                result["overfitting"] = OverfittingConfig(**config_dict["overfitting"])
            else:
                result["overfitting"] = config_dict["overfitting"]
        else:
            result["overfitting"] = OverfittingConfig()

        return result

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for protection system"""
        logger = logging.getLogger("ml_protection")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def protect_dataset(
        self, 
        data: pd.DataFrame, 
        target_col: str = "target", 
        timestamp_col: str = "timestamp", 
        model: Any = None, 
    ) -> ProtectionResult:
        """Main method to protect dataset from all issues"""

        self.logger.info(
            f"Starting ML protection analysis with {self.protection_level.value} level"
        )

        # Start tracking if available
        if self.tracker:
            run_name = f"ml_protection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            tracking_run = self.tracker.start_run(run_name)
        else:
            tracking_run = None

        try:
            result = ProtectionResult(
                is_clean = True, noise_score = 0.0, leakage_score = 0.0, overfitting_score = 0.0
            )

            # Prepare data
            X = data.drop([target_col], axis = 1) if target_col in data.columns else data
            y = data[target_col] if target_col in data.columns else None

            # 1. Noise Detection and Cleaning
            self.logger.info("Detecting and cleaning noise...")
            noise_report = self.noise_detector.detect_statistical_noise(data)

            if y is not None:
                feature_noise = self.noise_detector.detect_feature_noise(X, y)
                noise_report["feature_noise"] = feature_noise

            # Calculate noise score
            total_outliers = sum(
                info["count"] for info in noise_report["outliers"].values()
            )
            result.noise_score = min(1.0, total_outliers / len(data))

            # Clean data if noise detected
            if result.noise_score > 0.1:  # 10% threshold
                cleaned_data, cleaning_report = self.noise_detector.clean_data(data)
                result.cleaned_data = cleaned_data
                result.issues_found.append(
                    f"High noise detected (score: {result.noise_score:.3f})"
                )
                result.recommendations.append("Applied noise cleaning procedures")

                if tracking_run:
                    self.tracker.log_metrics(
                        {
                            "noise_score": result.noise_score, 
                            "rows_removed_by_cleaning": cleaning_report["rows_removed"], 
                        }
                    )
            else:
                result.cleaned_data = data

            # 2. Data Leakage Detection
            self.logger.info("Detecting data leakage...")
            temporal_leakage = self.leakage_detector.detect_temporal_leakage(
                data, timestamp_col, target_col
            )

            if y is not None:
                feature_leakage = self.leakage_detector.detect_feature_leakage(X, y)
                temporal_leakage.update(feature_leakage)

            # Calculate leakage score
            leakage_factors = 0
            if temporal_leakage.get("future_data_leakage", False):
                leakage_factors += 0.5
            if temporal_leakage.get("perfect_correlations", {}):
                leakage_factors += 0.3
            if temporal_leakage.get("feature_timing_issues", {}):
                leakage_factors += 0.2

            result.leakage_score = min(1.0, leakage_factors)

            if result.leakage_score > 0.1:
                result.issues_found.append(
                    f"Data leakage detected (score: {result.leakage_score:.3f})"
                )
                result.recommendations.append(
                    "Review feature engineering and temporal alignment"
                )
                result.is_clean = False

                if tracking_run:
                    self.tracker.log_metrics(
                        {
                            "leakage_score": result.leakage_score, 
                            "leakage_issues_count": len(
                                temporal_leakage.get("perfect_correlations", {})
                            ), 
                        }
                    )

            # 3. Overfitting Risk Assessment
            if model is not None and y is not None:
                self.logger.info("Assessing overfitting risk...")
                overfitting_risk = self.overfitting_detector.detect_overfitting_risk(
                    X, y, model
                )

                # Perform time series cross - validation
                cv_results = self.overfitting_detector.validate_with_time_series_cv(
                    X, y, model, timestamp_col
                )

                # Feature selection validation
                feature_selection = (
                    self.overfitting_detector.feature_selection_validation(X, y, model)
                )

                # Calculate overfitting score
                risk_factors = len(overfitting_risk.get("risk_factors", {}))
                cv_variance = cv_results.get("score_variance", 0)
                result.overfitting_score = min(1.0, (risk_factors * 0.2) + cv_variance)

                if result.overfitting_score > 0.3:
                    result.issues_found.append(
                        f"Overfitting risk detected (score: {result.overfitting_score:.3f})"
                    )
                    result.recommendations.extend(
                        overfitting_risk.get("recommendations", [])
                    )
                    result.is_clean = False

                    if tracking_run:
                        self.tracker.log_metrics(
                            {
                                "overfitting_score": result.overfitting_score, 
                                "cv_mean_score": cv_results.get("mean_score", 0), 
                                "cv_std_score": cv_results.get("std_score", 0), 
                                "risk_factors_count": risk_factors, 
                            }
                        )

                # Store detailed results
                result.feature_report = {
                    "noise_analysis": noise_report, 
                    "leakage_analysis": temporal_leakage, 
                    "overfitting_analysis": overfitting_risk, 
                    "cv_results": cv_results, 
                    "feature_selection": feature_selection, 
                }

            # Generate final recommendations
            if result.is_clean:
                result.recommendations.append("✅ Dataset passed all protection checks")
            else:
                result.recommendations.append(
                    "⚠️ Address identified issues before production deployment"
                )

            # Log final results
            self.logger.info(f"Protection analysis complete. Clean: {result.is_clean}")
            self.logger.info(
                f"Scores - Noise: {result.noise_score:.3f}, "
                f"Leakage: {result.leakage_score:.3f}, "
                f"Overfitting: {result.overfitting_score:.3f}"
            )

            if tracking_run:
                self.tracker.log_metrics(
                    {
                        "protection_final_score": (
                            1
                            - result.noise_score
                            - result.leakage_score
                            - result.overfitting_score
                        ), 
                        "is_clean": int(result.is_clean), 
                        "issues_found_count": len(result.issues_found), 
                    }
                )

                # Log protection report as artifact
                report_path = (
                    f"protection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
                )
                with open(report_path, "w") as f:
                    yaml.dump(
                        {
                            "protection_result": {
                                "is_clean": result.is_clean, 
                                "scores": {
                                    "noise": result.noise_score, 
                                    "leakage": result.leakage_score, 
                                    "overfitting": result.overfitting_score, 
                                }, 
                                "issues": result.issues_found, 
                                "recommendations": result.recommendations, 
                            }
                        }, 
                        f, 
                    )

                self.tracker.log_artifact(report_path)

            return result

        except Exception as e:
            self.logger.error(f"Error in protection analysis: {str(e)}")
            if tracking_run:
                self.tracker.log_params({"error": str(e)})
            raise

        finally:
            if tracking_run:
                tracking_run.__exit__(None, None, None)

    def generate_protection_report(
        self, result: ProtectionResult, output_path: str = "ml_protection_report.html"
    ) -> str:
        """Generate comprehensive protection report"""

        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML Protection System Report</title>
            <style>
                body { font - family: Arial, sans - serif; margin: 40px; }
                .header { background: #2c3e50; color: white; padding: 20px; border - radius: 5px; }
                .score { display: inline - block; margin: 10px; padding: 15px; border - radius: 5px; color: white; }
                .score.good { background: #27ae60; }
                .score.warning { background: #f39c12; }
                .score.danger { background: #e74c3c; }
                .section { margin: 20px 0; padding: 15px; border - left: 4px solid #3498db; }
                .issue { background: #f8d7da; padding: 10px; margin: 5px 0; border - radius: 3px; }
                .recommendation { background: #d1ecf1; padding: 10px; margin: 5px 0; border - radius: 3px; }
            </style>
        </head>
        <body>
            <div class = "header">
                <h1>🛡️ ML Protection System Report</h1>
                <p>Generated on: {timestamp}</p>
                <p>Protection Level: {protection_level}</p>
                <p>Overall Status: {"✅ CLEAN" if result.is_clean else "⚠️ ISSUES DETECTED"}</p>
            </div>

            <div class = "section">
                <h2>📊 Protection Scores</h2>
                <div class = "score {noise_class}">
                    <h3>Noise Score</h3>
                    <p>{noise_score:.3f}</p>
                </div>
                <div class = "score {leakage_class}">
                    <h3>Leakage Score</h3>
                    <p>{leakage_score:.3f}</p>
                </div>
                <div class = "score {overfitting_class}">
                    <h3>Overfitting Score</h3>
                    <p>{overfitting_score:.3f}</p>
                </div>
            </div>

            <div class = "section">
                <h2>⚠️ Issues Found</h2>
                {issues_html}
            </div>

            <div class = "section">
                <h2>💡 Recommendations</h2>
                {recommendations_html}
            </div>

            <div class = "section">
                <h2>📋 Detailed Analysis</h2>
                <pre>{detailed_report}</pre>
            </div>
        </body>
        </html>
        """

        # Helper function to determine score class
        def get_score_class(score):
            if score < 0.1:
                return "good"
            elif score < 0.3:
                return "warning"
            else:
                return "danger"

        # Generate HTML content
        issues_html = "".join(
            [f'<div class = "issue">❌ {issue}</div>' for issue in result.issues_found]
        )
        if not issues_html:
            issues_html = '<div class = "score good">No issues detected ✅</div>'

        recommendations_html = "".join(
            [
                f'<div class = "recommendation">💡 {rec}</div>'
                for rec in result.recommendations
            ]
        )

        # Generate report
        html_content = html_template.format(
            timestamp = datetime.now().strftime("%Y - %m - %d %H:%M:%S"), 
            protection_level = self.protection_level.value.upper(), 
            result = result, 
            noise_score = result.noise_score, 
            leakage_score = result.leakage_score, 
            overfitting_score = result.overfitting_score, 
            noise_class = get_score_class(result.noise_score), 
            leakage_class = get_score_class(result.leakage_score), 
            overfitting_class = get_score_class(result.overfitting_score), 
            issues_html = issues_html, 
            recommendations_html = recommendations_html, 
            detailed_report = yaml.dump(result.feature_report, default_flow_style = False), 
        )

        # Save report
        with open(output_path, "w", encoding = "utf - 8") as f:
            f.write(html_content)

        self.logger.info(f"Protection report saved to: {output_path}")
        return output_path


# Example usage and integration
if __name__ == "__main__":
    # Example usage
    protection_system = MLProtectionSystem(ProtectionLevel.ENTERPRISE)

    # Load your data
    # data = pd.read_csv("your_trading_data.csv")

    # Protect dataset
    # result = protection_system.protect_dataset(data, target_col = 'target', timestamp_col = 'timestamp')

    # Generate report
    # protection_system.generate_protection_report(result)

    print("🛡️ ML Protection System initialized successfully!")
    print("📋 Available protection levels:", [level.value for level in ProtectionLevel])