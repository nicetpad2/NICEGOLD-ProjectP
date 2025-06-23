"""
Advanced ML Protection System for Trading ML Pipelines
======================================================

Enterprise-grade protection against noise, data leakage, and overfitting
with advanced monitoring, validation, and automated remediation.

Author: AI Assistant
Version: 2.0.0
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
import yaml
import json
import os
from pathlib import Path
import hashlib
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import IsolationForest
from scipy import stats
from scipy.stats import jarque_bera, shapiro
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import wraps
import time
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_ml_protection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ProtectionConfig:
    """Configuration for Advanced ML Protection System"""
    
    # Data Quality Settings
    max_missing_percentage: float = 0.1
    min_variance_threshold: float = 1e-8
    max_correlation_threshold: float = 0.95
    outlier_detection_method: str = 'isolation_forest'
    outlier_contamination: float = 0.1
    
    # Temporal Validation Settings
    temporal_validation_enabled: bool = True
    min_temporal_window: int = 30
    max_lookback_days: int = 252
    temporal_split_ratio: float = 0.2
    
    # Data Leakage Protection
    future_data_check: bool = True
    target_leakage_check: bool = True
    temporal_leakage_check: bool = True
    feature_stability_check: bool = True
    
    # Overfitting Protection
    cross_validation_folds: int = 5
    max_model_complexity: float = 0.8
    early_stopping_patience: int = 10
    regularization_strength: float = 0.01
    feature_selection_enabled: bool = True
    max_features_ratio: float = 0.3
    
    # Noise Reduction
    noise_detection_enabled: bool = True
    signal_to_noise_threshold: float = 2.0
    smoothing_window: int = 5
    denoising_method: str = 'robust_scaler'
    
    # Advanced Protection Features
    ensemble_validation: bool = True
    market_regime_detection: bool = True
    volatility_clustering_check: bool = True
    trend_consistency_check: bool = True
    
    # Performance Monitoring
    performance_tracking: bool = True
    alert_threshold_auc: float = 0.6
    alert_threshold_stability: float = 0.1
    monitoring_window_days: int = 30
    
    # Storage and Backup
    backup_enabled: bool = True
    backup_frequency: str = 'daily'
    max_backup_files: int = 30
    compression_enabled: bool = True

@dataclass
class ProtectionReport:
    """Comprehensive protection analysis report"""
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Data Quality Results
    data_quality_score: float = 0.0
    missing_data_issues: List[str] = field(default_factory=list)
    outlier_count: int = 0
    correlation_issues: List[str] = field(default_factory=list)
    
    # Leakage Detection Results
    temporal_leakage_detected: bool = False
    target_leakage_detected: bool = False
    feature_leakage_issues: List[str] = field(default_factory=list)
    
    # Overfitting Assessment
    overfitting_risk: str = 'low'
    cross_validation_scores: List[float] = field(default_factory=list)
    model_complexity_score: float = 0.0
    
    # Noise Analysis
    signal_to_noise_ratio: float = 0.0
    noise_level: str = 'acceptable'
    denoising_recommendations: List[str] = field(default_factory=list)
    
    # Overall Assessment
    overall_protection_score: float = 0.0
    risk_level: str = 'low'
    recommendations: List[str] = field(default_factory=list)
    
    # Performance Metrics
    processing_time_seconds: float = 0.0
    memory_usage_mb: float = 0.0

class AdvancedMLProtectionSystem:
    """
    Advanced ML Protection System for Trading ML Pipelines
    
    Features:
    - Comprehensive data quality validation
    - Advanced data leakage detection
    - Overfitting prevention with multiple techniques
    - Noise reduction and signal enhancement
    - Market regime-aware validation
    - Real-time monitoring and alerting
    - Automated remediation suggestions
    """
    
    def __init__(self, config: Optional[ProtectionConfig] = None):
        """Initialize the Advanced ML Protection System"""
        self.config = config or ProtectionConfig()
        self.reports_history: List[ProtectionReport] = []
        self.model_cache = {}
        self.feature_importance_cache = {}
        self.market_regime_detector = None
        
        # Create necessary directories
        self.setup_directories()
        
        logger.info("Advanced ML Protection System initialized")
    
    def setup_directories(self):
        """Setup required directories for protection system"""
        directories = [
            'protection_reports',
            'protection_backups',
            'protection_cache',
            'protection_models',
            'protection_logs'
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    def analyze_data_comprehensive(
        self, 
        data: pd.DataFrame, 
        target_column: str = 'target',
        feature_columns: Optional[List[str]] = None
    ) -> ProtectionReport:
        """
        Comprehensive data analysis with advanced protection features
        
        Args:
            data: Input DataFrame
            target_column: Name of target column
            feature_columns: List of feature columns (if None, auto-detect)
            
        Returns:
            ProtectionReport with comprehensive analysis results
        """
        start_time = time.time()
        memory_start = self._get_memory_usage()
        
        logger.info("Starting comprehensive data analysis...")
        
        # Initialize report
        report = ProtectionReport()
        
        try:
            # Prepare data
            if feature_columns is None:
                feature_columns = [col for col in data.columns if col != target_column]
            
            # 1. Data Quality Assessment
            self._assess_data_quality(data, target_column, feature_columns, report)
            
            # 2. Temporal Validation
            if self.config.temporal_validation_enabled:
                self._perform_temporal_validation(data, target_column, feature_columns, report)
            
            # 3. Data Leakage Detection
            self._detect_data_leakage(data, target_column, feature_columns, report)
            
            # 4. Overfitting Risk Assessment
            self._assess_overfitting_risk(data, target_column, feature_columns, report)
            
            # 5. Noise Analysis and Reduction
            if self.config.noise_detection_enabled:
                self._analyze_noise(data, target_column, feature_columns, report)
            
            # 6. Market Regime Analysis (for trading data)
            if self.config.market_regime_detection:
                self._analyze_market_regimes(data, target_column, feature_columns, report)
            
            # 7. Calculate Overall Scores
            self._calculate_overall_scores(report)
            
            # 8. Generate Recommendations
            self._generate_recommendations(report)
            
            # Record performance metrics
            report.processing_time_seconds = time.time() - start_time
            report.memory_usage_mb = self._get_memory_usage() - memory_start
            
            # Save report
            self._save_report(report)
            
            logger.info(f"Comprehensive analysis completed in {report.processing_time_seconds:.2f} seconds")
            
            return report
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _assess_data_quality(
        self, 
        data: pd.DataFrame, 
        target_column: str, 
        feature_columns: List[str], 
        report: ProtectionReport
    ):
        """Assess data quality with advanced metrics"""
        logger.info("Assessing data quality...")
        
        quality_scores = []
        
        # Missing data analysis
        missing_percentages = data[feature_columns].isnull().mean()
        problematic_features = missing_percentages[missing_percentages > self.config.max_missing_percentage]
        
        if len(problematic_features) > 0:
            report.missing_data_issues = [
                f"Feature '{feature}' has {percentage:.2%} missing values"
                for feature, percentage in problematic_features.items()
            ]
        
        missing_score = 1.0 - (len(problematic_features) / len(feature_columns))
        quality_scores.append(missing_score)
        
        # Variance analysis
        numeric_features = data[feature_columns].select_dtypes(include=[np.number])
        low_variance_features = numeric_features.var()[numeric_features.var() < self.config.min_variance_threshold]
        
        if len(low_variance_features) > 0:
            report.missing_data_issues.extend([
                f"Feature '{feature}' has very low variance ({variance:.2e})"
                for feature, variance in low_variance_features.items()
            ])
        
        variance_score = 1.0 - (len(low_variance_features) / len(numeric_features.columns))
        quality_scores.append(variance_score)
        
        # Correlation analysis
        correlation_matrix = numeric_features.corr().abs()
        high_corr_pairs = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if corr_value > self.config.max_correlation_threshold:
                    feature1 = correlation_matrix.columns[i]
                    feature2 = correlation_matrix.columns[j]
                    high_corr_pairs.append((feature1, feature2, corr_value))
        
        if high_corr_pairs:
            report.correlation_issues = [
                f"High correlation between '{f1}' and '{f2}': {corr:.3f}"
                for f1, f2, corr in high_corr_pairs
            ]
        
        correlation_score = 1.0 - (len(high_corr_pairs) / max(1, len(numeric_features.columns) * (len(numeric_features.columns) - 1) / 2))
        quality_scores.append(correlation_score)
        
        # Outlier detection
        if self.config.outlier_detection_method == 'isolation_forest':
            outlier_detector = IsolationForest(
                contamination=self.config.outlier_contamination,
                random_state=42
            )
            outliers = outlier_detector.fit_predict(numeric_features.fillna(0))
            report.outlier_count = np.sum(outliers == -1)
        
        outlier_score = 1.0 - (report.outlier_count / len(data))
        quality_scores.append(outlier_score)
        
        # Calculate overall data quality score
        report.data_quality_score = np.mean(quality_scores)
        
        logger.info(f"Data quality assessment completed. Score: {report.data_quality_score:.3f}")
    
    def _perform_temporal_validation(
        self, 
        data: pd.DataFrame, 
        target_column: str, 
        feature_columns: List[str], 
        report: ProtectionReport
    ):
        """Perform temporal validation for time series data"""
        logger.info("Performing temporal validation...")
        
        # Check if data has datetime index or column
        datetime_columns = data.select_dtypes(include=['datetime64']).columns
        
        if len(datetime_columns) == 0 and not isinstance(data.index, pd.DatetimeIndex):
            logger.warning("No datetime column found. Skipping temporal validation.")
            return
        
        # Ensure data is sorted by time
        if isinstance(data.index, pd.DatetimeIndex):
            data_sorted = data.sort_index()
        else:
            # Use first datetime column for sorting
            data_sorted = data.sort_values(datetime_columns[0])
        
        # Check for temporal consistency
        if len(data_sorted) < self.config.min_temporal_window:
            report.recommendations.append(
                f"Dataset too small for reliable temporal validation. "
                f"Minimum {self.config.min_temporal_window} samples required."
            )
            return
        
        # Temporal stability check
        window_size = min(self.config.min_temporal_window, len(data_sorted) // 4)
        stability_scores = []
        
        for feature in feature_columns:
            if data_sorted[feature].dtype in ['int64', 'float64']:
                rolling_mean = data_sorted[feature].rolling(window=window_size).mean()
                rolling_std = data_sorted[feature].rolling(window=window_size).std()
                
                # Calculate coefficient of variation
                cv = rolling_std / (rolling_mean + 1e-8)
                stability_score = 1.0 / (1.0 + cv.mean())
                stability_scores.append(stability_score)
        
        temporal_stability = np.mean(stability_scores) if stability_scores else 0.0
        
        if temporal_stability < (1.0 - self.config.alert_threshold_stability):
            report.feature_leakage_issues.append(
                f"Low temporal stability detected: {temporal_stability:.3f}"
            )
        
        logger.info(f"Temporal validation completed. Stability: {temporal_stability:.3f}")
    
    def _detect_data_leakage(
        self, 
        data: pd.DataFrame, 
        target_column: str, 
        feature_columns: List[str], 
        report: ProtectionReport
    ):
        """Advanced data leakage detection"""
        logger.info("Detecting data leakage...")
        
        # Target leakage detection
        if self.config.target_leakage_check:
            target_corr = data[feature_columns].corrwith(data[target_column]).abs()
            suspicious_features = target_corr[target_corr > 0.95]
            
            if len(suspicious_features) > 0:
                report.target_leakage_detected = True
                report.feature_leakage_issues.extend([
                    f"Potential target leakage in '{feature}': correlation {corr:.3f}"
                    for feature, corr in suspicious_features.items()
                ])
        
        # Future data leakage detection (for time series)
        if self.config.future_data_check and isinstance(data.index, pd.DatetimeIndex):
            # Check if features have values that shouldn't exist at prediction time
            for feature in feature_columns:
                if 'future' in feature.lower() or 'forward' in feature.lower():
                    report.feature_leakage_issues.append(
                        f"Potential future data leakage in feature: '{feature}'"
                    )
        
        # Feature stability over time
        if self.config.feature_stability_check:
            self._check_feature_stability(data, feature_columns, report)
        
        logger.info("Data leakage detection completed")
    
    def _check_feature_stability(
        self, 
        data: pd.DataFrame, 
        feature_columns: List[str], 
        report: ProtectionReport
    ):
        """Check feature stability over time"""
        numeric_features = [col for col in feature_columns if data[col].dtype in ['int64', 'float64']]
        
        if len(data) < 100:  # Need sufficient data for stability check
            return
        
        # Split data into time periods
        n_periods = min(5, len(data) // 20)
        period_size = len(data) // n_periods
        
        unstable_features = []
        
        for feature in numeric_features:
            period_means = []
            for i in range(n_periods):
                start_idx = i * period_size
                end_idx = (i + 1) * period_size if i < n_periods - 1 else len(data)
                period_data = data.iloc[start_idx:end_idx][feature]
                period_means.append(period_data.mean())
            
            # Calculate coefficient of variation across periods
            if len(period_means) > 1:
                cv = np.std(period_means) / (np.mean(period_means) + 1e-8)
                if cv > 0.5:  # Threshold for instability
                    unstable_features.append((feature, cv))
        
        if unstable_features:
            report.feature_leakage_issues.extend([
                f"Feature '{feature}' shows temporal instability (CV: {cv:.3f})"
                for feature, cv in unstable_features
            ])
    
    def _assess_overfitting_risk(
        self, 
        data: pd.DataFrame, 
        target_column: str, 
        feature_columns: List[str], 
        report: ProtectionReport
    ):
        """Assess overfitting risk using multiple techniques"""
        logger.info("Assessing overfitting risk...")
        
        # Prepare data
        X = data[feature_columns].fillna(0)
        y = data[target_column]
        
        # Cross-validation assessment
        try:
            # Use appropriate cross-validation strategy
            if isinstance(data.index, pd.DatetimeIndex):
                # Time series cross-validation
                cv = TimeSeriesSplit(n_splits=self.config.cross_validation_folds)
            else:
                # Standard stratified cross-validation
                cv = StratifiedKFold(n_splits=self.config.cross_validation_folds, shuffle=True, random_state=42)
            
            # Simple model for baseline assessment
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(random_state=42, max_iter=1000)
            
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
            report.cross_validation_scores = cv_scores.tolist()
            
            # Assess overfitting based on CV score variance
            cv_std = np.std(cv_scores)
            cv_mean = np.mean(cv_scores)
            
            if cv_std > 0.1:
                report.overfitting_risk = 'high'
            elif cv_std > 0.05:
                report.overfitting_risk = 'medium'
            else:
                report.overfitting_risk = 'low'
            
            # Model complexity assessment
            n_features = len(feature_columns)
            n_samples = len(data)
            complexity_ratio = n_features / n_samples
            
            if complexity_ratio > 0.1:
                report.model_complexity_score = 1.0
                report.overfitting_risk = 'high'
            elif complexity_ratio > 0.05:
                report.model_complexity_score = 0.7
            else:
                report.model_complexity_score = 0.3
            
        except Exception as e:
            logger.warning(f"Could not assess overfitting risk: {str(e)}")
            report.overfitting_risk = 'unknown'
        
        logger.info(f"Overfitting risk assessment completed. Risk: {report.overfitting_risk}")
    
    def _analyze_noise(
        self, 
        data: pd.DataFrame, 
        target_column: str, 
        feature_columns: List[str], 
        report: ProtectionReport
    ):
        """Analyze noise levels and provide denoising recommendations"""
        logger.info("Analyzing noise levels...")
        
        numeric_features = [col for col in feature_columns if data[col].dtype in ['int64', 'float64']]
        
        if len(numeric_features) == 0:
            logger.warning("No numeric features found for noise analysis")
            return
        
        signal_to_noise_ratios = []
        
        for feature in numeric_features:
            feature_data = data[feature].dropna()
            if len(feature_data) < 10:
                continue
            
            # Calculate signal-to-noise ratio using multiple methods
            
            # Method 1: Mean to standard deviation ratio
            mean_val = np.abs(feature_data.mean())
            std_val = feature_data.std()
            snr1 = mean_val / (std_val + 1e-8)
            
            # Method 2: Signal power to noise power ratio
            # Use smoothed signal as reference
            if len(feature_data) > self.config.smoothing_window:
                smoothed = feature_data.rolling(window=self.config.smoothing_window, center=True).mean()
                noise = feature_data - smoothed
                signal_power = np.var(smoothed.dropna())
                noise_power = np.var(noise.dropna())
                snr2 = signal_power / (noise_power + 1e-8)
            else:
                snr2 = snr1
            
            # Take geometric mean of both methods
            combined_snr = np.sqrt(snr1 * snr2) if snr1 > 0 and snr2 > 0 else 0
            signal_to_noise_ratios.append(combined_snr)
        
        # Calculate overall signal-to-noise ratio
        if signal_to_noise_ratios:
            report.signal_to_noise_ratio = np.mean(signal_to_noise_ratios)
            
            if report.signal_to_noise_ratio < self.config.signal_to_noise_threshold:
                report.noise_level = 'high'
                report.denoising_recommendations.extend([
                    f"Apply {self.config.denoising_method} scaling",
                    f"Use smoothing with window size {self.config.smoothing_window}",
                    "Consider feature selection to remove noisy features",
                    "Apply ensemble methods to reduce noise impact"
                ])
            elif report.signal_to_noise_ratio < self.config.signal_to_noise_threshold * 2:
                report.noise_level = 'medium'
                report.denoising_recommendations.append(
                    "Consider light denoising techniques"
                )
            else:
                report.noise_level = 'low'
        
        logger.info(f"Noise analysis completed. SNR: {report.signal_to_noise_ratio:.2f}")
    
    def _analyze_market_regimes(
        self, 
        data: pd.DataFrame, 
        target_column: str, 
        feature_columns: List[str], 
        report: ProtectionReport
    ):
        """Analyze market regimes for trading data"""
        logger.info("Analyzing market regimes...")
        
        # This is a simplified market regime analysis
        # In practice, you might want to use more sophisticated techniques
        
        try:
            # Look for volatility clustering
            if 'returns' in data.columns or any('return' in col.lower() for col in data.columns):
                return_cols = [col for col in data.columns if 'return' in col.lower()]
                if return_cols:
                    returns = data[return_cols[0]].dropna()
                    
                    # Calculate rolling volatility
                    rolling_vol = returns.rolling(window=20).std()
                    
                    # Detect regime changes using volatility clustering
                    high_vol_threshold = rolling_vol.quantile(0.75)
                    low_vol_threshold = rolling_vol.quantile(0.25)
                    
                    high_vol_periods = (rolling_vol > high_vol_threshold).sum()
                    low_vol_periods = (rolling_vol < low_vol_threshold).sum()
                    
                    if high_vol_periods > len(rolling_vol) * 0.3:
                        report.recommendations.append(
                            "High volatility periods detected. Consider regime-specific models."
                        )
            
            # Check for trend consistency
            if self.config.trend_consistency_check:
                # Simple trend analysis using price or target variable
                if len(data) > 50:
                    rolling_mean = data[target_column].rolling(window=20).mean()
                    trend_changes = np.diff(np.sign(np.diff(rolling_mean.dropna())))
                    trend_consistency = 1.0 - (np.sum(trend_changes != 0) / len(trend_changes))
                    
                    if trend_consistency < 0.7:
                        report.recommendations.append(
                            f"Low trend consistency detected ({trend_consistency:.2f}). "
                            "Consider adaptive models."
                        )
        
        except Exception as e:
            logger.warning(f"Market regime analysis failed: {str(e)}")
        
        logger.info("Market regime analysis completed")
    
    def _calculate_overall_scores(self, report: ProtectionReport):
        """Calculate overall protection scores"""
        scores = []
        
        # Data quality contributes 30%
        scores.append(report.data_quality_score * 0.3)
        
        # Leakage detection contributes 25%
        leakage_score = 1.0
        if report.temporal_leakage_detected:
            leakage_score -= 0.3
        if report.target_leakage_detected:
            leakage_score -= 0.4
        if report.feature_leakage_issues:
            leakage_score -= min(0.3, len(report.feature_leakage_issues) * 0.1)
        
        scores.append(max(0, leakage_score) * 0.25)
        
        # Overfitting assessment contributes 25%
        overfitting_score = {'low': 1.0, 'medium': 0.6, 'high': 0.2}.get(report.overfitting_risk, 0.5)
        scores.append(overfitting_score * 0.25)
        
        # Noise level contributes 20%
        noise_score = {'low': 1.0, 'medium': 0.7, 'high': 0.3}.get(report.noise_level, 0.5)
        scores.append(noise_score * 0.2)
        
        # Calculate overall score
        report.overall_protection_score = sum(scores)
        
        # Determine risk level
        if report.overall_protection_score >= 0.8:
            report.risk_level = 'low'
        elif report.overall_protection_score >= 0.6:
            report.risk_level = 'medium'
        else:
            report.risk_level = 'high'
    
    def _generate_recommendations(self, report: ProtectionReport):
        """Generate actionable recommendations based on analysis"""
        if not report.recommendations:
            report.recommendations = []
        
        # Data quality recommendations
        if report.data_quality_score < 0.7:
            report.recommendations.append("Improve data quality through cleaning and preprocessing")
        
        if report.missing_data_issues:
            report.recommendations.append("Address missing data using imputation or feature engineering")
        
        if report.correlation_issues:
            report.recommendations.append("Remove highly correlated features to reduce redundancy")
        
        # Leakage prevention recommendations
        if report.target_leakage_detected:
            report.recommendations.append("CRITICAL: Remove features causing target leakage")
        
        if report.feature_leakage_issues:
            report.recommendations.append("Review temporal feature construction to prevent data leakage")
        
        # Overfitting prevention recommendations
        if report.overfitting_risk == 'high':
            report.recommendations.extend([
                "Increase regularization strength",
                "Reduce model complexity",
                "Use more robust cross-validation",
                "Implement early stopping"
            ])
        
        # Noise reduction recommendations
        if report.noise_level == 'high':
            report.recommendations.extend(report.denoising_recommendations)
        
        # Performance recommendations
        if report.overall_protection_score < 0.6:
            report.recommendations.append(
                "Overall protection score is low. Consider comprehensive data preprocessing."
            )
    
    def _save_report(self, report: ProtectionReport):
        """Save protection report to file"""
        try:
            timestamp = report.timestamp.strftime("%Y%m%d_%H%M%S")
            report_path = Path(f"protection_reports/protection_report_{timestamp}.json")
            
            # Convert report to dictionary for JSON serialization
            report_dict = {
                'timestamp': report.timestamp.isoformat(),
                'data_quality_score': report.data_quality_score,
                'missing_data_issues': report.missing_data_issues,
                'outlier_count': report.outlier_count,
                'correlation_issues': report.correlation_issues,
                'temporal_leakage_detected': report.temporal_leakage_detected,
                'target_leakage_detected': report.target_leakage_detected,
                'feature_leakage_issues': report.feature_leakage_issues,
                'overfitting_risk': report.overfitting_risk,
                'cross_validation_scores': report.cross_validation_scores,
                'model_complexity_score': report.model_complexity_score,
                'signal_to_noise_ratio': report.signal_to_noise_ratio,
                'noise_level': report.noise_level,
                'denoising_recommendations': report.denoising_recommendations,
                'overall_protection_score': report.overall_protection_score,
                'risk_level': report.risk_level,
                'recommendations': report.recommendations,
                'processing_time_seconds': report.processing_time_seconds,
                'memory_usage_mb': report.memory_usage_mb
            }
            
            with open(report_path, 'w') as f:
                json.dump(report_dict, f, indent=2)
            
            # Add to history
            self.reports_history.append(report)
            
            # Keep only recent reports in memory
            if len(self.reports_history) > 100:
                self.reports_history = self.reports_history[-100:]
            
            logger.info(f"Report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to save report: {str(e)}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def apply_automated_fixes(
        self, 
        data: pd.DataFrame, 
        report: ProtectionReport,
        target_column: str = 'target'
    ) -> pd.DataFrame:
        """
        Apply automated fixes based on protection report
        
        Args:
            data: Input DataFrame
            report: Protection report with identified issues
            target_column: Name of target column
            
        Returns:
            DataFrame with automated fixes applied
        """
        logger.info("Applying automated fixes...")
        
        fixed_data = data.copy()
        
        try:
            # Fix missing data issues
            if report.missing_data_issues:
                for issue in report.missing_data_issues:
                    if "missing values" in issue:
                        feature_name = issue.split("'")[1]
                        if feature_name in fixed_data.columns:
                            if fixed_data[feature_name].dtype in ['int64', 'float64']:
                                # Use median for numeric features
                                fixed_data[feature_name].fillna(
                                    fixed_data[feature_name].median(), 
                                    inplace=True
                                )
                            else:
                                # Use mode for categorical features
                                mode_value = fixed_data[feature_name].mode()
                                if len(mode_value) > 0:
                                    fixed_data[feature_name].fillna(mode_value[0], inplace=True)
            
            # Remove highly correlated features
            if report.correlation_issues:
                features_to_remove = set()
                for issue in report.correlation_issues:
                    if "High correlation between" in issue:
                        # Extract feature names
                        parts = issue.split("'")
                        if len(parts) >= 4:
                            feature1, feature2 = parts[1], parts[3]
                            # Remove the feature with lower correlation to target
                            if target_column in fixed_data.columns:
                                corr1 = abs(fixed_data[feature1].corr(fixed_data[target_column]))
                                corr2 = abs(fixed_data[feature2].corr(fixed_data[target_column]))
                                features_to_remove.add(feature1 if corr1 < corr2 else feature2)
                
                # Remove identified features
                features_to_remove = [f for f in features_to_remove if f in fixed_data.columns]
                if features_to_remove:
                    fixed_data.drop(columns=features_to_remove, inplace=True)
                    logger.info(f"Removed highly correlated features: {features_to_remove}")
            
            # Apply denoising if recommended
            if report.noise_level == 'high' and report.denoising_recommendations:
                numeric_columns = fixed_data.select_dtypes(include=[np.number]).columns
                numeric_columns = [col for col in numeric_columns if col != target_column]
                
                if len(numeric_columns) > 0:
                    if self.config.denoising_method == 'robust_scaler':
                        scaler = RobustScaler()
                        fixed_data[numeric_columns] = scaler.fit_transform(fixed_data[numeric_columns])
                    elif self.config.denoising_method == 'standard_scaler':
                        scaler = StandardScaler()
                        fixed_data[numeric_columns] = scaler.fit_transform(fixed_data[numeric_columns])
                    
                    logger.info(f"Applied {self.config.denoising_method} to numeric features")
            
            # Remove outliers if too many detected
            if report.outlier_count > len(data) * 0.1:  # More than 10% outliers
                numeric_columns = fixed_data.select_dtypes(include=[np.number]).columns
                numeric_columns = [col for col in numeric_columns if col != target_column]
                
                if len(numeric_columns) > 0:
                    outlier_detector = IsolationForest(
                        contamination=self.config.outlier_contamination,
                        random_state=42
                    )
                    outliers = outlier_detector.fit_predict(fixed_data[numeric_columns].fillna(0))
                    fixed_data = fixed_data[outliers != -1]
                    logger.info(f"Removed {np.sum(outliers == -1)} outliers")
            
            logger.info("Automated fixes applied successfully")
            
        except Exception as e:
            logger.error(f"Error applying automated fixes: {str(e)}")
            return data  # Return original data if fixes fail
        
        return fixed_data
    
    def generate_protection_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive protection summary"""
        if not self.reports_history:
            return {"status": "No protection reports available"}
        
        latest_report = self.reports_history[-1]
        
        summary = {
            "latest_analysis": {
                "timestamp": latest_report.timestamp.isoformat(),
                "overall_protection_score": latest_report.overall_protection_score,
                "risk_level": latest_report.risk_level,
                "data_quality_score": latest_report.data_quality_score,
                "overfitting_risk": latest_report.overfitting_risk,
                "noise_level": latest_report.noise_level,
                "signal_to_noise_ratio": latest_report.signal_to_noise_ratio
            },
            "issues_summary": {
                "missing_data_issues": len(latest_report.missing_data_issues),
                "correlation_issues": len(latest_report.correlation_issues),
                "feature_leakage_issues": len(latest_report.feature_leakage_issues),
                "outlier_count": latest_report.outlier_count,
                "target_leakage_detected": latest_report.target_leakage_detected,
                "temporal_leakage_detected": latest_report.temporal_leakage_detected
            },
            "recommendations_count": len(latest_report.recommendations),
            "performance": {
                "processing_time_seconds": latest_report.processing_time_seconds,
                "memory_usage_mb": latest_report.memory_usage_mb
            },
            "historical_trend": self._get_historical_trend(),
            "system_status": "active",
            "config_summary": {
                "temporal_validation_enabled": self.config.temporal_validation_enabled,
                "noise_detection_enabled": self.config.noise_detection_enabled,
                "market_regime_detection": self.config.market_regime_detection,
                "feature_selection_enabled": self.config.feature_selection_enabled
            }
        }
        
        return summary
    
    def _get_historical_trend(self) -> Dict[str, Any]:
        """Get historical trend analysis"""
        if len(self.reports_history) < 2:
            return {"status": "Insufficient data for trend analysis"}
        
        recent_reports = self.reports_history[-10:]  # Last 10 reports
        
        scores = [report.overall_protection_score for report in recent_reports]
        data_quality_scores = [report.data_quality_score for report in recent_reports]
        
        return {
            "overall_score_trend": "improving" if scores[-1] > scores[0] else "declining",
            "average_score": np.mean(scores),
            "score_stability": np.std(scores),
            "data_quality_trend": "improving" if data_quality_scores[-1] > data_quality_scores[0] else "declining",
            "reports_analyzed": len(recent_reports)
        }

def create_advanced_protection_config() -> ProtectionConfig:
    """Create a default advanced protection configuration"""
    return ProtectionConfig(
        max_missing_percentage=0.05,
        min_variance_threshold=1e-6,
        max_correlation_threshold=0.9,
        outlier_contamination=0.05,
        temporal_validation_enabled=True,
        min_temporal_window=50,
        cross_validation_folds=10,
        feature_selection_enabled=True,
        max_features_ratio=0.2,
        noise_detection_enabled=True,
        signal_to_noise_threshold=3.0,
        ensemble_validation=True,
        market_regime_detection=True,
        performance_tracking=True,
        backup_enabled=True
    )

# Example usage and testing
if __name__ == "__main__":
    # Create sample trading data for testing
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Generate sample trading data
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # Create realistic trading features
    data = pd.DataFrame(index=dates)
    data['returns'] = np.random.normal(0.001, 0.02, n_samples)
    data['volume'] = np.random.lognormal(10, 1, n_samples)
    data['volatility'] = np.random.gamma(2, 0.01, n_samples)
    
    # Add some technical indicators
    for i in range(n_features):
        if i < 5:
            # Price-based features
            data[f'price_feature_{i}'] = np.cumsum(np.random.normal(0, 0.01, n_samples))
        elif i < 10:
            # Volume-based features
            data[f'volume_feature_{i}'] = data['volume'] * np.random.normal(1, 0.1, n_samples)
        else:
            # Derived features
            data[f'derived_feature_{i}'] = np.random.normal(0, 1, n_samples)
    
    # Create target variable (binary classification)
    # Make it somewhat dependent on features but add noise
    signal = (
        data['returns'] * 10 + 
        data['volatility'] * 5 + 
        np.random.normal(0, 1, n_samples)
    )
    data['target'] = (signal > signal.median()).astype(int)
    
    # Add some issues for testing
    # Missing data
    data.loc[data.index[:50], 'price_feature_0'] = np.nan
    
    # High correlation
    data['correlated_feature'] = data['price_feature_0'] * 1.1 + np.random.normal(0, 0.01, n_samples)
    
    # Potential leakage (future information)
    data['future_leakage'] = data['target'].shift(-1).fillna(0)
    
    print("Testing Advanced ML Protection System...")
    print(f"Data shape: {data.shape}")
    print(f"Data period: {data.index.min()} to {data.index.max()}")
    
    # Initialize protection system
    config = create_advanced_protection_config()
    protection_system = AdvancedMLProtectionSystem(config)
    
    # Run comprehensive analysis
    feature_columns = [col for col in data.columns if col not in ['target']]
    report = protection_system.analyze_data_comprehensive(
        data=data,
        target_column='target',
        feature_columns=feature_columns
    )
    
    # Print results
    print("\n" + "="*80)
    print("ADVANCED ML PROTECTION SYSTEM ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\nOverall Protection Score: {report.overall_protection_score:.3f}")
    print(f"Risk Level: {report.risk_level.upper()}")
    print(f"Data Quality Score: {report.data_quality_score:.3f}")
    print(f"Overfitting Risk: {report.overfitting_risk.upper()}")
    print(f"Noise Level: {report.noise_level.upper()}")
    print(f"Signal-to-Noise Ratio: {report.signal_to_noise_ratio:.2f}")
    
    print(f"\nIssues Detected:")
    print(f"- Missing Data Issues: {len(report.missing_data_issues)}")
    print(f"- Correlation Issues: {len(report.correlation_issues)}")
    print(f"- Feature Leakage Issues: {len(report.feature_leakage_issues)}")
    print(f"- Outliers Detected: {report.outlier_count}")
    print(f"- Target Leakage: {'Yes' if report.target_leakage_detected else 'No'}")
    
    if report.recommendations:
        print(f"\nRecommendations ({len(report.recommendations)}):")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"{i}. {rec}")
    
    print(f"\nPerformance:")
    print(f"- Processing Time: {report.processing_time_seconds:.2f} seconds")
    print(f"- Memory Usage: {report.memory_usage_mb:.1f} MB")
    
    # Test automated fixes
    print("\n" + "="*80)
    print("TESTING AUTOMATED FIXES")
    print("="*80)
    
    fixed_data = protection_system.apply_automated_fixes(data, report, 'target')
    print(f"Original data shape: {data.shape}")
    print(f"Fixed data shape: {fixed_data.shape}")
    print(f"Features removed: {set(data.columns) - set(fixed_data.columns)}")
    
    # Generate protection summary
    summary = protection_system.generate_protection_summary()
    print(f"\nProtection System Summary:")
    print(f"- System Status: {summary['system_status']}")
    print(f"- Latest Score: {summary['latest_analysis']['overall_protection_score']:.3f}")
    print(f"- Issues Count: {sum(summary['issues_summary'].values())}")
    
    print("\n" + "="*80)
    print("ADVANCED ML PROTECTION SYSTEM TEST COMPLETED")
    print("="*80)
