"""
ProjectP Integration with Advanced ML Protection System
======================================================

This module provides seamless integration between ProjectP trading pipeline
and the Advanced ML Protection System for comprehensive ML model protection.

Author: AI Assistant
Version: 2.0.0
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import yaml
from datetime import datetime, timedelta
import warnings

# Import the advanced protection system
from advanced_ml_protection_system import AdvancedMLProtectionSystem, ProtectionConfig, ProtectionReport

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProjectPProtectionIntegration:
    """
    Integration layer between ProjectP and Advanced ML Protection System
    
    Features:
    - Automated protection analysis for ProjectP data
    - Real-time monitoring during model training
    - Automated issue detection and remediation
    - Performance tracking and alerting
    - Seamless integration with existing ProjectP workflow
    """
    
    def __init__(
        self, 
        config_path: Optional[str] = None,
        projectp_config_path: Optional[str] = 'config.yaml'
    ):
        """Initialize ProjectP Protection Integration"""
        
        # Load configurations
        self.protection_config = self._load_protection_config(config_path)
        self.projectp_config = self._load_projectp_config(projectp_config_path)
        
        # Initialize protection system
        self.protection_system = AdvancedMLProtectionSystem(self.protection_config)
        
        # Integration state
        self.last_analysis_report = None
        self.monitoring_active = False
        self.auto_remediation_enabled = True
        self.performance_history = []
        
        logger.info("ProjectP Protection Integration initialized")
    
    def _load_protection_config(self, config_path: Optional[str]) -> ProtectionConfig:
        """Load protection configuration"""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
                
                # Convert YAML config to ProtectionConfig
                return ProtectionConfig(
                    max_missing_percentage=config_dict.get('data_quality', {}).get('max_missing_percentage', 0.05),
                    min_variance_threshold=config_dict.get('data_quality', {}).get('min_variance_threshold', 1e-6),
                    max_correlation_threshold=config_dict.get('data_quality', {}).get('max_correlation_threshold', 0.9),
                    temporal_validation_enabled=config_dict.get('temporal_validation', {}).get('enabled', True),
                    cross_validation_folds=config_dict.get('overfitting_protection', {}).get('cross_validation_folds', 10),
                    noise_detection_enabled=config_dict.get('noise_reduction', {}).get('enabled', True),
                    signal_to_noise_threshold=config_dict.get('noise_reduction', {}).get('signal_to_noise_threshold', 3.0),
                    market_regime_detection=config_dict.get('advanced_features', {}).get('market_regime_detection', True),
                    performance_tracking=config_dict.get('monitoring', {}).get('performance_tracking', True)
                )
            except Exception as e:
                logger.warning(f"Could not load protection config from {config_path}: {str(e)}")
        
        # Return default config
        return ProtectionConfig()
    
    def _load_projectp_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load ProjectP configuration"""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Could not load ProjectP config from {config_path}: {str(e)}")
        
        return {}
    
    def analyze_projectp_data(
        self, 
        data: pd.DataFrame,
        target_column: str = 'target',
        timeframe: str = 'M15',
        market_data: bool = True
    ) -> ProtectionReport:
        """
        Analyze ProjectP data with protection system
        
        Args:
            data: ProjectP data DataFrame
            target_column: Name of target column
            timeframe: Trading timeframe (M1, M5, M15, H1, etc.)
            market_data: Whether this is market/trading data
            
        Returns:
            ProtectionReport with analysis results
        """
        logger.info(f"Analyzing ProjectP data for {timeframe} timeframe...")
        
        try:
            # Prepare feature columns (exclude target and metadata columns)
            metadata_columns = ['timestamp', 'symbol', 'timeframe', 'date', 'time']
            feature_columns = [
                col for col in data.columns 
                if col != target_column and col not in metadata_columns
            ]
            
            # Enhance protection config for trading data
            if market_data:
                self._enhance_config_for_trading_data(timeframe)
            
            # Run comprehensive analysis
            report = self.protection_system.analyze_data_comprehensive(
                data=data,
                target_column=target_column,
                feature_columns=feature_columns
            )
            
            # Add ProjectP-specific analysis
            self._add_projectp_specific_analysis(data, report, timeframe)
            
            # Store latest report
            self.last_analysis_report = report
            
            # Trigger alerts if necessary
            self._check_and_trigger_alerts(report)
            
            logger.info(f"ProjectP data analysis completed. Score: {report.overall_protection_score:.3f}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error analyzing ProjectP data: {str(e)}")
            raise
    
    def _enhance_config_for_trading_data(self, timeframe: str):
        """Enhance protection config for trading data specifics"""
        
        # Adjust temporal validation based on timeframe
        timeframe_minutes = self._timeframe_to_minutes(timeframe)
        
        if timeframe_minutes <= 5:  # High frequency data
            self.protection_config.min_temporal_window = 100
            self.protection_config.outlier_contamination = 0.02
            self.protection_config.signal_to_noise_threshold = 2.0
        elif timeframe_minutes <= 60:  # Medium frequency data
            self.protection_config.min_temporal_window = 50
            self.protection_config.outlier_contamination = 0.05
            self.protection_config.signal_to_noise_threshold = 2.5
        else:  # Daily or longer
            self.protection_config.min_temporal_window = 30
            self.protection_config.outlier_contamination = 0.1
            self.protection_config.signal_to_noise_threshold = 3.0
        
        # Enable market-specific features
        self.protection_config.market_regime_detection = True
        self.protection_config.volatility_clustering_check = True
        self.protection_config.trend_consistency_check = True
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        timeframe_map = {
            'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
            'H1': 60, 'H4': 240, 'D1': 1440, 'W1': 10080
        }
        return timeframe_map.get(timeframe, 15)
    
    def _add_projectp_specific_analysis(
        self, 
        data: pd.DataFrame, 
        report: ProtectionReport, 
        timeframe: str
    ):
        """Add ProjectP-specific analysis to the report"""
        
        # Trading-specific validations
        trading_issues = []
        
        # Check for forward-looking bias in features
        forward_bias_features = [
            col for col in data.columns 
            if any(keyword in col.lower() for keyword in ['future', 'next', 'ahead', 'forward'])
        ]
        
        if forward_bias_features:
            trading_issues.append(f"Potential forward-looking bias in features: {forward_bias_features}")
        
        # Check for weekend data issues (if trading data)
        if 'timestamp' in data.columns or isinstance(data.index, pd.DatetimeIndex):
            date_col = data.index if isinstance(data.index, pd.DatetimeIndex) else data['timestamp']
            weekend_data = date_col[date_col.weekday >= 5]  # Saturday, Sunday
            
            if len(weekend_data) > 0:
                trading_issues.append(f"Found {len(weekend_data)} weekend data points (may be problematic for trading)")
        
        # Check for excessive volatility in returns
        if 'returns' in data.columns:
            returns = data['returns'].dropna()
            extreme_returns = returns[np.abs(returns) > returns.std() * 5]
            
            if len(extreme_returns) > len(returns) * 0.01:  # More than 1% extreme returns
                trading_issues.append(f"Excessive extreme returns detected: {len(extreme_returns)} events")
        
        # Add trading issues to report
        if trading_issues:
            report.feature_leakage_issues.extend(trading_issues)
        
        # Add timeframe-specific recommendations
        if timeframe in ['M1', 'M5']:
            report.recommendations.append(
                "High-frequency data detected. Consider noise reduction and robust feature engineering."
            )
        elif timeframe in ['D1', 'W1']:
            report.recommendations.append(
                "Daily/weekly data detected. Ensure sufficient historical data for robust modeling."
            )
    
    def _check_and_trigger_alerts(self, report: ProtectionReport):
        """Check report for issues and trigger alerts if necessary"""
        critical_issues = []
        
        # Check for critical issues
        if report.overall_protection_score < 0.4:
            critical_issues.append(f"Very low protection score: {report.overall_protection_score:.3f}")
        
        if report.target_leakage_detected:
            critical_issues.append("Target leakage detected!")
        
        if report.data_quality_score < 0.5:
            critical_issues.append(f"Poor data quality: {report.data_quality_score:.3f}")
        
        if report.overfitting_risk == 'high':
            critical_issues.append("High overfitting risk detected")
        
        # Log critical issues
        if critical_issues:
            logger.warning("CRITICAL ISSUES DETECTED:")
            for issue in critical_issues:
                logger.warning(f"  - {issue}")
        
        # TODO: Add email/webhook alerts here if configured
    
    def apply_projectp_fixes(
        self, 
        data: pd.DataFrame, 
        target_column: str = 'target',
        auto_apply: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply automated fixes for ProjectP data
        
        Args:
            data: Input DataFrame
            target_column: Name of target column
            auto_apply: Whether to auto-apply fixes
            
        Returns:
            Tuple of (fixed_data, fix_summary)
        """
        logger.info("Applying ProjectP-specific fixes...")
        
        if self.last_analysis_report is None:
            # Run analysis first
            self.analyze_projectp_data(data, target_column)
        
        # Apply base protection system fixes
        fixed_data = self.protection_system.apply_automated_fixes(
            data, self.last_analysis_report, target_column
        )
        
        # Apply ProjectP-specific fixes
        fix_summary = {'base_fixes_applied': True, 'projectp_fixes': []}
        
        try:
            # Remove weekend data for trading datasets
            if isinstance(fixed_data.index, pd.DatetimeIndex):
                weekend_mask = fixed_data.index.weekday < 5  # Monday=0, Friday=4
                if weekend_mask.sum() < len(fixed_data):
                    fixed_data = fixed_data[weekend_mask]
                    fix_summary['projectp_fixes'].append(f"Removed {(~weekend_mask).sum()} weekend data points")
            
            # Handle extreme returns
            if 'returns' in fixed_data.columns:
                returns_col = 'returns'
                returns = fixed_data[returns_col]
                
                # Cap extreme returns at 5 standard deviations
                returns_std = returns.std()
                returns_mean = returns.mean()
                
                lower_bound = returns_mean - 5 * returns_std
                upper_bound = returns_mean + 5 * returns_std
                
                extreme_count = ((returns < lower_bound) | (returns > upper_bound)).sum()
                
                if extreme_count > 0:
                    fixed_data[returns_col] = np.clip(returns, lower_bound, upper_bound)
                    fix_summary['projectp_fixes'].append(f"Capped {extreme_count} extreme returns")
            
            # Forward-fill missing values for time series continuity
            if isinstance(fixed_data.index, pd.DatetimeIndex):
                numeric_columns = fixed_data.select_dtypes(include=[np.number]).columns
                for col in numeric_columns:
                    if col != target_column and fixed_data[col].isnull().any():
                        before_count = fixed_data[col].isnull().sum()
                        fixed_data[col] = fixed_data[col].fillna(method='ffill').fillna(method='bfill')
                        after_count = fixed_data[col].isnull().sum()
                        if before_count > after_count:
                            fix_summary['projectp_fixes'].append(
                                f"Forward-filled {before_count - after_count} missing values in {col}"
                            )
            
            logger.info(f"ProjectP fixes applied: {len(fix_summary['projectp_fixes'])} specific fixes")
            
        except Exception as e:
            logger.error(f"Error applying ProjectP fixes: {str(e)}")
            fix_summary['projectp_fixes'].append(f"Error: {str(e)}")
        
        return fixed_data, fix_summary
    
    def monitor_projectp_pipeline(
        self, 
        data: pd.DataFrame,
        model_performance: Dict[str, float],
        target_column: str = 'target'
    ) -> Dict[str, Any]:
        """
        Monitor ProjectP pipeline during execution
        
        Args:
            data: Current data being processed
            model_performance: Current model performance metrics
            target_column: Name of target column
            
        Returns:
            Monitoring report
        """
        logger.info("Monitoring ProjectP pipeline...")
        
        monitoring_report = {
            'timestamp': datetime.now().isoformat(),
            'data_health': {},
            'model_health': {},
            'alerts': [],
            'recommendations': []
        }
        
        try:
            # Quick data health check
            monitoring_report['data_health'] = {
                'sample_count': len(data),
                'feature_count': len([col for col in data.columns if col != target_column]),
                'missing_percentage': data.isnull().mean().mean(),
                'target_distribution': data[target_column].value_counts().to_dict() if target_column in data.columns else {}
            }
            
            # Model performance monitoring
            if model_performance:
                monitoring_report['model_health'] = model_performance.copy()
                
                # Check for performance degradation
                auc_score = model_performance.get('auc', 0)
                if auc_score < self.protection_config.alert_threshold_auc:
                    monitoring_report['alerts'].append(
                        f"Model performance below threshold: AUC {auc_score:.3f} < {self.protection_config.alert_threshold_auc}"
                    )
            
            # Data quality alerts
            missing_pct = monitoring_report['data_health']['missing_percentage']
            if missing_pct > self.protection_config.max_missing_percentage:
                monitoring_report['alerts'].append(
                    f"High missing data percentage: {missing_pct:.2%}"
                )
            
            # Add to performance history
            self.performance_history.append({
                'timestamp': datetime.now(),
                'data_samples': len(data),
                'model_performance': model_performance,
                'alerts_count': len(monitoring_report['alerts'])
            })
            
            # Keep only recent history
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
            
            logger.info(f"Pipeline monitoring completed. Alerts: {len(monitoring_report['alerts'])}")
            
        except Exception as e:
            logger.error(f"Error in pipeline monitoring: {str(e)}")
            monitoring_report['alerts'].append(f"Monitoring error: {str(e)}")
        
        return monitoring_report
    
    def generate_projectp_protection_report(self) -> Dict[str, Any]:
        """Generate comprehensive protection report for ProjectP"""
        
        report = {
            'integration_status': 'active',
            'timestamp': datetime.now().isoformat(),
            'protection_system_info': {},
            'projectp_specific_metrics': {},
            'recommendations': [],
            'performance_history': self.performance_history[-10:] if self.performance_history else []
        }
        
        # Protection system summary
        if self.protection_system:
            protection_summary = self.protection_system.generate_protection_summary()
            report['protection_system_info'] = protection_summary
        
        # Latest analysis summary
        if self.last_analysis_report:
            report['latest_analysis'] = {
                'timestamp': self.last_analysis_report.timestamp.isoformat(),
                'overall_score': self.last_analysis_report.overall_protection_score,
                'risk_level': self.last_analysis_report.risk_level,
                'issues_count': (
                    len(self.last_analysis_report.missing_data_issues) +
                    len(self.last_analysis_report.correlation_issues) +
                    len(self.last_analysis_report.feature_leakage_issues)
                ),
                'recommendations_count': len(self.last_analysis_report.recommendations)
            }
        
        # ProjectP-specific metrics
        if self.performance_history:
            recent_performance = self.performance_history[-5:]
            
            avg_alerts = np.mean([p['alerts_count'] for p in recent_performance])
            avg_samples = np.mean([p['data_samples'] for p in recent_performance])
            
            report['projectp_specific_metrics'] = {
                'average_alerts_recent': avg_alerts,
                'average_data_samples': avg_samples,
                'monitoring_periods': len(self.performance_history),
                'auto_remediation_enabled': self.auto_remediation_enabled
            }
        
        # Integration recommendations
        if report.get('protection_system_info', {}).get('latest_analysis', {}).get('risk_level') == 'high':
            report['recommendations'].append("High risk detected: Review data quality and model configuration")
        
        if self.last_analysis_report and self.last_analysis_report.overfitting_risk == 'high':
            report['recommendations'].append("High overfitting risk: Implement regularization and cross-validation")
        
        return report
    
    def save_integration_state(self, filepath: str = 'projectp_protection_state.json'):
        """Save integration state for persistence"""
        try:
            state = {
                'auto_remediation_enabled': self.auto_remediation_enabled,
                'monitoring_active': self.monitoring_active,
                'performance_history_count': len(self.performance_history),
                'last_analysis_timestamp': self.last_analysis_report.timestamp.isoformat() if self.last_analysis_report else None,
                'config_summary': {
                    'temporal_validation_enabled': self.protection_config.temporal_validation_enabled,
                    'noise_detection_enabled': self.protection_config.noise_detection_enabled,
                    'market_regime_detection': self.protection_config.market_regime_detection
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Integration state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving integration state: {str(e)}")
    
    def load_integration_state(self, filepath: str = 'projectp_protection_state.json'):
        """Load integration state from file"""
        try:
            if Path(filepath).exists():
                with open(filepath, 'r') as f:
                    state = json.load(f)
                
                self.auto_remediation_enabled = state.get('auto_remediation_enabled', True)
                self.monitoring_active = state.get('monitoring_active', False)
                
                logger.info(f"Integration state loaded from {filepath}")
            else:
                logger.info("No existing state file found, using defaults")
                
        except Exception as e:
            logger.error(f"Error loading integration state: {str(e)}")

def create_projectp_integration(
    protection_config_path: Optional[str] = 'advanced_ml_protection_config.yaml',
    projectp_config_path: Optional[str] = 'config.yaml'
) -> ProjectPProtectionIntegration:
    """
    Create ProjectP Protection Integration with configurations
    
    Args:
        protection_config_path: Path to protection configuration file
        projectp_config_path: Path to ProjectP configuration file
        
    Returns:
        Configured ProjectPProtectionIntegration instance
    """
    return ProjectPProtectionIntegration(
        config_path=protection_config_path,
        projectp_config_path=projectp_config_path
    )

# Example usage and testing
if __name__ == "__main__":
    print("Testing ProjectP Protection Integration...")
    
    # Create sample ProjectP data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic trading data
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='15T')  # 15-minute data
    
    # Create ProjectP-style DataFrame
    data = pd.DataFrame(index=dates)
    
    # Trading features
    data['open'] = 100 + np.cumsum(np.random.normal(0, 0.1, n_samples))
    data['high'] = data['open'] + np.random.gamma(1, 0.05, n_samples)
    data['low'] = data['open'] - np.random.gamma(1, 0.05, n_samples)
    data['close'] = data['open'] + np.random.normal(0, 0.05, n_samples)
    data['volume'] = np.random.lognormal(10, 1, n_samples)
    
    # Technical indicators
    data['sma_20'] = data['close'].rolling(20).mean()
    data['rsi'] = 50 + np.random.normal(0, 15, n_samples)
    data['macd'] = np.random.normal(0, 0.1, n_samples)
    data['bb_upper'] = data['close'] + np.random.normal(2, 0.2, n_samples)
    data['bb_lower'] = data['close'] - np.random.normal(2, 0.2, n_samples)
    
    # Returns
    data['returns'] = data['close'].pct_change()
    
    # Target variable (binary classification for trade signal)
    data['target'] = (data['returns'].shift(-1) > data['returns'].median()).astype(int)
    
    # Add some issues for testing
    data.loc[data.index[:50], 'sma_20'] = np.nan  # Missing data
    data['correlated_feature'] = data['close'] * 1.05 + np.random.normal(0, 0.01, n_samples)  # High correlation
    
    # Drop NaN values
    data = data.dropna()
    
    print(f"Generated ProjectP test data: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(f"Features: {[col for col in data.columns if col != 'target']}")
    
    # Initialize integration
    integration = create_projectp_integration()
    
    # Test analysis
    print("\n" + "="*60)
    print("RUNNING PROJECTP PROTECTION ANALYSIS")
    print("="*60)
    
    report = integration.analyze_projectp_data(
        data=data,
        target_column='target',
        timeframe='M15',
        market_data=True
    )
    
    print(f"\nAnalysis Results:")
    print(f"- Overall Protection Score: {report.overall_protection_score:.3f}")
    print(f"- Risk Level: {report.risk_level.upper()}")
    print(f"- Data Quality Score: {report.data_quality_score:.3f}")
    print(f"- Overfitting Risk: {report.overfitting_risk.upper()}")
    print(f"- Noise Level: {report.noise_level.upper()}")
    print(f"- Issues Found: {len(report.missing_data_issues) + len(report.correlation_issues) + len(report.feature_leakage_issues)}")
    
    # Test fixes
    print("\n" + "="*60)
    print("APPLYING PROJECTP FIXES")
    print("="*60)
    
    fixed_data, fix_summary = integration.apply_projectp_fixes(data, 'target')
    
    print(f"Original data shape: {data.shape}")
    print(f"Fixed data shape: {fixed_data.shape}")
    print(f"Base fixes applied: {fix_summary['base_fixes_applied']}")
    print(f"ProjectP-specific fixes: {len(fix_summary['projectp_fixes'])}")
    
    for fix in fix_summary['projectp_fixes']:
        print(f"  - {fix}")
    
    # Test monitoring
    print("\n" + "="*60)
    print("TESTING PIPELINE MONITORING")
    print("="*60)
    
    # Simulate model performance
    model_performance = {
        'auc': 0.75,
        'accuracy': 0.68,
        'precision': 0.72,
        'recall': 0.65
    }
    
    monitoring_report = integration.monitor_projectp_pipeline(
        data=fixed_data,
        model_performance=model_performance,
        target_column='target'
    )
    
    print(f"Monitoring Report:")
    print(f"- Data samples: {monitoring_report['data_health']['sample_count']}")
    print(f"- Feature count: {monitoring_report['data_health']['feature_count']}")
    print(f"- Missing data: {monitoring_report['data_health']['missing_percentage']:.2%}")
    print(f"- Model AUC: {model_performance['auc']:.3f}")
    print(f"- Alerts: {len(monitoring_report['alerts'])}")
    
    # Generate comprehensive report
    comprehensive_report = integration.generate_projectp_protection_report()
    
    print(f"\nComprehensive Report Summary:")
    print(f"- Integration Status: {comprehensive_report['integration_status']}")
    print(f"- Latest Analysis Score: {comprehensive_report.get('latest_analysis', {}).get('overall_score', 'N/A')}")
    print(f"- Performance History Entries: {len(comprehensive_report['performance_history'])}")
    print(f"- Recommendations: {len(comprehensive_report['recommendations'])}")
    
    # Save state
    integration.save_integration_state()
    
    print("\n" + "="*60)
    print("PROJECTP PROTECTION INTEGRATION TEST COMPLETED")
    print("="*60)
