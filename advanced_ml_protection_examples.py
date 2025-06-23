"""
Advanced ML Protection Examples and Use Cases
============================================

Comprehensive examples showing how to use the Advanced ML Protection System
for trading ML pipelines with real-world scenarios and best practices.

Author: AI Assistant
Version: 2.0.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import logging
from pathlib import Path

# Import protection system components
from advanced_ml_protection_system import AdvancedMLProtectionSystem, ProtectionConfig, ProtectionReport
from projectp_advanced_protection_integration import ProjectPProtectionIntegration

# Configure logging and warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class ProtectionExamples:
    """
    Comprehensive examples for Advanced ML Protection System
    
    This class provides real-world examples and use cases for:
    - Data quality analysis for trading data
    - Overfitting prevention in ML models
    - Noise reduction and signal enhancement
    - Data leakage detection and prevention
    - Market regime-aware validation
    - Performance monitoring and alerting
    """
    
    def __init__(self):
        """Initialize protection examples"""
        self.setup_directories()
    
    def setup_directories(self):
        """Setup directories for examples"""
        directories = ['examples_output', 'examples_data', 'examples_reports']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    def example_1_basic_data_protection(self):
        """
        Example 1: Basic Data Protection for Trading Data
        
        Demonstrates:
        - Loading and preprocessing trading data
        - Running basic protection analysis
        - Interpreting protection results
        - Applying automated fixes
        """
        print("\n" + "="*80)
        print("EXAMPLE 1: BASIC DATA PROTECTION FOR TRADING DATA")
        print("="*80)
        
        # Generate sample trading data with intentional issues
        data = self.generate_sample_trading_data_with_issues()
        print(f"📊 Generated sample data: {data.shape}")
        
        # Initialize protection system with default config
        protection_system = AdvancedMLProtectionSystem()
        
        # Run comprehensive analysis
        print("\n🔍 Running protection analysis...")
        report = protection_system.analyze_data_comprehensive(
            data=data,
            target_column='target',
            feature_columns=[col for col in data.columns if col != 'target']
        )
        
        # Display results
        self.display_protection_results(report, "Basic Protection Analysis")
        
        # Apply automated fixes
        print("\n🔧 Applying automated fixes...")
        fixed_data = protection_system.apply_automated_fixes(data, report, 'target')
        
        print(f"Original data shape: {data.shape}")
        print(f"Fixed data shape: {fixed_data.shape}")
        
        # Save results
        data.to_csv('examples_output/example1_original_data.csv', index=False)
        fixed_data.to_csv('examples_output/example1_fixed_data.csv', index=False)
        
        print("✅ Example 1 completed successfully")
        return data, fixed_data, report
    
    def example_2_projectp_integration(self):
        """
        Example 2: ProjectP Integration with Protection System
        
        Demonstrates:
        - Setting up ProjectP protection integration
        - Analyzing real trading pipeline data
        - Monitoring pipeline performance
        - Generating comprehensive reports
        """
        print("\n" + "="*80)
        print("EXAMPLE 2: PROJECTP INTEGRATION")
        print("="*80)
        
        # Generate realistic ProjectP data
        projectp_data = self.generate_realistic_projectp_data()
        print(f"📈 Generated ProjectP data: {projectp_data.shape}")
        
        # Initialize ProjectP integration
        integration = ProjectPProtectionIntegration()
        
        # Analyze ProjectP data
        print("\n🔍 Analyzing ProjectP data...")
        report = integration.analyze_projectp_data(
            data=projectp_data,
            target_column='target',
            timeframe='M15',
            market_data=True
        )
        
        # Display results
        self.display_protection_results(report, "ProjectP Analysis")
        
        # Apply ProjectP-specific fixes
        print("\n🔧 Applying ProjectP fixes...")
        fixed_data, fix_summary = integration.apply_projectp_fixes(
            projectp_data, 'target'
        )
        
        print(f"ProjectP fixes applied: {len(fix_summary['projectp_fixes'])}")
        for fix in fix_summary['projectp_fixes']:
            print(f"  • {fix}")
        
        # Monitor pipeline performance
        print("\n📊 Monitoring pipeline...")
        model_performance = {
            'auc': 0.72,
            'accuracy': 0.68,
            'precision': 0.70,
            'recall': 0.66,
            'f1': 0.68
        }
        
        monitoring_report = integration.monitor_projectp_pipeline(
            data=fixed_data,
            model_performance=model_performance,
            target_column='target'
        )
        
        print(f"Monitoring alerts: {len(monitoring_report['alerts'])}")
        for alert in monitoring_report['alerts']:
            print(f"  ⚠️  {alert}")
        
        # Generate comprehensive report
        comprehensive_report = integration.generate_projectp_protection_report()
        
        # Save results
        projectp_data.to_csv('examples_output/example2_projectp_data.csv', index=False)
        fixed_data.to_csv('examples_output/example2_projectp_fixed.csv', index=False)
        
        print("✅ Example 2 completed successfully")
        return projectp_data, fixed_data, comprehensive_report
    
    def example_3_advanced_noise_detection(self):
        """
        Example 3: Advanced Noise Detection and Signal Enhancement
        
        Demonstrates:
        - Detecting different types of noise in trading data
        - Signal-to-noise ratio analysis
        - Advanced denoising techniques
        - Market regime-aware filtering
        """
        print("\n" + "="*80)
        print("EXAMPLE 3: ADVANCED NOISE DETECTION")
        print("="*80)
        
        # Generate noisy trading data
        noisy_data = self.generate_noisy_trading_data()
        print(f"📊 Generated noisy trading data: {noisy_data.shape}")
        
        # Configure protection system for noise detection
        noise_config = ProtectionConfig(
            noise_detection_enabled=True,
            signal_to_noise_threshold=2.0,
            denoising_method='robust_scaler',
            market_regime_detection=True,
            volatility_clustering_check=True
        )
        
        protection_system = AdvancedMLProtectionSystem(noise_config)
        
        # Analyze noise levels
        print("\n🔍 Analyzing noise levels...")
        report = protection_system.analyze_data_comprehensive(
            data=noisy_data,
            target_column='target',
            feature_columns=[col for col in noisy_data.columns if col != 'target']
        )
        
        # Display noise analysis results
        print(f"\n📊 Noise Analysis Results:")
        print(f"  • Signal-to-Noise Ratio: {report.signal_to_noise_ratio:.2f}")
        print(f"  • Noise Level: {report.noise_level}")
        print(f"  • Denoising Recommendations: {len(report.denoising_recommendations)}")
        
        for rec in report.denoising_recommendations:
            print(f"    - {rec}")
        
        # Apply denoising
        print("\n🔧 Applying denoising techniques...")
        denoised_data = protection_system.apply_automated_fixes(noisy_data, report, 'target')
        
        # Compare before and after
        self.plot_noise_comparison(noisy_data, denoised_data)
        
        # Save results
        noisy_data.to_csv('examples_output/example3_noisy_data.csv', index=False)
        denoised_data.to_csv('examples_output/example3_denoised_data.csv', index=False)
        
        print("✅ Example 3 completed successfully")
        return noisy_data, denoised_data, report
    
    def example_4_data_leakage_detection(self):
        """
        Example 4: Data Leakage Detection and Prevention
        
        Demonstrates:
        - Detecting various types of data leakage
        - Target leakage identification
        - Temporal leakage in time series
        - Feature stability analysis
        """
        print("\n" + "="*80)
        print("EXAMPLE 4: DATA LEAKAGE DETECTION")
        print("="*80)
        
        # Generate data with intentional leakage
        leaky_data = self.generate_data_with_leakage()
        print(f"📊 Generated data with leakage: {leaky_data.shape}")
        
        # Configure protection system for leakage detection
        leakage_config = ProtectionConfig(
            target_leakage_check=True,
            temporal_leakage_check=True,
            future_data_check=True,
            feature_stability_check=True,
            max_correlation_threshold=0.95
        )
        
        protection_system = AdvancedMLProtectionSystem(leakage_config)
        
        # Analyze for leakage
        print("\n🔍 Detecting data leakage...")
        report = protection_system.analyze_data_comprehensive(
            data=leaky_data,
            target_column='target',
            feature_columns=[col for col in leaky_data.columns if col != 'target']
        )
        
        # Display leakage detection results
        print(f"\n🚨 Leakage Detection Results:")
        print(f"  • Target Leakage Detected: {report.target_leakage_detected}")
        print(f"  • Temporal Leakage Detected: {report.temporal_leakage_detected}")
        print(f"  • Feature Leakage Issues: {len(report.feature_leakage_issues)}")
        
        for issue in report.feature_leakage_issues:
            print(f"    - {issue}")
        
        # Clean leaky features
        print("\n🔧 Removing leaky features...")
        clean_data = self.remove_leaky_features(leaky_data, report)
        
        print(f"Original features: {leaky_data.shape[1] - 1}")
        print(f"Clean features: {clean_data.shape[1] - 1}")
        print(f"Features removed: {leaky_data.shape[1] - clean_data.shape[1]}")
        
        # Save results
        leaky_data.to_csv('examples_output/example4_leaky_data.csv', index=False)
        clean_data.to_csv('examples_output/example4_clean_data.csv', index=False)
        
        print("✅ Example 4 completed successfully")
        return leaky_data, clean_data, report
    
    def example_5_overfitting_prevention(self):
        """
        Example 5: Overfitting Prevention and Model Validation
        
        Demonstrates:
        - Overfitting risk assessment
        - Cross-validation strategies for time series
        - Model complexity analysis
        - Feature selection for overfitting prevention
        """
        print("\n" + "="*80)
        print("EXAMPLE 5: OVERFITTING PREVENTION")
        print("="*80)
        
        # Generate data prone to overfitting
        overfit_data = self.generate_overfitting_prone_data()
        print(f"📊 Generated overfitting-prone data: {overfit_data.shape}")
        
        # Configure protection system for overfitting detection
        overfit_config = ProtectionConfig(
            cross_validation_folds=10,
            feature_selection_enabled=True,
            max_features_ratio=0.2,
            ensemble_validation=True,
            early_stopping_patience=10
        )
        
        protection_system = AdvancedMLProtectionSystem(overfit_config)
        
        # Analyze overfitting risk
        print("\n🔍 Assessing overfitting risk...")
        report = protection_system.analyze_data_comprehensive(
            data=overfit_data,
            target_column='target',
            feature_columns=[col for col in overfit_data.columns if col != 'target']
        )
        
        # Display overfitting analysis results
        print(f"\n⚠️  Overfitting Risk Assessment:")
        print(f"  • Risk Level: {report.overfitting_risk}")
        print(f"  • Model Complexity Score: {report.model_complexity_score:.3f}")
        print(f"  • Cross-Validation Scores: {[f'{score:.3f}' for score in report.cross_validation_scores]}")
        
        if report.cross_validation_scores:
            cv_mean = np.mean(report.cross_validation_scores)
            cv_std = np.std(report.cross_validation_scores)
            print(f"  • CV Mean ± Std: {cv_mean:.3f} ± {cv_std:.3f}")
        
        # Apply overfitting prevention
        print("\n🔧 Applying overfitting prevention...")
        protected_data = self.apply_overfitting_prevention(overfit_data, report)
        
        print(f"Original features: {overfit_data.shape[1] - 1}")
        print(f"Selected features: {protected_data.shape[1] - 1}")
        print(f"Feature reduction: {((overfit_data.shape[1] - protected_data.shape[1]) / overfit_data.shape[1]) * 100:.1f}%")
        
        # Save results
        overfit_data.to_csv('examples_output/example5_overfit_data.csv', index=False)
        protected_data.to_csv('examples_output/example5_protected_data.csv', index=False)
        
        print("✅ Example 5 completed successfully")
        return overfit_data, protected_data, report
    
    def example_6_market_regime_analysis(self):
        """
        Example 6: Market Regime-Aware Protection
        
        Demonstrates:
        - Market regime detection
        - Volatility clustering analysis
        - Regime-specific model validation
        - Adaptive protection thresholds
        """
        print("\n" + "="*80)
        print("EXAMPLE 6: MARKET REGIME-AWARE PROTECTION")
        print("="*80)
        
        # Generate multi-regime market data
        regime_data = self.generate_multi_regime_data()
        print(f"📊 Generated multi-regime data: {regime_data.shape}")
        
        # Configure protection system for regime analysis
        regime_config = ProtectionConfig(
            market_regime_detection=True,
            volatility_clustering_check=True,
            trend_consistency_check=True,
            temporal_validation_enabled=True,
            adaptive_thresholds=True
        )
        
        protection_system = AdvancedMLProtectionSystem(regime_config)
        
        # Analyze market regimes
        print("\n🔍 Analyzing market regimes...")
        report = protection_system.analyze_data_comprehensive(
            data=regime_data,
            target_column='target',
            feature_columns=[col for col in regime_data.columns if col != 'target']
        )
        
        # Display regime analysis results
        print(f"\n📈 Market Regime Analysis:")
        print(f"  • Overall Protection Score: {report.overall_protection_score:.3f}")
        print(f"  • Temporal Stability Issues: {len([r for r in report.recommendations if 'regime' in r.lower()])}")
        
        regime_recommendations = [r for r in report.recommendations if any(keyword in r.lower() for keyword in ['regime', 'volatility', 'trend'])]
        if regime_recommendations:
            print(f"  • Regime-Specific Recommendations:")
            for rec in regime_recommendations:
                print(f"    - {rec}")
        
        # Analyze volatility clustering
        volatility_analysis = self.analyze_volatility_clustering(regime_data)
        print(f"\n📊 Volatility Analysis:")
        print(f"  • High Volatility Periods: {volatility_analysis['high_vol_periods']}")
        print(f"  • Low Volatility Periods: {volatility_analysis['low_vol_periods']}")
        print(f"  • Volatility Clustering Score: {volatility_analysis['clustering_score']:.3f}")
        
        # Save results
        regime_data.to_csv('examples_output/example6_regime_data.csv', index=False)
        
        print("✅ Example 6 completed successfully")
        return regime_data, report, volatility_analysis
    
    def example_7_production_monitoring(self):
        """
        Example 7: Production Monitoring and Alerting
        
        Demonstrates:
        - Real-time protection monitoring
        - Performance degradation detection
        - Automated alerting system
        - Historical trend analysis
        """
        print("\n" + "="*80)
        print("EXAMPLE 7: PRODUCTION MONITORING")
        print("="*80)
        
        # Initialize ProjectP integration for monitoring
        integration = ProjectPProtectionIntegration()
        
        # Simulate production data streams
        print("\n📡 Simulating production data streams...")
        
        monitoring_results = []
        
        for day in range(5):  # Simulate 5 days of data
            # Generate daily data with varying quality
            daily_data = self.generate_daily_production_data(day)
            
            # Simulate model performance (degrading over time)
            model_performance = {
                'auc': max(0.5, 0.8 - day * 0.05),
                'accuracy': max(0.5, 0.75 - day * 0.04),
                'precision': max(0.5, 0.72 - day * 0.03),
                'recall': max(0.5, 0.70 - day * 0.02)
            }
            
            # Monitor pipeline
            monitoring_report = integration.monitor_projectp_pipeline(
                data=daily_data,
                model_performance=model_performance,
                target_column='target'
            )
            
            monitoring_results.append({
                'day': day + 1,
                'data_samples': len(daily_data),
                'alerts': len(monitoring_report['alerts']),
                'auc': model_performance['auc'],
                'missing_pct': monitoring_report['data_health']['missing_percentage']
            })
            
            print(f"  Day {day + 1}: AUC={model_performance['auc']:.3f}, Alerts={len(monitoring_report['alerts'])}")
        
        # Analyze monitoring trends
        print(f"\n📊 Monitoring Trends Analysis:")
        
        aucs = [r['auc'] for r in monitoring_results]
        alerts = [r['alerts'] for r in monitoring_results]
        
        print(f"  • AUC Trend: {aucs[0]:.3f} → {aucs[-1]:.3f} (Change: {aucs[-1] - aucs[0]:.3f})")
        print(f"  • Total Alerts: {sum(alerts)}")
        print(f"  • Alert Trend: {'Increasing' if alerts[-1] > alerts[0] else 'Stable/Decreasing'}")
        
        # Generate comprehensive monitoring report
        comprehensive_report = integration.generate_projectp_protection_report()
        
        print(f"\n📋 Production Health Summary:")
        print(f"  • Integration Status: {comprehensive_report['integration_status']}")
        print(f"  • Performance History: {len(comprehensive_report['performance_history'])} entries")
        print(f"  • Recommendations: {len(comprehensive_report['recommendations'])}")
        
        # Save monitoring results
        monitoring_df = pd.DataFrame(monitoring_results)
        monitoring_df.to_csv('examples_output/example7_monitoring_results.csv', index=False)
        
        print("✅ Example 7 completed successfully")
        return monitoring_results, comprehensive_report
    
    # Helper methods for generating test data
    
    def generate_sample_trading_data_with_issues(self):
        """Generate sample trading data with intentional issues for testing"""
        np.random.seed(42)
        n_samples = 500
        
        # Create time index
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1H')
        data = pd.DataFrame(index=dates)
        
        # Basic OHLCV features
        data['open'] = 100 + np.cumsum(np.random.normal(0, 0.5, n_samples))
        data['high'] = data['open'] + np.random.exponential(0.5, n_samples)
        data['low'] = data['open'] - np.random.exponential(0.5, n_samples)
        data['close'] = data['open'] + np.random.normal(0, 0.3, n_samples)
        data['volume'] = np.random.lognormal(8, 1, n_samples)
        
        # Technical indicators
        data['sma_20'] = data['close'].rolling(20).mean()
        data['rsi'] = 50 + np.random.normal(0, 20, n_samples)
        data['macd'] = np.random.normal(0, 0.5, n_samples)
        
        # Add intentional issues
        # Missing data
        data.loc[data.index[:50], 'sma_20'] = np.nan
        data.loc[data.index[100:120], 'volume'] = np.nan
        
        # Highly correlated feature (redundant)
        data['close_copy'] = data['close'] * 1.01 + np.random.normal(0, 0.01, n_samples)
        
        # Constant feature
        data['constant_feature'] = 1.0
        
        # Outliers
        outlier_indices = np.random.choice(n_samples, 10, replace=False)
        data.loc[data.index[outlier_indices], 'volume'] *= 10
        
        # Target variable
        data['returns'] = data['close'].pct_change()
        data['target'] = (data['returns'].shift(-1) > data['returns'].median()).astype(int)
        
        return data.dropna().reset_index(drop=True)
    
    def generate_realistic_projectp_data(self):
        """Generate realistic ProjectP-style trading data"""
        np.random.seed(123)
        n_samples = 800
        
        # 15-minute timeframe data
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='15T')
        data = pd.DataFrame(index=dates)
        
        # Market microstructure features
        data['bid'] = 100 + np.cumsum(np.random.normal(0, 0.02, n_samples))
        data['ask'] = data['bid'] + np.random.gamma(1, 0.001, n_samples)
        data['mid'] = (data['bid'] + data['ask']) / 2
        data['spread'] = data['ask'] - data['bid']
        data['volume'] = np.random.lognormal(9, 1.5, n_samples)
        
        # Technical analysis features
        data['sma_10'] = data['mid'].rolling(10).mean()
        data['sma_50'] = data['mid'].rolling(50).mean()
        data['ema_20'] = data['mid'].ewm(span=20).mean()
        data['rsi'] = 50 + np.random.normal(0, 15, n_samples)
        data['bb_upper'] = data['mid'] + 2 * data['mid'].rolling(20).std()
        data['bb_lower'] = data['mid'] - 2 * data['mid'].rolling(20).std()
        
        # Order book features
        data['order_imbalance'] = np.random.normal(0, 0.1, n_samples)
        data['price_impact'] = np.random.exponential(0.01, n_samples)
        
        # Market timing features
        data['hour'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek
        data['is_market_hours'] = ((data['hour'] >= 9) & (data['hour'] <= 16)).astype(int)
        
        # Returns and volatility
        data['returns'] = data['mid'].pct_change()
        data['volatility'] = data['returns'].rolling(20).std()
        
        # Target variable (trading signal)
        future_returns = data['returns'].shift(-1)
        data['target'] = (future_returns > future_returns.quantile(0.6)).astype(int)
        
        return data.dropna().reset_index(drop=True)
    
    def generate_noisy_trading_data(self):
        """Generate trading data with various types of noise"""
        np.random.seed(456)
        n_samples = 600
        
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='5T')
        data = pd.DataFrame(index=dates)
        
        # Base signal
        trend = np.linspace(100, 105, n_samples)
        seasonal = 2 * np.sin(2 * np.pi * np.arange(n_samples) / 100)
        base_signal = trend + seasonal
        
        # Add different types of noise
        # White noise
        white_noise = np.random.normal(0, 0.5, n_samples)
        
        # Pink noise (1/f noise)
        pink_noise = np.cumsum(np.random.normal(0, 0.1, n_samples))
        pink_noise = (pink_noise - pink_noise.mean()) / pink_noise.std() * 0.3
        
        # Impulse noise (spikes)
        impulse_noise = np.zeros(n_samples)
        spike_indices = np.random.choice(n_samples, 20, replace=False)
        impulse_noise[spike_indices] = np.random.normal(0, 2, 20)
        
        # Create noisy features
        data['clean_price'] = base_signal
        data['noisy_price'] = base_signal + white_noise + pink_noise + impulse_noise
        data['very_noisy'] = base_signal + np.random.normal(0, 1.5, n_samples)
        data['low_snr_feature'] = np.random.normal(0, 2, n_samples) + 0.1 * base_signal
        
        # Technical indicators on noisy data
        data['noisy_sma'] = data['noisy_price'].rolling(10).mean()
        data['noisy_rsi'] = 50 + np.random.normal(0, 25, n_samples)  # Very noisy RSI
        
        # Some clean features for comparison
        data['clean_volume'] = np.random.lognormal(8, 0.5, n_samples)
        data['clean_indicator'] = np.sin(2 * np.pi * np.arange(n_samples) / 50)
        
        # Target based on clean signal
        returns = np.diff(base_signal)
        returns = np.append(returns, returns[-1])  # Pad to same length
        data['target'] = (returns > np.median(returns)).astype(int)
        
        return data.reset_index(drop=True)
    
    def generate_data_with_leakage(self):
        """Generate data with various types of data leakage"""
        np.random.seed(789)
        n_samples = 400
        
        data = pd.DataFrame()
        
        # Normal features
        data['feature1'] = np.random.normal(0, 1, n_samples)
        data['feature2'] = np.random.normal(0, 1, n_samples)
        data['feature3'] = np.random.normal(0, 1, n_samples)
        
        # Create target first
        data['target'] = (data['feature1'] + data['feature2'] + np.random.normal(0, 0.5, n_samples) > 0).astype(int)
        
        # Add leaky features
        # Direct target leakage (feature is almost identical to target)
        data['leaky_target'] = data['target'] + np.random.normal(0, 0.01, n_samples)
        
        # Future information leakage
        data['future_feature'] = data['target'].shift(-1).fillna(0)  # Uses future target
        
        # Perfect correlation (redundant feature)
        data['perfect_corr'] = data['feature1'] * 1.0000001  # Almost perfect correlation
        
        # Conditional leakage (high correlation with target in certain conditions)
        data['conditional_leak'] = np.where(
            data['target'] == 1,
            np.random.normal(2, 0.1, n_samples),  # High values when target=1
            np.random.normal(-2, 0.1, n_samples)  # Low values when target=0
        )
        
        # Temporal leakage (using information not available at prediction time)
        data['temporal_leak'] = data['feature1'].rolling(window=5, center=True).mean()
        
        return data.dropna().reset_index(drop=True)
    
    def generate_overfitting_prone_data(self):
        """Generate data that's prone to overfitting"""
        np.random.seed(101)
        n_samples = 200  # Small dataset
        n_features = 150  # Many features (high dimensional)
        
        data = pd.DataFrame()
        
        # Generate many random features
        for i in range(n_features):
            data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
        
        # Create target with minimal signal
        # Only first 5 features have weak relationship with target
        signal = (
            0.3 * data['feature_0'] +
            0.2 * data['feature_1'] +
            0.1 * data['feature_2'] +
            0.05 * data['feature_3'] +
            0.05 * data['feature_4']
        )
        
        # Add noise to make signal weak
        noisy_signal = signal + np.random.normal(0, 2, n_samples)
        data['target'] = (noisy_signal > noisy_signal.median()).astype(int)
        
        return data
    
    def generate_multi_regime_data(self):
        """Generate data with multiple market regimes"""
        np.random.seed(202)
        n_samples = 1000
        
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1H')
        data = pd.DataFrame(index=dates)
        
        # Define regime periods
        regime_1_end = n_samples // 3
        regime_2_end = 2 * n_samples // 3
        
        # Initialize arrays
        returns = np.zeros(n_samples)
        volatility = np.zeros(n_samples)
        
        # Regime 1: Low volatility, trending
        returns[:regime_1_end] = np.random.normal(0.001, 0.01, regime_1_end)
        volatility[:regime_1_end] = np.random.gamma(1, 0.01, regime_1_end)
        
        # Regime 2: High volatility, mean-reverting
        returns[regime_1_end:regime_2_end] = np.random.normal(0, 0.05, regime_2_end - regime_1_end)
        volatility[regime_1_end:regime_2_end] = np.random.gamma(3, 0.02, regime_2_end - regime_1_end)
        
        # Regime 3: Medium volatility, sideways
        returns[regime_2_end:] = np.random.normal(0, 0.02, n_samples - regime_2_end)
        volatility[regime_2_end:] = np.random.gamma(2, 0.015, n_samples - regime_2_end)
        
        # Build price from returns
        data['returns'] = returns
        data['price'] = 100 * np.exp(np.cumsum(returns))
        data['volatility'] = volatility
        
        # Add regime indicators
        data['regime'] = 1
        data.loc[regime_1_end:regime_2_end-1, 'regime'] = 2
        data.loc[regime_2_end:, 'regime'] = 3
        
        # Technical indicators that work differently in each regime
        data['sma_20'] = data['price'].rolling(20).mean()
        data['price_vs_sma'] = data['price'] / data['sma_20'] - 1
        data['vol_rank'] = data['volatility'].rolling(50).rank() / 50
        
        # Target depends on regime
        data['target'] = 0
        for regime in [1, 2, 3]:
            regime_mask = data['regime'] == regime
            if regime == 1:  # Trending regime - momentum works
                data.loc[regime_mask, 'target'] = (data.loc[regime_mask, 'returns'] > 0).astype(int)
            elif regime == 2:  # High vol regime - mean reversion works
                data.loc[regime_mask, 'target'] = (data.loc[regime_mask, 'price_vs_sma'] < -0.01).astype(int)
            else:  # Sideways regime - volatility breakout works
                data.loc[regime_mask, 'target'] = (data.loc[regime_mask, 'vol_rank'] > 0.8).astype(int)
        
        return data.dropna().reset_index(drop=True)
    
    def generate_daily_production_data(self, day):
        """Generate daily production data with varying quality"""
        np.random.seed(day * 100)
        n_samples = 96  # 15-minute data for one day
        
        data = pd.DataFrame()
        
        # Data quality degrades over time
        missing_prob = min(0.1, day * 0.02)  # Increasing missing data
        noise_level = 1 + day * 0.2  # Increasing noise
        
        # Generate features
        for i in range(10):
            feature = np.random.normal(0, noise_level, n_samples)
            
            # Add missing values
            missing_mask = np.random.random(n_samples) < missing_prob
            feature[missing_mask] = np.nan
            
            data[f'feature_{i}'] = feature
        
        # Target variable
        signal = data['feature_0'].fillna(0) + data['feature_1'].fillna(0)
        data['target'] = (signal > signal.median()).astype(int)
        
        return data
    
    # Helper methods for analysis and visualization
    
    def display_protection_results(self, report: ProtectionReport, title: str):
        """Display protection analysis results in a formatted way"""
        print(f"\n📊 {title}")
        print("-" * 60)
        print(f"Overall Protection Score: {report.overall_protection_score:.3f}")
        print(f"Risk Level: {report.risk_level.upper()}")
        print(f"Data Quality Score: {report.data_quality_score:.3f}")
        print(f"Overfitting Risk: {report.overfitting_risk.upper()}")
        print(f"Noise Level: {report.noise_level.upper()}")
        print(f"Signal-to-Noise Ratio: {report.signal_to_noise_ratio:.2f}")
        
        # Issues summary
        total_issues = (
            len(report.missing_data_issues) +
            len(report.correlation_issues) +
            len(report.feature_leakage_issues)
        )
        
        print(f"\nIssues Detected: {total_issues}")
        if report.missing_data_issues:
            print(f"  • Missing Data: {len(report.missing_data_issues)} issues")
        if report.correlation_issues:
            print(f"  • Correlation: {len(report.correlation_issues)} issues")
        if report.feature_leakage_issues:
            print(f"  • Feature Leakage: {len(report.feature_leakage_issues)} issues")
        if report.target_leakage_detected:
            print(f"  • ⚠️  Target Leakage Detected!")
        
        # Top recommendations
        if report.recommendations:
            print(f"\nTop Recommendations:")
            for i, rec in enumerate(report.recommendations[:3], 1):
                print(f"  {i}. {rec}")
        
        print(f"\nProcessing Time: {report.processing_time_seconds:.2f}s")
    
    def plot_noise_comparison(self, noisy_data, denoised_data):
        """Plot comparison between noisy and denoised data"""
        try:
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Original vs Denoised Prices
            plt.subplot(2, 3, 1)
            plt.plot(noisy_data['noisy_price'][:100], label='Noisy', alpha=0.7)
            plt.plot(denoised_data['noisy_price'][:100], label='Denoised', alpha=0.8)
            plt.title('Price Comparison')
            plt.legend()
            
            # Plot 2: Signal-to-Noise Comparison
            plt.subplot(2, 3, 2)
            features = ['noisy_price', 'very_noisy', 'low_snr_feature']
            snr_before = []
            snr_after = []
            
            for feature in features:
                if feature in noisy_data.columns and feature in denoised_data.columns:
                    # Calculate SNR (simplified)
                    snr_before.append(np.abs(noisy_data[feature].mean()) / noisy_data[feature].std())
                    snr_after.append(np.abs(denoised_data[feature].mean()) / denoised_data[feature].std())
            
            x = np.arange(len(features))
            plt.bar(x - 0.2, snr_before, 0.4, label='Before', alpha=0.7)
            plt.bar(x + 0.2, snr_after, 0.4, label='After', alpha=0.7)
            plt.xticks(x, features, rotation=45)
            plt.title('Signal-to-Noise Ratio')
            plt.legend()
            
            # Plot 3: Volatility Analysis
            plt.subplot(2, 3, 3)
            if 'volatility' in noisy_data.columns:
                rolling_vol_before = noisy_data['noisy_price'].rolling(20).std()
                rolling_vol_after = denoised_data['noisy_price'].rolling(20).std()
                plt.plot(rolling_vol_before[:200], label='Before', alpha=0.7)
                plt.plot(rolling_vol_after[:200], label='After', alpha=0.7)
                plt.title('Rolling Volatility')
                plt.legend()
            
            # Plot 4: Distribution Comparison
            plt.subplot(2, 3, 4)
            plt.hist(noisy_data['noisy_price'].dropna(), bins=50, alpha=0.7, label='Noisy', density=True)
            plt.hist(denoised_data['noisy_price'].dropna(), bins=50, alpha=0.7, label='Denoised', density=True)
            plt.title('Price Distribution')
            plt.legend()
            
            # Plot 5: Correlation Matrix Before
            plt.subplot(2, 3, 5)
            numeric_cols = noisy_data.select_dtypes(include=[np.number]).columns[:6]  # First 6 numeric columns
            corr_before = noisy_data[numeric_cols].corr()
            sns.heatmap(corr_before, annot=True, cmap='coolwarm', center=0, square=True)
            plt.title('Correlation Before')
            
            # Plot 6: Correlation Matrix After
            plt.subplot(2, 3, 6)
            numeric_cols_after = denoised_data.select_dtypes(include=[np.number]).columns[:6]
            corr_after = denoised_data[numeric_cols_after].corr()
            sns.heatmap(corr_after, annot=True, cmap='coolwarm', center=0, square=True)
            plt.title('Correlation After')
            
            plt.tight_layout()
            plt.savefig('examples_output/noise_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("📊 Noise comparison plot saved to examples_output/noise_comparison.png")
            
        except Exception as e:
            print(f"⚠️  Could not create plots: {e}")
    
    def remove_leaky_features(self, data, report):
        """Remove features identified as having data leakage"""
        clean_data = data.copy()
        
        # Features to remove based on report
        features_to_remove = []
        
        # Remove features mentioned in leakage issues
        for issue in report.feature_leakage_issues:
            if "'" in issue:
                # Extract feature name from issue description
                parts = issue.split("'")
                if len(parts) >= 2:
                    feature_name = parts[1]
                    if feature_name in clean_data.columns and feature_name != 'target':
                        features_to_remove.append(feature_name)
        
        # Remove features with very high correlation to target
        if 'target' in clean_data.columns:
            feature_cols = [col for col in clean_data.columns if col != 'target']
            for col in feature_cols:
                if clean_data[col].dtype in ['int64', 'float64']:
                    corr = abs(clean_data[col].corr(clean_data['target']))
                    if corr > 0.95:  # Very high correlation
                        features_to_remove.append(col)
        
        # Remove identified features
        features_to_remove = list(set(features_to_remove))  # Remove duplicates
        if features_to_remove:
            clean_data = clean_data.drop(columns=features_to_remove)
            print(f"🧹 Removed leaky features: {features_to_remove}")
        
        return clean_data
    
    def apply_overfitting_prevention(self, data, report):
        """Apply overfitting prevention techniques"""
        protected_data = data.copy()
        
        # Feature selection to reduce overfitting
        if 'target' in protected_data.columns:
            feature_cols = [col for col in protected_data.columns if col != 'target']
            
            # Keep only the most important features (simple selection)
            max_features = max(10, int(len(feature_cols) * 0.3))  # Keep at most 30% of features
            
            if len(feature_cols) > max_features:
                # Simple feature selection based on correlation with target
                correlations = {}
                for col in feature_cols:
                    if protected_data[col].dtype in ['int64', 'float64']:
                        correlations[col] = abs(protected_data[col].corr(protected_data['target']))
                
                # Select top features
                top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:max_features]
                selected_features = [feat[0] for feat in top_features]
                
                # Keep only selected features plus target
                protected_data = protected_data[selected_features + ['target']]
                
                print(f"🎯 Selected {len(selected_features)} most important features")
        
        return protected_data
    
    def analyze_volatility_clustering(self, data):
        """Analyze volatility clustering in the data"""
        if 'returns' not in data.columns:
            return {'error': 'No returns column found'}
        
        returns = data['returns'].dropna()
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=20).std()
        
        # Identify high and low volatility periods
        high_threshold = rolling_vol.quantile(0.75)
        low_threshold = rolling_vol.quantile(0.25)
        
        high_vol_periods = (rolling_vol > high_threshold).sum()
        low_vol_periods = (rolling_vol < low_threshold).sum()
        
        # Simple clustering score based on persistence
        vol_changes = np.diff(rolling_vol.dropna())
        clustering_score = 1 - (np.sum(np.abs(vol_changes)) / len(vol_changes)) / rolling_vol.std()
        
        return {
            'high_vol_periods': high_vol_periods,
            'low_vol_periods': low_vol_periods,
            'clustering_score': max(0, clustering_score)
        }
    
    def run_all_examples(self):
        """Run all protection examples"""
        print("🚀 Running All Advanced ML Protection Examples")
        print("=" * 80)
        
        try:
            # Run all examples
            example1_results = self.example_1_basic_data_protection()
            example2_results = self.example_2_projectp_integration()
            example3_results = self.example_3_advanced_noise_detection()
            example4_results = self.example_4_data_leakage_detection()
            example5_results = self.example_5_overfitting_prevention()
            example6_results = self.example_6_market_regime_analysis()
            example7_results = self.example_7_production_monitoring()
            
            # Summary
            print("\n" + "=" * 80)
            print("🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY")
            print("=" * 80)
            
            print("\n📁 Output Files Generated:")
            output_files = list(Path('examples_output').glob('*'))
            for file_path in sorted(output_files):
                print(f"  • {file_path}")
            
            print("\n📊 Example Summary:")
            print("  1. ✅ Basic Data Protection - Data quality and automated fixes")
            print("  2. ✅ ProjectP Integration - Trading pipeline protection")
            print("  3. ✅ Noise Detection - Signal enhancement and denoising")
            print("  4. ✅ Data Leakage Detection - Leakage prevention")
            print("  5. ✅ Overfitting Prevention - Model validation")
            print("  6. ✅ Market Regime Analysis - Regime-aware protection")
            print("  7. ✅ Production Monitoring - Real-time monitoring")
            
            return {
                'example1': example1_results,
                'example2': example2_results,
                'example3': example3_results,
                'example4': example4_results,
                'example5': example5_results,
                'example6': example6_results,
                'example7': example7_results
            }
            
        except Exception as e:
            print(f"❌ Error running examples: {e}")
            logger.error(f"Examples error: {e}", exc_info=True)
            return None

def main():
    """Main function to run all examples"""
    examples = ProtectionExamples()
    results = examples.run_all_examples()
    
    if results:
        print("\n🎯 All examples completed successfully!")
        print("📖 Check the examples_output directory for detailed results")
        print("📊 Review the generated reports and visualizations")
    else:
        print("\n❌ Some examples failed. Check the logs for details.")

if __name__ == "__main__":
    main()
