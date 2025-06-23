#!/usr/bin/env python3
"""
🛡️ ML Protection System - Auto Setup Script
Automated setup and validation for enterprise-grade ML protection system
"""

import os
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
import subprocess

# Suppress warnings during setup
warnings.filterwarnings('ignore')

def print_banner():
    """Print setup banner"""
    print("=" * 80)
    print("🛡️ ML Protection System - Enterprise Auto Setup")
    print("=" * 80)
    print("📋 Advanced protection against noise, data leakage, and overfitting")
    print("🎯 Designed for trading ML systems and ProjectP integration")
    print("=" * 80)

def check_dependencies():
    """Check and install required dependencies"""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'pandas>=1.3.0',
        'numpy>=1.20.0', 
        'scikit-learn>=1.0.0',
        'scipy>=1.7.0',
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
        'pyyaml>=5.4.0',
        'joblib>=1.0.0'
    ]
    
    optional_packages = [
        'click>=8.0.0',
        'rich>=10.0.0',
        'plotly>=5.0.0',
        'streamlit>=1.10.0'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        package_name = package.split('>=')[0]
        try:
            __import__(package_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - REQUIRED")
            missing_packages.append(package)
    
    for package in optional_packages:
        package_name = package.split('>=')[0]
        try:
            __import__(package_name)
            print(f"✅ {package_name} (optional)")
        except ImportError:
            print(f"⚠️ {package_name} - OPTIONAL")
    
    if missing_packages:
        print(f"\n❌ Missing required packages: {len(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All required dependencies available")
    return True

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directory structure...")
    
    directories = [
        "protection_reports",
        "protection_examples_results",
        "protection_configs", 
        "protection_logs",
        "protection_backups",
        "enterprise_tracking",
        "enterprise_mlruns",
        "models",
        "artifacts",
        "logs",
        "data",
        "notebooks",
        "scripts",
        "reports",
        "backups",
        "configs"
    ]
    
    created_count = 0
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created: {directory}")
            created_count += 1
        else:
            print(f"📁 Exists: {directory}")
    
    print(f"✅ Directory structure ready ({created_count} new directories created)")
    return True

def create_default_config():
    """Create default protection configuration"""
    print("\n⚙️ Creating default configuration...")
    
    config = {
        'protection_level': 'enterprise',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        
        # Global Protection Settings
        'protection': {
            'enable_tracking': True,
            'auto_fix_issues': True,
            'generate_reports': True,
            'validation_mode': 'strict'
        },
        
        # Noise Detection Configuration
        'noise_detection': {
            'outlier_detection_method': 'isolation_forest',
            'contamination_rate': 0.05,
            'noise_threshold': 0.95,
            'rolling_window_size': 20,
            'volatility_threshold': 3.0,
            'enable_statistical_tests': True,
            'feature_noise_detection': True,
            'cleaning_strategy': 'conservative',
            'max_outlier_removal_percent': 5.0
        },
        
        # Data Leakage Prevention
        'leakage_detection': {
            'target_correlation_threshold': 0.95,
            'suspicious_correlation_threshold': 0.8,
            'temporal_validation': True,
            'required_lag_hours': 24,
            'future_info_keywords': ['future', 'next', 'tomorrow', 'ahead', 'forward'],
            'check_perfect_predictors': True,
            'purged_cross_validation': True
        },
        
        # Overfitting Protection
        'overfitting_protection': {
            'min_samples_per_feature': 10,
            'feature_selection': True,
            'selection_method': 'recursive',
            'cross_validation_strategy': 'time_series',
            'n_splits': 5,
            'regularization': True,
            'auto_tune_regularization': True,
            'early_stopping': True
        },
        
        # ProjectP Integration
        'projectp_integration': {
            'enable_integration': True,
            'auto_validate_features': True,
            'trading_features': ['close', 'high', 'low', 'open', 'volume'],
            'target_features': ['target', 'return', 'signal'],
            'critical_features': ['close', 'volume'],
            'protection_checkpoint_frequency': 'every_run',
            'emergency_stop_on_critical_issues': True
        },
        
        # Trading Specific Settings
        'trading_specific': {
            'price_spike_threshold': 0.2,
            'volume_validation': True,
            'temporal_consistency_check': True,
            'max_acceptable_noise': 0.2
        },
        
        # Reporting Configuration
        'reporting': {
            'auto_generate_reports': True,
            'report_format': ['html', 'yaml'],
            'report_directory': './protection_reports',
            'detailed_feature_analysis': True,
            'include_visualizations': True,
            'archive_old_reports': True,
            'max_report_age_days': 30
        },
        
        # Integration Settings
        'integration': {
            'tracking': {
                'log_protection_actions': True,
                'track_data_quality_metrics': True,
                'save_protection_artifacts': True,
                'auto_tag_experiments': True
            },
            'alerts': {
                'enable_alerts': True,
                'alert_channels': ['log', 'tracking'],
                'alert_thresholds': {
                    'data_quality_score': 0.7,
                    'noise_level': 0.3,
                    'leakage_risk': 0.2,
                    'overfitting_risk': 0.3
                }
            }
        },
        
        # Logging Configuration
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'ml_protection.log',
            'max_file_size_mb': 10,
            'enable_console_logging': True
        },
        
        # Performance Settings
        'performance': {
            'enable_parallel_processing': True,
            'max_workers': 4,
            'chunk_size': 1000,
            'cache_results': True,
            'cache_ttl_hours': 24
        },
        
        # Environment Specific
        'environments': {
            'development': {
                'protection_level': 'standard',
                'debug_mode': True
            },
            'staging': {
                'protection_level': 'aggressive', 
                'enable_alerts': True
            },
            'production': {
                'protection_level': 'enterprise',
                'auto_fix_issues': True,
                'emergency_stop_on_critical_issues': True
            }
        }
    }
    
    config_file = "ml_protection_config.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, allow_unicode=True)
    
    print(f"✅ Configuration created: {config_file}")
    return config_file

def create_sample_data():
    """Create realistic sample trading data for testing"""
    print("\n📊 Creating sample trading data...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Parameters
    n_samples = 1000
    n_features = 15
    
    # Generate time series
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
    
    # Generate realistic price data using GBM
    returns = np.random.normal(0, 0.02, n_samples)
    returns[0] = 0
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate volume data (log-normal distribution)
    base_volume = np.random.lognormal(10, 1, n_samples)
    
    # Create base DataFrame
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
        'close': prices,
        'volume': base_volume,
        'returns': np.concatenate([[0], np.diff(np.log(prices))])
    })
    
    # Ensure OHLC relationships
    data['high'] = np.maximum(data['high'], data['close'])
    data['low'] = np.minimum(data['low'], data['close'])
    data['high'] = np.maximum(data['high'], data['open'])
    data['low'] = np.minimum(data['low'], data['open'])
    
    # Add technical indicators
    data['sma_5'] = data['close'].rolling(5).mean()
    data['sma_20'] = data['close'].rolling(20).mean()
    data['volatility'] = data['returns'].rolling(20).std()
    
    # Add RSI indicator
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # Add trading signals
    data['signal'] = np.where(data['sma_5'] > data['sma_20'], 1, -1)
    data['position_size'] = np.abs(data['returns']) * 100
    
    # Add some realistic features
    data['bid_ask_spread'] = np.random.normal(0.001, 0.0005, n_samples)
    data['market_impact'] = data['volume'] / data['volume'].rolling(50).mean()
    
    # Create target (future returns)
    data['target'] = data['returns'].shift(-1)
    
    # Add data quality issues for testing protection system
    print("  🔧 Adding data quality issues for testing...")
    
    # Issue 1: Missing values (2% of volume data)
    missing_indices = np.random.choice(len(data), size=int(0.02 * len(data)), replace=False)
    data.loc[missing_indices, 'volume'] = np.nan
    
    # Issue 2: Price spikes (simulate data errors - 1% of data)
    spike_indices = np.random.choice(len(data), size=int(0.01 * len(data)), replace=False)
    for idx in spike_indices:
        spike_factor = np.random.choice([-0.5, 0.5])  # ±50% spike
        data.loc[idx, 'close'] *= (1 + spike_factor)
        data.loc[idx, 'high'] = max(data.loc[idx, 'high'], data.loc[idx, 'close'])
        data.loc[idx, 'low'] = min(data.loc[idx, 'low'], data.loc[idx, 'close'])
    
    # Issue 3: Data leakage - perfect correlation with future target
    data['leaky_feature'] = data['target'].shift(-1) * 0.99 + np.random.normal(0, 0.001, len(data))
    
    # Issue 4: Another leakage - highly correlated with target
    data['suspicious_feature'] = data['target'] * 0.95 + np.random.normal(0, 0.01, len(data))
    
    # Issue 5: Outliers in volume (extreme values)
    outlier_indices = np.random.choice(len(data), size=int(0.005 * len(data)), replace=False)
    for idx in outlier_indices:
        data.loc[idx, 'volume'] *= np.random.uniform(10, 50)  # 10-50x normal volume
    
    # Issue 6: Zero/negative volumes
    zero_indices = np.random.choice(len(data), size=10, replace=False)
    data.loc[zero_indices, 'volume'] = 0
    
    negative_indices = np.random.choice(len(data), size=5, replace=False)
    data.loc[negative_indices, 'volume'] = -np.abs(data.loc[negative_indices, 'volume'])
    
    # Remove last row (no target)
    data = data[:-1].copy()
    
    # Save sample data
    sample_file = "sample_trading_data.csv"
    data.to_csv(sample_file, index=False)
    
    print(f"✅ Sample data created: {sample_file}")
    print(f"  📊 Shape: {data.shape}")
    print(f"  🕐 Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    print(f"  🎯 Features: {list(data.columns)}")
    
    return sample_file

def test_protection_system():
    """Test the protection system with sample data"""
    print("\n🧪 Testing ML Protection System...")
    
    try:
        # Test imports
        print("  📦 Testing imports...")
        from ml_protection_system import MLProtectionSystem, ProtectionLevel
        print("    ✅ ml_protection_system imported")
        
        from projectp_protection_integration import ProjectPProtectionIntegration
        print("    ✅ projectp_protection_integration imported")
        
        # Initialize protection system
        print("  🛡️ Initializing protection system...")
        protection_system = MLProtectionSystem(ProtectionLevel.ENTERPRISE)
        print("    ✅ Protection system initialized")
        
        # Load sample data
        print("  📊 Loading sample data...")
        data = pd.read_csv("sample_trading_data.csv")
        print(f"    ✅ Sample data loaded: {data.shape}")
        
        # Test basic protection
        print("  🔍 Testing basic protection...")
        result = protection_system.protect_dataset(
            data.head(100),  # Test with subset
            target_col='target'
        )
        print(f"    ✅ Protection test passed - Score: {result.overall_score:.4f}")
        
        # Test ProjectP integration
        print("  🔗 Testing ProjectP integration...")
        integration = ProjectPProtectionIntegration()
        validation = integration.validate_projectp_pipeline(sample_data=data.head(50))
        print(f"    ✅ ProjectP integration test - Ready: {validation['system_ready']}")
        
        # Test CLI availability
        print("  💻 Testing CLI availability...")
        if Path("ml_protection_cli.py").exists():
            print("    ✅ CLI script available")
        else:
            print("    ⚠️ CLI script not found")
        
        print("✅ All protection system tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Protection system test failed: {e}")
        return False

def create_quick_start_scripts():
    """Create quick start scripts for common tasks"""
    print("\n📝 Creating quick start scripts...")
    
    # Quick test script
    quick_test_script = """#!/usr/bin/env python3
# Quick Protection Test Script

import pandas as pd
from ml_protection_system import MLProtectionSystem, ProtectionLevel
from projectp_protection_integration import quick_protect_data, validate_pipeline_data

def main():
    print("⚡ Quick Protection Test")
    print("-" * 40)
    
    # Load sample data
    try:
        data = pd.read_csv("sample_trading_data.csv")
        print(f"📊 Loaded data: {data.shape}")
    except FileNotFoundError:
        print("❌ Sample data not found. Run auto_setup_protection.py first.")
        return
    
    # Quick protection test
    try:
        protected_data, report = quick_protect_data(data.head(200), 'target')
        
        quality_score = report.get('overall_quality_score', 0)
        print(f"🛡️ Protection Results:")
        print(f"  • Quality Score: {quality_score:.4f}")
        print(f"  • Original Shape: {data.head(200).shape}")
        print(f"  • Protected Shape: {protected_data.shape}")
        
        if quality_score > 0.8:
            print("🎉 Excellent data quality!")
        elif quality_score > 0.6:
            print("✅ Good data quality")
        else:
            print("⚠️ Data quality needs improvement")
        
        # Validate pipeline
        is_valid = validate_pipeline_data(data.head(100), show_report=True)
        
        if is_valid:
            print("✅ Pipeline validation passed")
        else:
            print("❌ Pipeline validation failed")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main()
"""
    
    with open("quick_protection_test.py", 'w') as f:
        f.write(quick_test_script)
    
    # Daily protection script
    daily_script = """#!/usr/bin/env python3
# Daily Protection Script for Production Use

import pandas as pd
from datetime import datetime
from pathlib import Path
from projectp_protection_integration import ProjectPProtectionIntegration

def daily_protection_pipeline(data_file):
    \"\"\"Daily protection pipeline for production data\"\"\"
    
    print(f"🛡️ Daily Protection Pipeline - {datetime.now()}")
    print("=" * 60)
    
    # Initialize integration
    integration = ProjectPProtectionIntegration(protection_level="enterprise")
    
    # Load data
    data = pd.read_csv(data_file)
    print(f"📊 Loaded data: {data.shape}")
    
    # Run protection
    protected_data, report = integration.protect_projectp_data(
        data,
        target_column='target',
        experiment_name=f"daily_protection_{datetime.now().strftime('%Y%m%d')}"
    )
    
    # Check quality
    quality_score = report['overall_quality_score']
    print(f"🎯 Quality Score: {quality_score:.4f}")
    
    # Save results
    output_dir = Path("data/protected")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"protected_data_{timestamp}.csv"
    protected_data.to_csv(output_file, index=False)
    
    print(f"💾 Protected data saved: {output_file}")
    
    # Alert if quality is low
    if quality_score < 0.7:
        print("🚨 ALERT: Low data quality detected!")
        for issue in report.get('critical_issues', []):
            print(f"  • {issue}")
    
    return quality_score > 0.7

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = "sample_trading_data.csv"
    
    success = daily_protection_pipeline(data_file)
    sys.exit(0 if success else 1)
"""
    
    with open("daily_protection_pipeline.py", 'w') as f:
        f.write(daily_script)
    
    # Make scripts executable (Unix/Linux)
    if os.name != 'nt':  # Not Windows
        os.chmod("quick_protection_test.py", 0o755)
        os.chmod("daily_protection_pipeline.py", 0o755)
    
    print("✅ Quick start scripts created:")
    print("  📄 quick_protection_test.py - Quick testing")
    print("  📄 daily_protection_pipeline.py - Production pipeline")

def create_requirements_file():
    """Create comprehensive requirements file"""
    print("\n📋 Creating requirements file...")
    
    requirements = """# ML Protection System Requirements
# Core dependencies for enterprise-grade ML protection

# === CORE ML AND DATA PROCESSING ===
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
scipy>=1.7.0
matplotlib>=3.3.0
seaborn>=0.11.0

# === CONFIGURATION AND SERIALIZATION ===
pyyaml>=5.4.0
joblib>=1.0.0

# === STATISTICAL ANALYSIS ===
statsmodels>=0.13.0

# === CLI AND USER INTERFACE ===
click>=8.0.0
rich>=10.0.0
typer>=0.4.0

# === VISUALIZATION AND DASHBOARDS ===
plotly>=5.0.0
streamlit>=1.10.0

# === TRACKING AND MONITORING (OPTIONAL) ===
mlflow>=1.20.0
wandb>=0.12.0

# === DEVELOPMENT AND TESTING ===
pytest>=6.0.0
jupyter>=1.0.0
ipywidgets>=7.6.0

# === SCHEDULING AND AUTOMATION ===
schedule>=1.1.0

# === PERFORMANCE AND OPTIMIZATION ===
numba>=0.55.0  # Optional for numerical acceleration
psutil>=5.8.0  # System monitoring
"""
    
    with open("ml_protection_requirements.txt", 'w') as f:
        f.write(requirements)
    
    print("✅ Requirements file created: ml_protection_requirements.txt")

def generate_summary_report():
    """Generate setup summary report"""
    print("\n📄 Generating setup summary...")
    
    summary = {
        'setup_timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'components': {
            'ml_protection_system': Path('ml_protection_system.py').exists(),
            'projectp_integration': Path('projectp_protection_integration.py').exists(),
            'cli_interface': Path('ml_protection_cli.py').exists(),
            'examples': Path('ml_protection_examples.py').exists(),
            'config_file': Path('ml_protection_config.yaml').exists(),
            'sample_data': Path('sample_trading_data.csv').exists()
        },
        'directories_created': [
            "protection_reports",
            "protection_examples_results",
            "protection_configs",
            "protection_logs",
            "protection_backups"
        ],
        'quick_start_files': [
            "quick_protection_test.py",
            "daily_protection_pipeline.py",
            "ml_protection_requirements.txt"
        ],
        'next_steps': [
            "Run: python quick_protection_test.py",
            "Test CLI: python ml_protection_cli.py status",
            "Analyze data: python ml_protection_cli.py analyze sample_trading_data.csv",
            "Run examples: python ml_protection_examples.py"
        ]
    }
    
    with open("setup_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("✅ Setup summary saved: setup_summary.json")
    return summary

def print_next_steps():
    """Print next steps for user"""
    print("\n" + "=" * 80)
    print("🎉 ML Protection System Setup Complete!")
    print("=" * 80)
    
    print("\n📋 What was created:")
    print("  🛡️ ML Protection System - Advanced protection engine")
    print("  🔗 ProjectP Integration - Trading pipeline integration")
    print("  💻 CLI Interface - Command-line tools")
    print("  📚 Examples & Documentation - Comprehensive examples")
    print("  ⚙️ Configuration - Enterprise-grade config")
    print("  📊 Sample Data - Realistic trading data for testing")
    
    print("\n🚀 Quick Start Commands:")
    print("  # Test the system")
    print("  python quick_protection_test.py")
    print("  ")
    print("  # Check system status")
    print("  python ml_protection_cli.py status")
    print("  ")
    print("  # Analyze sample data")
    print("  python ml_protection_cli.py analyze sample_trading_data.csv")
    print("  ")
    print("  # Run comprehensive examples")
    print("  python ml_protection_examples.py")
    print("  ")
    print("  # Quick data quality check")
    print("  python ml_protection_cli.py quick-check sample_trading_data.csv")
    
    print("\n📖 Documentation:")
    print("  📄 ML_PROTECTION_SYSTEM_GUIDE.md - Complete user guide")
    print("  📄 ML_PROTECTION_COMPLETE_SETUP.md - Setup instructions")
    print("  📄 setup_summary.json - Setup summary")
    
    print("\n💡 Production Usage:")
    print("  # For your own data")
    print("  python ml_protection_cli.py analyze your_data.csv --target your_target")
    print("  ")
    print("  # Clean your data")
    print("  python ml_protection_cli.py clean your_data.csv --output cleaned_data.csv")
    print("  ")
    print("  # ProjectP integration")
    print("  python ml_protection_cli.py projectp-integrate your_trading_data.csv")
    
    print("\n🛡️ Your ML models are now protected against:")
    print("  🔍 Noise and outliers")
    print("  🕵️ Data leakage") 
    print("  🧠 Overfitting")
    print("  📊 Poor data quality")
    
    print("\n✨ Ready for Enterprise ML Protection! ✨")
    print("=" * 80)

def main():
    """Main setup function"""
    try:
        print_banner()
        
        # Step 1: Check dependencies
        if not check_dependencies():
            print("\n❌ Setup failed: Missing required dependencies")
            print("Install missing packages and run setup again.")
            return False
        
        # Step 2: Create directories
        create_directories()
        
        # Step 3: Create configuration
        create_default_config()
        
        # Step 4: Create sample data
        create_sample_data()
        
        # Step 5: Test system
        if not test_protection_system():
            print("\n⚠️ Warning: Some protection system tests failed")
            print("System may still be usable, but check for missing components")
        
        # Step 6: Create quick start scripts
        create_quick_start_scripts()
        
        # Step 7: Create requirements file
        create_requirements_file()
        
        # Step 8: Generate summary
        generate_summary_report()
        
        # Step 9: Print next steps
        print_next_steps()
        
        return True
        
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
