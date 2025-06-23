#!/usr/bin/env python3
"""
Auto Setup Script for Advanced ML Protection System
===================================================

This script automatically sets up the complete Advanced ML Protection System
for ProjectP trading pipeline with one-click installation and validation.

Author: AI Assistant
Version: 2.0.0
"""

import os
import sys
import subprocess
import pkg_resources
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedProtectionAutoSetup:
    """
    Automated setup for Advanced ML Protection System
    
    Features:
    - Dependency installation and validation
    - Directory structure creation
    - Configuration file generation
    - System validation and testing
    - Example data generation
    - Quick-start guide creation
    """
    
    def __init__(self, workspace_path: str = "."):
        """Initialize auto setup"""
        self.workspace_path = Path(workspace_path).resolve()
        self.setup_log = []
        self.errors = []
        
        print("🛡️  Advanced ML Protection System - Auto Setup")
        print("=" * 60)
        print(f"📁 Workspace: {self.workspace_path}")
        print(f"🐍 Python: {sys.version}")
        print(f"📅 Setup Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
    
    def log_step(self, message: str, success: bool = True):
        """Log a setup step"""
        status = "✅" if success else "❌"
        print(f"{status} {message}")
        
        self.setup_log.append({
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'success': success
        })
        
        if not success:
            self.errors.append(message)
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        try:
            version_info = sys.version_info
            
            if version_info.major == 3 and version_info.minor >= 7:
                self.log_step(f"Python version {version_info.major}.{version_info.minor} is compatible")
                return True
            else:
                self.log_step(f"Python version {version_info.major}.{version_info.minor} is not compatible (requires 3.7+)", False)
                return False
        except Exception as e:
            self.log_step(f"Error checking Python version: {e}", False)
            return False
    
    def install_dependencies(self) -> bool:
        """Install required dependencies"""
        try:
            print("\n📦 Installing Dependencies...")
            
            # Core dependencies
            core_packages = [
                'pandas>=1.3.0',
                'numpy>=1.21.0',
                'scikit-learn>=1.0.0',
                'matplotlib>=3.3.0',
                'seaborn>=0.11.0',
                'pyyaml>=5.4.0',
                'click>=8.0.0',
                'rich>=10.0.0',
                'joblib>=1.0.0',
                'scipy>=1.7.0'
            ]
            
            # Optional dependencies for enhanced functionality
            optional_packages = [
                'psutil>=5.8.0',  # For memory monitoring
                'tqdm>=4.60.0',   # For progress bars
                'pytest>=6.0.0', # For testing
                'black>=21.0.0',  # For code formatting
            ]
            
            all_packages = core_packages + optional_packages
            
            # Check which packages are already installed
            installed_packages = [pkg.project_name.lower() for pkg in pkg_resources.working_set]
            packages_to_install = []
            
            for package in core_packages:
                package_name = package.split('>=')[0].lower()
                if package_name not in installed_packages:
                    packages_to_install.append(package)
            
            # Install missing packages
            if packages_to_install:
                print(f"📥 Installing {len(packages_to_install)} packages...")
                
                for package in packages_to_install:
                    try:
                        subprocess.check_call([
                            sys.executable, "-m", "pip", "install", package, "--quiet"
                        ])
                        self.log_step(f"Installed {package}")
                    except subprocess.CalledProcessError as e:
                        self.log_step(f"Failed to install {package}: {e}", False)
            else:
                self.log_step("All core packages already installed")
            
            # Install optional packages (don't fail if these don't install)
            for package in optional_packages:
                package_name = package.split('>=')[0].lower()
                if package_name not in installed_packages:
                    try:
                        subprocess.check_call([
                            sys.executable, "-m", "pip", "install", package, "--quiet"
                        ])
                        self.log_step(f"Installed optional package {package}")
                    except subprocess.CalledProcessError:
                        self.log_step(f"Optional package {package} not installed (non-critical)")
            
            return len(self.errors) == 0
            
        except Exception as e:
            self.log_step(f"Error installing dependencies: {e}", False)
            return False
    
    def create_directory_structure(self) -> bool:
        """Create required directory structure"""
        try:
            print("\n📁 Creating Directory Structure...")
            
            directories = [
                'protection_reports',
                'protection_backups',
                'protection_cache',
                'protection_models',
                'protection_logs',
                'protection_config',
                'protection_examples',
                'protection_tests',
                'protection_docs',
                'data',
                'models',
                'artifacts',
                'logs',
                'notebooks',
                'scripts',
                'reports'
            ]
            
            for directory in directories:
                dir_path = self.workspace_path / directory
                dir_path.mkdir(exist_ok=True)
                self.log_step(f"Created directory: {directory}")
            
            # Create .gitignore for protection system
            gitignore_content = """
# Advanced ML Protection System
protection_cache/
protection_logs/*.log
protection_backups/
*.prof
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""
            
            gitignore_path = self.workspace_path / '.gitignore'
            if not gitignore_path.exists():
                with open(gitignore_path, 'w') as f:
                    f.write(gitignore_content)
                self.log_step("Created .gitignore file")
            
            return True
            
        except Exception as e:
            self.log_step(f"Error creating directories: {e}", False)
            return False
    
    def generate_configuration_files(self) -> bool:
        """Generate configuration files"""
        try:
            print("\n⚙️  Generating Configuration Files...")
            
            # Main protection configuration
            protection_config = {
                'data_quality': {
                    'max_missing_percentage': 0.05,
                    'min_variance_threshold': 1e-6,
                    'max_correlation_threshold': 0.9,
                    'outlier_detection_method': 'isolation_forest',
                    'outlier_contamination': 0.05
                },
                'temporal_validation': {
                    'enabled': True,
                    'min_temporal_window': 50,
                    'max_lookback_days': 252,
                    'temporal_split_ratio': 0.2
                },
                'leakage_protection': {
                    'future_data_check': True,
                    'target_leakage_check': True,
                    'temporal_leakage_check': True,
                    'feature_stability_check': True,
                    'correlation_threshold': 0.95
                },
                'overfitting_protection': {
                    'cross_validation_folds': 10,
                    'max_model_complexity': 0.8,
                    'early_stopping_patience': 10,
                    'regularization_strength': 0.01,
                    'feature_selection_enabled': True,
                    'max_features_ratio': 0.3,
                    'ensemble_validation': True
                },
                'noise_reduction': {
                    'enabled': True,
                    'signal_to_noise_threshold': 3.0,
                    'smoothing_window': 5,
                    'denoising_method': 'robust_scaler'
                },
                'advanced_features': {
                    'ensemble_validation': True,
                    'market_regime_detection': True,
                    'volatility_clustering_check': True,
                    'trend_consistency_check': True,
                    'adaptive_thresholds': True
                },
                'monitoring': {
                    'performance_tracking': True,
                    'alert_threshold_auc': 0.6,
                    'alert_threshold_stability': 0.1,
                    'monitoring_window_days': 30,
                    'memory_limit_mb': 4096,
                    'processing_timeout_seconds': 3600
                },
                'storage': {
                    'backup_enabled': True,
                    'backup_frequency': 'daily',
                    'max_backup_files': 30,
                    'compression_enabled': True,
                    'report_retention_days': 90,
                    'reports_path': 'protection_reports',
                    'backups_path': 'protection_backups',
                    'cache_path': 'protection_cache',
                    'models_path': 'protection_models',
                    'logs_path': 'protection_logs'
                },
                'trading_specific': {
                    'market_hours_only': False,
                    'handle_weekends': True,
                    'volatility_regime_detection': True,
                    'trend_following_bias_check': True,
                    'tick_size_normalization': False,
                    'volume_profile_analysis': True,
                    'market_impact_consideration': True
                },
                'logging': {
                    'level': 'INFO',
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    'file_handler': True,
                    'console_handler': True,
                    'max_log_size_mb': 100,
                    'backup_count': 5
                },
                'metadata': {
                    'config_version': '2.0.0',
                    'last_updated': datetime.now().isoformat(),
                    'author': 'Advanced ML Protection System Auto Setup',
                    'description': 'Auto-generated configuration for enterprise ML protection'
                }
            }
            
            # Save main config
            config_path = self.workspace_path / 'protection_config' / 'main_config.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(protection_config, f, default_flow_style=False, sort_keys=False)
            self.log_step("Generated main protection configuration")
            
            # Trading-specific config
            trading_config = protection_config.copy()
            trading_config['data_quality']['max_missing_percentage'] = 0.02
            trading_config['overfitting_protection']['cross_validation_folds'] = 15
            trading_config['noise_reduction']['signal_to_noise_threshold'] = 2.5
            
            trading_config_path = self.workspace_path / 'protection_config' / 'trading_config.yaml'
            with open(trading_config_path, 'w') as f:
                yaml.dump(trading_config, f, default_flow_style=False, sort_keys=False)
            self.log_step("Generated trading-specific configuration")
            
            # Development config (more lenient for testing)
            dev_config = protection_config.copy()
            dev_config['data_quality']['max_missing_percentage'] = 0.1
            dev_config['overfitting_protection']['cross_validation_folds'] = 5
            dev_config['monitoring']['performance_tracking'] = False
            
            dev_config_path = self.workspace_path / 'protection_config' / 'development_config.yaml'
            with open(dev_config_path, 'w') as f:
                yaml.dump(dev_config, f, default_flow_style=False, sort_keys=False)
            self.log_step("Generated development configuration")
            
            return True
            
        except Exception as e:
            self.log_step(f"Error generating configuration files: {e}", False)
            return False
    
    def validate_system(self) -> bool:
        """Validate the protection system installation"""
        try:
            print("\n🔍 Validating System Installation...")
            
            # Check if main files exist
            required_files = [
                'advanced_ml_protection_system.py',
                'projectp_advanced_protection_integration.py',
                'advanced_ml_protection_cli.py',
                'advanced_ml_protection_examples.py'
            ]
            
            for file_name in required_files:
                file_path = self.workspace_path / file_name
                if file_path.exists():
                    self.log_step(f"Found required file: {file_name}")
                else:
                    self.log_step(f"Missing required file: {file_name}", False)
            
            # Try to import the main modules
            sys.path.insert(0, str(self.workspace_path))
            
            try:
                from advanced_ml_protection_system import AdvancedMLProtectionSystem, ProtectionConfig
                self.log_step("Successfully imported AdvancedMLProtectionSystem")
                
                # Test instantiation
                config = ProtectionConfig()
                system = AdvancedMLProtectionSystem(config)
                self.log_step("Successfully created protection system instance")
                
            except ImportError as e:
                self.log_step(f"Failed to import protection system: {e}", False)
                return False
            
            try:
                from projectp_advanced_protection_integration import ProjectPProtectionIntegration
                integration = ProjectPProtectionIntegration()
                self.log_step("Successfully created ProjectP integration")
                
            except ImportError as e:
                self.log_step(f"Failed to import ProjectP integration: {e}", False)
            
            # Test CLI
            try:
                cli_path = self.workspace_path / 'advanced_ml_protection_cli.py'
                if cli_path.exists():
                    result = subprocess.run([
                        sys.executable, str(cli_path), '--help'
                    ], capture_output=True, text=True, timeout=10)
                    
                    if result.returncode == 0:
                        self.log_step("CLI interface is working")
                    else:
                        self.log_step("CLI interface has issues", False)
                
            except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
                self.log_step(f"CLI test failed: {e}", False)
            
            return len(self.errors) == 0
            
        except Exception as e:
            self.log_step(f"Error validating system: {e}", False)
            return False
    
    def generate_example_data(self) -> bool:
        """Generate example data for testing"""
        try:
            print("\n📊 Generating Example Data...")
            
            import pandas as pd
            import numpy as np
            
            # Generate sample trading data
            np.random.seed(42)
            n_samples = 1000
            
            dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='15T')
            
            # Create realistic trading data
            data = pd.DataFrame(index=dates)
            
            # OHLCV data
            data['open'] = 100 + np.cumsum(np.random.normal(0, 0.1, n_samples))
            data['high'] = data['open'] + np.random.gamma(1, 0.05, n_samples)
            data['low'] = data['open'] - np.random.gamma(1, 0.05, n_samples)
            data['close'] = data['open'] + np.random.normal(0, 0.05, n_samples)
            data['volume'] = np.random.lognormal(10, 1, n_samples)
            
            # Technical indicators
            data['sma_20'] = data['close'].rolling(20).mean()
            data['rsi'] = 50 + np.random.normal(0, 15, n_samples)
            data['macd'] = np.random.normal(0, 0.1, n_samples)
            data['bb_upper'] = data['close'] + 2 * data['close'].rolling(20).std()
            data['bb_lower'] = data['close'] - 2 * data['close'].rolling(20).std()
            
            # Returns and target
            data['returns'] = data['close'].pct_change()
            data['target'] = (data['returns'].shift(-1) > 0).astype(int)
            
            # Add some issues for testing
            # Missing data
            data.loc[data.index[:50], 'sma_20'] = np.nan
            
            # High correlation feature
            data['correlated_feature'] = data['close'] * 1.01 + np.random.normal(0, 0.01, n_samples)
            
            # Clean the data
            data = data.dropna()
            
            # Save example data
            example_data_path = self.workspace_path / 'data' / 'example_trading_data.csv'
            data.to_csv(example_data_path, index=True)
            self.log_step(f"Generated example trading data: {len(data)} samples")
            
            # Generate clean test data
            clean_data = data.drop(columns=['correlated_feature'])
            clean_data = clean_data.fillna(method='ffill').fillna(method='bfill')
            
            clean_data_path = self.workspace_path / 'data' / 'clean_example_data.csv'
            clean_data.to_csv(clean_data_path, index=True)
            self.log_step("Generated clean example data")
            
            # Generate problematic data for testing
            problematic_data = data.copy()
            
            # Add more issues
            problematic_data['constant_feature'] = 1.0
            problematic_data['target_leak'] = problematic_data['target'] + np.random.normal(0, 0.01, len(problematic_data))
            
            # Add extreme outliers
            outlier_indices = np.random.choice(len(problematic_data), 50, replace=False)
            problematic_data.iloc[outlier_indices, problematic_data.columns.get_loc('volume')] *= 100
            
            problematic_data_path = self.workspace_path / 'data' / 'problematic_example_data.csv'
            problematic_data.to_csv(problematic_data_path, index=True)
            self.log_step("Generated problematic example data")
            
            return True
            
        except Exception as e:
            self.log_step(f"Error generating example data: {e}", False)
            return False
    
    def create_quick_start_scripts(self) -> bool:
        """Create quick start scripts and examples"""
        try:
            print("\n🚀 Creating Quick Start Scripts...")
            
            # Create quick start Python script
            quick_start_script = '''#!/usr/bin/env python3
"""
Quick Start Script for Advanced ML Protection System
Generated by Auto Setup
"""

import pandas as pd
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from advanced_ml_protection_system import AdvancedMLProtectionSystem, ProtectionConfig
    from projectp_advanced_protection_integration import ProjectPProtectionIntegration
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔧 Please run the auto setup script first")
    sys.exit(1)

def quick_protection_test():
    """Quick test of the protection system"""
    print("🛡️  Quick Protection System Test")
    print("=" * 40)
    
    # Load example data
    try:
        data_path = Path("data/example_trading_data.csv")
        if not data_path.exists():
            print("❌ Example data not found. Run auto setup with --example-data")
            return
        
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        print(f"📊 Loaded data: {data.shape}")
        
        # Initialize protection system
        protection_system = AdvancedMLProtectionSystem()
        
        # Run analysis
        print("🔍 Running protection analysis...")
        report = protection_system.analyze_data_comprehensive(
            data=data,
            target_column='target',
            feature_columns=[col for col in data.columns if col != 'target']
        )
        
        # Display results
        print(f"\\n📋 Results:")
        print(f"  Protection Score: {report.overall_protection_score:.3f}")
        print(f"  Risk Level: {report.risk_level}")
        print(f"  Data Quality: {report.data_quality_score:.3f}")
        print(f"  Issues Found: {len(report.missing_data_issues + report.correlation_issues)}")
        
        if report.recommendations:
            print(f"\\n💡 Top Recommendations:")
            for i, rec in enumerate(report.recommendations[:3], 1):
                print(f"  {i}. {rec}")
        
        print("\\n✅ Quick test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in quick test: {e}")

def quick_projectp_test():
    """Quick test of ProjectP integration"""
    print("\\n🔗 Quick ProjectP Integration Test")
    print("=" * 40)
    
    try:
        # Load example data
        data_path = Path("data/example_trading_data.csv")
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        # Initialize ProjectP integration
        integration = ProjectPProtectionIntegration()
        
        # Analyze ProjectP data
        print("🔍 Analyzing ProjectP data...")
        report = integration.analyze_projectp_data(
            data=data,
            target_column='target',
            timeframe='M15',
            market_data=True
        )
        
        print(f"\\n📋 ProjectP Results:")
        print(f"  Protection Score: {report.overall_protection_score:.3f}")
        print(f"  Trading Issues: {len([r for r in report.feature_leakage_issues if 'trading' in r.lower()])}")
        
        # Test fixes
        fixed_data, fix_summary = integration.apply_projectp_fixes(data, 'target')
        print(f"  Fixes Applied: {len(fix_summary.get('projectp_fixes', []))}")
        
        print("\\n✅ ProjectP test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in ProjectP test: {e}")

if __name__ == "__main__":
    quick_protection_test()
    quick_projectp_test()
    
    print("\\n🎯 Next Steps:")
    print("  1. Run 'python -m advanced_ml_protection_cli --help' for CLI usage")
    print("  2. Check 'protection_examples/' for detailed examples")
    print("  3. Review 'protection_config/' for configuration options")
    print("  4. Read 'ADVANCED_ML_PROTECTION_COMPLETE_GUIDE.md' for full documentation")
'''
            
            quick_start_path = self.workspace_path / 'scripts' / 'quick_start.py'
            with open(quick_start_path, 'w') as f:
                f.write(quick_start_script)
            self.log_step("Created quick start script")
            
            # Create batch file for Windows
            batch_script = f'''@echo off
echo 🛡️  Advanced ML Protection System - Quick Start
echo.
cd /d "{self.workspace_path}"
python scripts\\quick_start.py
pause
'''
            
            batch_path = self.workspace_path / 'scripts' / 'quick_start.bat'
            with open(batch_path, 'w') as f:
                f.write(batch_script)
            self.log_step("Created Windows batch script")
            
            # Create shell script for Unix
            shell_script = f'''#!/bin/bash
echo "🛡️  Advanced ML Protection System - Quick Start"
echo
cd "{self.workspace_path}"
python scripts/quick_start.py
'''
            
            shell_path = self.workspace_path / 'scripts' / 'quick_start.sh'
            with open(shell_path, 'w') as f:
                f.write(shell_script)
            
            # Make shell script executable
            try:
                import stat
                shell_path.chmod(shell_path.stat().st_mode | stat.S_IEXEC)
                self.log_step("Created Unix shell script")
            except Exception:
                self.log_step("Created Unix shell script (not executable)")
            
            return True
            
        except Exception as e:
            self.log_step(f"Error creating quick start scripts: {e}", False)
            return False
    
    def create_documentation(self) -> bool:
        """Create documentation files"""
        try:
            print("\n📚 Creating Documentation...")
            
            # Create README for the setup
            readme_content = f'''# Advanced ML Protection System - Setup Complete

## 🎉 Installation Summary

**Setup Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Workspace**: {self.workspace_path}
**Status**: {"✅ Success" if len(self.errors) == 0 else "⚠️  With Issues"}

## 📁 Directory Structure

```
{self.workspace_path.name}/
├── protection_config/          # Configuration files
├── protection_reports/         # Analysis reports
├── protection_examples/        # Example scripts
├── scripts/                   # Quick start scripts
├── data/                      # Example data files
├── models/                    # Saved models
├── logs/                      # Log files
└── docs/                      # Documentation
```

## 🚀 Quick Start

### 1. Run Quick Test
```bash
python scripts/quick_start.py
```

### 2. CLI Usage
```bash
# Analyze data
python -m advanced_ml_protection_cli analyze data/example_trading_data.csv

# Quick check
python -m advanced_ml_protection_cli quick-check data/example_trading_data.csv

# Clean data
python -m advanced_ml_protection_cli clean data/problematic_example_data.csv
```

### 3. Python API
```python
from advanced_ml_protection_system import AdvancedMLProtectionSystem
import pandas as pd

# Load data
data = pd.read_csv('data/example_trading_data.csv')

# Create protection system
protection = AdvancedMLProtectionSystem()

# Analyze data
report = protection.analyze_data_comprehensive(data, target_column='target')
print(f"Protection Score: {{report.overall_protection_score:.3f}}")
```

## 📊 Example Data Files

- `data/example_trading_data.csv` - Clean trading data for testing
- `data/problematic_example_data.csv` - Data with issues for testing fixes
- `data/clean_example_data.csv` - Pre-cleaned data

## ⚙️  Configuration Files

- `protection_config/main_config.yaml` - Main protection configuration
- `protection_config/trading_config.yaml` - Trading-specific settings
- `protection_config/development_config.yaml` - Development settings

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Permission Issues**: Check file permissions for scripts
3. **Path Issues**: Verify workspace path is correct

### Getting Help

- Check `ADVANCED_ML_PROTECTION_COMPLETE_GUIDE.md` for full documentation
- Run `python -m advanced_ml_protection_cli --help` for CLI help
- Check log files in `protection_logs/` for detailed error information

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review log files for detailed error information
3. Consult the complete guide documentation

---

*Auto-generated by Advanced ML Protection System Setup v2.0.0*
'''
            
            readme_path = self.workspace_path / 'README_SETUP.md'
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            self.log_step("Created setup README")
            
            # Create setup log file
            setup_log_path = self.workspace_path / 'protection_logs' / 'setup_log.json'
            setup_log_data = {
                'setup_timestamp': datetime.now().isoformat(),
                'workspace_path': str(self.workspace_path),
                'python_version': sys.version,
                'setup_steps': self.setup_log,
                'errors': self.errors,
                'success': len(self.errors) == 0
            }
            
            with open(setup_log_path, 'w') as f:
                json.dump(setup_log_data, f, indent=2)
            self.log_step("Created setup log file")
            
            return True
            
        except Exception as e:
            self.log_step(f"Error creating documentation: {e}", False)
            return False
    
    def run_complete_setup(self, include_examples: bool = True) -> bool:
        """Run the complete setup process"""
        try:
            print("🚀 Starting Complete Setup Process...")
            
            # Step 1: Check Python version
            if not self.check_python_version():
                print("❌ Setup failed: Incompatible Python version")
                return False
            
            # Step 2: Install dependencies
            if not self.install_dependencies():
                print("⚠️  Warning: Some dependencies failed to install")
            
            # Step 3: Create directories
            if not self.create_directory_structure():
                print("❌ Setup failed: Could not create directory structure")
                return False
            
            # Step 4: Generate configurations
            if not self.generate_configuration_files():
                print("❌ Setup failed: Could not generate configuration files")
                return False
            
            # Step 5: Validate system
            if not self.validate_system():
                print("⚠️  Warning: System validation had issues")
            
            # Step 6: Generate example data (optional)
            if include_examples:
                if not self.generate_example_data():
                    print("⚠️  Warning: Could not generate example data")
            
            # Step 7: Create quick start scripts
            if not self.create_quick_start_scripts():
                print("⚠️  Warning: Could not create quick start scripts")
            
            # Step 8: Create documentation
            if not self.create_documentation():
                print("⚠️  Warning: Could not create documentation")
            
            # Final status
            success = len(self.errors) == 0
            
            print("\n" + "=" * 60)
            if success:
                print("🎉 SETUP COMPLETED SUCCESSFULLY!")
            else:
                print("⚠️  SETUP COMPLETED WITH ISSUES")
            print("=" * 60)
            
            print(f"\n📊 Setup Summary:")
            print(f"  ✅ Steps Completed: {len([s for s in self.setup_log if s['success']])}")
            print(f"  ❌ Steps Failed: {len(self.errors)}")
            print(f"  📁 Workspace: {self.workspace_path}")
            
            if self.errors:
                print(f"\n⚠️  Issues Encountered:")
                for error in self.errors[:5]:  # Show first 5 errors
                    print(f"    - {error}")
                if len(self.errors) > 5:
                    print(f"    ... and {len(self.errors) - 5} more issues")
            
            print(f"\n🚀 Next Steps:")
            print(f"  1. Run: python scripts/quick_start.py")
            print(f"  2. Check: README_SETUP.md for usage instructions")
            print(f"  3. Review: protection_logs/setup_log.json for details")
            
            return success
            
        except Exception as e:
            self.log_step(f"Fatal error in setup process: {e}", False)
            print(f"❌ Setup failed with fatal error: {e}")
            return False

def main():
    """Main entry point for auto setup"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Auto Setup for Advanced ML Protection System"
    )
    parser.add_argument(
        '--workspace', '-w',
        default='.',
        help='Workspace directory (default: current directory)'
    )
    parser.add_argument(
        '--no-examples',
        action='store_true',
        help='Skip example data generation'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Run setup
    setup = AdvancedProtectionAutoSetup(args.workspace)
    success = setup.run_complete_setup(include_examples=not args.no_examples)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
