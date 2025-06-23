#!/usr/bin/env python3
# 🛡️ ML Protection System CLI
# Command-line interface for advanced ML protection features

import click
import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ml_protection_system import MLProtectionSystem, ProtectionLevel
    from projectp_protection_integration import ProjectPProtectionIntegration
    PROTECTION_AVAILABLE = True
except ImportError as e:
    print(f"❌ Error: ML Protection system not available: {e}")
    PROTECTION_AVAILABLE = False
    sys.exit(1)

# CLI Configuration
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(version='2.0.0')
def cli():
    """
    🛡️ ML Protection System CLI
    
    Advanced protection against noise, data leakage, and overfitting
    for enterprise ML trading systems.
    """
    pass

@cli.command()
@click.option('--data', '-d', required=True, type=click.Path(exists=True),
              help='Path to CSV data file')
@click.option('--target', '-t', default='target',
              help='Target column name (default: target)')
@click.option('--timestamp', '-ts', default='timestamp',
              help='Timestamp column name (default: timestamp)')
@click.option('--level', '-l', type=click.Choice(['basic', 'standard', 'aggressive', 'enterprise']),
              default='standard', help='Protection level (default: standard)')
@click.option('--output', '-o', type=click.Path(),
              help='Output directory for results')
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Path to protection config file')
@click.option('--model', '-m', type=click.Path(exists=True),
              help='Path to trained model file (optional)')
@click.option('--report/--no-report', default=True,
              help='Generate HTML report (default: True)')
@click.option('--verbose', '-v', is_flag=True,
              help='Verbose output')
def analyze(data, target, timestamp, level, output, config, model, report, verbose):
    """
    🔍 Analyze dataset for noise, leakage, and overfitting issues
    
    Example:
        ml_protection_cli.py analyze -d data.csv -l enterprise --report
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)
    
    click.echo(f"🛡️ Starting ML Protection Analysis")
    click.echo(f"📊 Data: {data}")
    click.echo(f"🎯 Protection Level: {level}")
    
    try:
        # Load data
        df = pd.read_csv(data)
        click.echo(f"📈 Loaded data shape: {df.shape}")
        
        # Initialize protection system
        protection_system = MLProtectionSystem(
            protection_level=ProtectionLevel(level),
            config_path=config
        )
        
        # Load model if provided
        trained_model = None
        if model:
            import joblib
            trained_model = joblib.load(model)
            click.echo(f"🤖 Loaded model: {model}")
        
        # Run protection analysis
        with click.progressbar(length=100, label='Analyzing') as bar:
            bar.update(25)
            result = protection_system.protect_dataset(
                data=df,
                target_col=target,
                timestamp_col=timestamp,
                model=trained_model
            )
            bar.update(100)
        
        # Display results
        click.echo("\n📊 Protection Analysis Results:")
        click.echo(f"✅ Overall Clean: {result.is_clean}")
        click.echo(f"🔇 Noise Score: {result.noise_score:.3f}")
        click.echo(f"💧 Leakage Score: {result.leakage_score:.3f}")
        click.echo(f"📈 Overfitting Score: {result.overfitting_score:.3f}")
        
        if result.issues_found:
            click.echo("\n⚠️ Issues Found:")
            for issue in result.issues_found:
                click.echo(f"  • {issue}")
        
        if result.recommendations:
            click.echo("\n💡 Recommendations:")
            for rec in result.recommendations:
                click.echo(f"  • {rec}")
        
        # Save results
        if output:
            output_dir = Path(output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save cleaned data
            if result.cleaned_data is not None:
                cleaned_path = output_dir / f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                result.cleaned_data.to_csv(cleaned_path, index=False)
                click.echo(f"💾 Cleaned data saved: {cleaned_path}")
            
            # Save analysis report
            report_data = {
                'analysis_timestamp': datetime.now().isoformat(),
                'protection_level': level,
                'data_file': str(data),
                'results': {
                    'is_clean': result.is_clean,
                    'noise_score': result.noise_score,
                    'leakage_score': result.leakage_score,
                    'overfitting_score': result.overfitting_score,
                    'issues_found': result.issues_found,
                    'recommendations': result.recommendations
                }
            }
            
            report_path = output_dir / f"protection_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            click.echo(f"📄 Analysis report saved: {report_path}")
            
            # Generate HTML report
            if report:
                html_report = protection_system.generate_protection_report(
                    result,
                    str(output_dir / f"protection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
                )
                click.echo(f"📊 HTML report generated: {html_report}")
        
        # Exit with appropriate code
        sys.exit(0 if result.is_clean else 1)
        
    except Exception as e:
        click.echo(f"❌ Error during analysis: {str(e)}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@cli.command()
@click.option('--data', '-d', required=True, type=click.Path(exists=True),
              help='Path to CSV data file')
@click.option('--target', '-t', default='target',
              help='Target column name (default: target)')
@click.option('--timestamp', '-ts', default='timestamp',
              help='Timestamp column name (default: timestamp)')
@click.option('--level', '-l', type=click.Choice(['basic', 'standard', 'aggressive', 'enterprise']),
              default='standard', help='Protection level (default: standard)')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output path for cleaned data')
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Path to protection config file')
@click.option('--backup/--no-backup', default=True,
              help='Create backup of original data')
def clean(data, target, timestamp, level, output, config, backup):
    """
    🧹 Clean dataset by removing noise and problematic data
    
    Example:
        ml_protection_cli.py clean -d data.csv -o cleaned_data.csv
    """
    click.echo(f"🧹 Starting Data Cleaning")
    click.echo(f"📊 Input: {data}")
    click.echo(f"💾 Output: {output}")
    
    try:
        # Load data
        df = pd.read_csv(data)
        original_shape = df.shape
        click.echo(f"📈 Original data shape: {original_shape}")
        
        # Create backup if requested
        if backup:
            backup_path = Path(data).parent / f"{Path(data).stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(backup_path, index=False)
            click.echo(f"💾 Backup created: {backup_path}")
        
        # Initialize protection system
        protection_system = MLProtectionSystem(
            protection_level=ProtectionLevel(level),
            config_path=config
        )
        
        # Run cleaning
        with click.progressbar(length=100, label='Cleaning') as bar:
            bar.update(50)
            result = protection_system.protect_dataset(
                data=df,
                target_col=target,
                timestamp_col=timestamp
            )
            bar.update(100)
        
        # Save cleaned data
        if result.cleaned_data is not None:
            result.cleaned_data.to_csv(output, index=False)
            cleaned_shape = result.cleaned_data.shape
            
            click.echo(f"✅ Data cleaned successfully!")
            click.echo(f"📊 Shape change: {original_shape} → {cleaned_shape}")
            click.echo(f"🗑️ Rows removed: {original_shape[0] - cleaned_shape[0]}")
            click.echo(f"💾 Cleaned data saved: {output}")
            
            if result.issues_found:
                click.echo("\n🔧 Issues addressed:")
                for issue in result.issues_found:
                    click.echo(f"  • {issue}")
        else:
            # No cleaning needed, save original
            df.to_csv(output, index=False)
            click.echo("✅ No cleaning required. Original data copied to output.")
        
    except Exception as e:
        click.echo(f"❌ Error during cleaning: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--data', '-d', required=True, type=click.Path(exists=True),
              help='Path to CSV data file')
@click.option('--target', '-t', default='target',
              help='Target column name (default: target)')
@click.option('--timestamp', '-ts', default='timestamp',
              help='Timestamp column name (default: timestamp)')
@click.option('--output', '-o', type=click.Path(),
              help='Output directory for report')
def validate(data, target, timestamp, output):
    """
    ✅ Validate dataset for ML readiness
    
    Example:
        ml_protection_cli.py validate -d data.csv
    """
    click.echo(f"✅ Validating Dataset for ML Readiness")
    
    try:
        # Load data
        df = pd.read_csv(data)
        click.echo(f"📊 Data shape: {df.shape}")
        
        # Basic validation checks
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'data_file': str(data),
            'checks': {}
        }
        
        # Check 1: Required columns
        required_cols = [target, timestamp]
        missing_cols = [col for col in required_cols if col not in df.columns]
        validation_results['checks']['required_columns'] = {
            'status': 'PASS' if not missing_cols else 'FAIL',
            'missing_columns': missing_cols
        }
        
        # Check 2: Data types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        validation_results['checks']['data_types'] = {
            'numeric_columns': len(numeric_cols),
            'total_columns': len(df.columns),
            'numeric_ratio': len(numeric_cols) / len(df.columns)
        }
        
        # Check 3: Missing values
        missing_summary = df.isnull().sum()
        high_missing_cols = missing_summary[missing_summary > len(df) * 0.5].index.tolist()
        validation_results['checks']['missing_values'] = {
            'total_missing': int(missing_summary.sum()),
            'high_missing_columns': high_missing_cols,
            'status': 'PASS' if not high_missing_cols else 'WARN'
        }
        
        # Check 4: Target distribution
        if target in df.columns:
            target_info = {
                'unique_values': int(df[target].nunique()),
                'null_count': int(df[target].isnull().sum()),
                'data_type': str(df[target].dtype)
            }
            
            if df[target].dtype in ['int64', 'float64']:
                target_info.update({
                    'mean': float(df[target].mean()),
                    'std': float(df[target].std()),
                    'min': float(df[target].min()),
                    'max': float(df[target].max())
                })
            
            validation_results['checks']['target_analysis'] = target_info
        
        # Check 5: Timestamp validation
        if timestamp in df.columns:
            try:
                ts_series = pd.to_datetime(df[timestamp])
                timestamp_info = {
                    'status': 'PASS',
                    'date_range': f"{ts_series.min()} to {ts_series.max()}",
                    'is_sorted': bool(ts_series.is_monotonic_increasing),
                    'duplicates': int(ts_series.duplicated().sum())
                }
            except:
                timestamp_info = {
                    'status': 'FAIL',
                    'error': 'Cannot parse timestamp column'
                }
            
            validation_results['checks']['timestamp_validation'] = timestamp_info
        
        # Display results
        click.echo("\n📋 Validation Results:")
        
        for check_name, check_result in validation_results['checks'].items():
            status = check_result.get('status', 'INFO')
            status_symbol = {'PASS': '✅', 'FAIL': '❌', 'WARN': '⚠️', 'INFO': 'ℹ️'}.get(status, 'ℹ️')
            click.echo(f"{status_symbol} {check_name.replace('_', ' ').title()}: {status}")
        
        # Show detailed info
        click.echo(f"\n📊 Dataset Summary:")
        click.echo(f"  • Total Rows: {len(df):,}")
        click.echo(f"  • Total Columns: {len(df.columns)}")
        click.echo(f"  • Numeric Columns: {len(numeric_cols)}")
        click.echo(f"  • Missing Values: {missing_summary.sum():,}")
        
        if target in df.columns:
            click.echo(f"  • Target Unique Values: {df[target].nunique()}")
        
        # Save validation report
        if output:
            output_dir = Path(output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            report_path = output_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(validation_results, f, indent=2)
            click.echo(f"📄 Validation report saved: {report_path}")
        
        # Exit code based on validation
        failed_checks = [name for name, result in validation_results['checks'].items() 
                        if result.get('status') == 'FAIL']
        sys.exit(1 if failed_checks else 0)
        
    except Exception as e:
        click.echo(f"❌ Error during validation: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--level', '-l', type=click.Choice(['basic', 'standard', 'aggressive', 'enterprise']),
              default='standard', help='Protection level')
@click.option('--output', '-o', type=click.Path(), default='ml_protection_config.yaml',
              help='Output config file path')
def init_config(level, output):
    """
    🔧 Initialize protection system configuration
    
    Example:
        ml_protection_cli.py init-config -l enterprise -o my_config.yaml
    """
    click.echo(f"🔧 Initializing {level} protection configuration")
    
    # Generate config based on level
    configs = {
        'basic': {
            'protection': {'level': 'basic'},
            'noise': {'contamination_rate': 0.05, 'volatility_threshold': 3.0},
            'leakage': {'temporal_gap_hours': 12, 'target_leakage_threshold': 0.9},
            'overfitting': {'max_features_ratio': 0.5, 'min_samples_per_feature': 5}
        },
        'standard': {
            'protection': {'level': 'standard'},
            'noise': {'contamination_rate': 0.1, 'volatility_threshold': 2.5},
            'leakage': {'temporal_gap_hours': 24, 'target_leakage_threshold': 0.8},
            'overfitting': {'max_features_ratio': 0.3, 'min_samples_per_feature': 10}
        },
        'aggressive': {
            'protection': {'level': 'aggressive'},
            'noise': {'contamination_rate': 0.15, 'volatility_threshold': 2.0},
            'leakage': {'temporal_gap_hours': 48, 'target_leakage_threshold': 0.6},
            'overfitting': {'max_features_ratio': 0.2, 'min_samples_per_feature': 20}
        },
        'enterprise': {
            'protection': {'level': 'enterprise'},
            'noise': {'contamination_rate': 0.2, 'volatility_threshold': 1.5},
            'leakage': {'temporal_gap_hours': 72, 'target_leakage_threshold': 0.5},
            'overfitting': {'max_features_ratio': 0.15, 'min_samples_per_feature': 30}
        }
    }
    
    config = configs[level]
    
    # Add common settings
    config.update({
        'monitoring': {
            'enable_realtime_monitoring': True,
            'alert_thresholds': {
                'noise_score': 0.2,
                'leakage_score': 0.1,
                'overfitting_score': 0.3
            }
        },
        'reporting': {
            'generate_html_report': True,
            'include_visualizations': True
        }
    })
    
    # Save config
    with open(output, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    click.echo(f"✅ Configuration saved: {output}")
    click.echo(f"🎯 Protection level: {level}")
    click.echo("📝 Edit the file to customize settings")

@cli.command()
def status():
    """
    📊 Show protection system status
    """
    click.echo("🛡️ ML Protection System Status")
    click.echo("="*50)
    
    # Check if system is available
    if PROTECTION_AVAILABLE:
        click.echo("✅ Protection System: Available")
    else:
        click.echo("❌ Protection System: Not Available")
        return
    
    # Check config files
    config_files = ['ml_protection_config.yaml', 'tracking_config.yaml']
    for config_file in config_files:
        if Path(config_file).exists():
            click.echo(f"✅ Config File: {config_file}")
        else:
            click.echo(f"⚠️ Config File: {config_file} (missing)")
    
    # Check dependencies
    dependencies = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('scikit-learn', 'sklearn'),
        ('mlflow', 'mlflow'),
        ('yaml', 'yaml')
    ]
    
    for dep_name, dep_import in dependencies:
        try:
            __import__(dep_import)
            click.echo(f"✅ Dependency: {dep_name}")
        except ImportError:
            click.echo(f"❌ Dependency: {dep_name} (missing)")
    
    # Show available protection levels
    click.echo("\n🎯 Available Protection Levels:")
    for level in ProtectionLevel:
        click.echo(f"  • {level.value}")
    
    click.echo("\n💡 Quick Start:")
    click.echo("  ml_protection_cli.py analyze -d your_data.csv -l enterprise")

@cli.command()
@click.argument('data_file', type=click.Path(exists=True))
def quick_check(data_file):
    """
    ⚡ Quick protection check for a dataset
    
    Example:
        ml_protection_cli.py quick-check data.csv
    """
    click.echo(f"⚡ Quick Protection Check: {data_file}")
    
    try:
        # Load data
        df = pd.read_csv(data_file)
        
        # Basic checks
        checks = []
        
        # Check 1: Data size
        if len(df) < 100:
            checks.append("⚠️ Small dataset (< 100 rows)")
        elif len(df) > 1000000:
            checks.append("ℹ️ Large dataset (> 1M rows)")
        else:
            checks.append("✅ Good dataset size")
        
        # Check 2: Missing values
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_pct > 50:
            checks.append("❌ Too many missing values (>50%)")
        elif missing_pct > 20:
            checks.append("⚠️ High missing values (>20%)")
        else:
            checks.append("✅ Acceptable missing values")
        
        # Check 3: Numeric columns
        numeric_ratio = len(df.select_dtypes(include=[np.number]).columns) / len(df.columns)
        if numeric_ratio < 0.3:
            checks.append("⚠️ Few numeric columns")
        else:
            checks.append("✅ Good numeric column ratio")
        
        # Check 4: Duplicates
        duplicate_pct = (df.duplicated().sum() / len(df)) * 100
        if duplicate_pct > 10:
            checks.append("⚠️ High duplicate rows (>10%)")
        else:
            checks.append("✅ Low duplicate rows")
        
        # Display results
        click.echo(f"📊 Shape: {df.shape}")
        for check in checks:
            click.echo(f"  {check}")
        
        # Recommendation
        issues = [check for check in checks if check.startswith('❌') or check.startswith('⚠️')]
        if issues:
            click.echo("\n💡 Recommendation: Run full analysis")
            click.echo("  ml_protection_cli.py analyze -d {} -l standard".format(data_file))
        else:
            click.echo("\n✅ Dataset looks good for ML!")
        
    except Exception as e:
        click.echo(f"❌ Error during quick check: {str(e)}", err=True)

if __name__ == '__main__':
    cli()
