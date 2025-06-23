"""
Advanced ML Protection CLI for ProjectP Trading Pipeline
======================================================

Command-line interface for the Advanced ML Protection System
providing easy access to all protection features and monitoring capabilities.

Author: AI Assistant
Version: 2.0.0
"""

import click
import pandas as pd
import numpy as np
import yaml
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import sys
import os

# Import protection system components
try:
    from advanced_ml_protection_system import AdvancedMLProtectionSystem, ProtectionConfig
    from projectp_advanced_protection_integration import ProjectPProtectionIntegration
except ImportError as e:
    print(f"Error importing protection system: {e}")
    print("Please ensure all protection system files are in the same directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@click.group()
@click.version_option(version='2.0.0')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--config', '-c', type=click.Path(exists=True), help='Protection config file path')
@click.pass_context
def cli(ctx, verbose, config):
    """
    Advanced ML Protection CLI for ProjectP Trading Pipeline
    
    Comprehensive protection against noise, data leakage, and overfitting
    for trading machine learning pipelines.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Set logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Store config path
    ctx.obj['config_path'] = config

@cli.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--target', '-t', default='target', help='Target column name')
@click.option('--timeframe', default='M15', help='Trading timeframe (M1, M5, M15, H1, D1)')
@click.option('--output', '-o', type=click.Path(), help='Output report file path')
@click.option('--format', 'output_format', type=click.Choice(['json', 'yaml', 'text']), default='json', help='Output format')
@click.pass_context
def analyze(ctx, data_file, target, timeframe, output, output_format):
    """
    Analyze data file for protection issues
    
    DATA_FILE: Path to CSV file containing trading data
    """
    try:
        click.echo(f"🔍 Analyzing data file: {data_file}")
        click.echo(f"📊 Target column: {target}")
        click.echo(f"⏱️  Timeframe: {timeframe}")
        
        # Load data
        try:
            data = pd.read_csv(data_file)
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data.set_index('timestamp', inplace=True)
            elif 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data.set_index('date', inplace=True)
        except Exception as e:
            click.echo(f"❌ Error loading data: {e}", err=True)
            return
        
        click.echo(f"📈 Data loaded: {data.shape[0]} samples, {data.shape[1]} features")
        
        # Initialize protection integration
        config_path = ctx.obj.get('config_path')
        integration = ProjectPProtectionIntegration(config_path=config_path)
        
        # Run analysis
        with click.progressbar(length=100, label='Running protection analysis') as bar:
            bar.update(20)
            report = integration.analyze_projectp_data(
                data=data,
                target_column=target,
                timeframe=timeframe,
                market_data=True
            )
            bar.update(80)
        
        # Display results
        click.echo("\n" + "="*60)
        click.echo("🛡️  PROTECTION ANALYSIS RESULTS")
        click.echo("="*60)
        
        # Overall scores
        overall_score = report.overall_protection_score
        risk_level = report.risk_level
        
        if risk_level == 'low':
            score_color = 'green'
        elif risk_level == 'medium':
            score_color = 'yellow'
        else:
            score_color = 'red'
        
        click.echo(f"Overall Protection Score: {click.style(f'{overall_score:.3f}', fg=score_color, bold=True)}")
        click.echo(f"Risk Level: {click.style(risk_level.upper(), fg=score_color, bold=True)}")
        click.echo(f"Data Quality Score: {report.data_quality_score:.3f}")
        click.echo(f"Overfitting Risk: {report.overfitting_risk.upper()}")
        click.echo(f"Noise Level: {report.noise_level.upper()}")
        click.echo(f"Signal-to-Noise Ratio: {report.signal_to_noise_ratio:.2f}")
        
        # Issues summary
        total_issues = (
            len(report.missing_data_issues) +
            len(report.correlation_issues) +
            len(report.feature_leakage_issues)
        )
        
        click.echo(f"\n📋 Issues Detected: {total_issues}")
        
        if report.missing_data_issues:
            click.echo(f"  • Missing Data Issues: {len(report.missing_data_issues)}")
        
        if report.correlation_issues:
            click.echo(f"  • Correlation Issues: {len(report.correlation_issues)}")
        
        if report.feature_leakage_issues:
            click.echo(f"  • Feature Leakage Issues: {len(report.feature_leakage_issues)}")
        
        if report.outlier_count > 0:
            click.echo(f"  • Outliers Detected: {report.outlier_count}")
        
        if report.target_leakage_detected:
            click.echo(f"  • {click.style('Target Leakage Detected!', fg='red', bold=True)}")
        
        # Recommendations
        if report.recommendations:
            click.echo(f"\n💡 Recommendations ({len(report.recommendations)}):")
            for i, rec in enumerate(report.recommendations[:5], 1):  # Show top 5
                click.echo(f"  {i}. {rec}")
            
            if len(report.recommendations) > 5:
                click.echo(f"  ... and {len(report.recommendations) - 5} more recommendations")
        
        # Performance metrics
        click.echo(f"\n⚡ Performance:")
        click.echo(f"  • Processing Time: {report.processing_time_seconds:.2f} seconds")
        click.echo(f"  • Memory Usage: {report.memory_usage_mb:.1f} MB")
        
        # Save output if requested
        if output:
            save_report(report, output, output_format)
            click.echo(f"\n💾 Report saved to: {output}")
        
    except Exception as e:
        click.echo(f"❌ Error during analysis: {e}", err=True)
        logger.error(f"Analysis error: {e}", exc_info=True)

@cli.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--target', '-t', default='target', help='Target column name')
@click.option('--output', '-o', type=click.Path(), help='Output file for cleaned data')
@click.option('--backup', is_flag=True, help='Create backup of original data')
@click.pass_context
def clean(ctx, data_file, target, output, backup):
    """
    Clean data file by applying automated fixes
    
    DATA_FILE: Path to CSV file to clean
    """
    try:
        click.echo(f"🧹 Cleaning data file: {data_file}")
        
        # Load data
        data = pd.read_csv(data_file)
        original_shape = data.shape
        
        # Create backup if requested
        if backup:
            backup_path = f"{data_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            data.to_csv(backup_path, index=False)
            click.echo(f"💾 Backup created: {backup_path}")
        
        # Initialize protection integration
        config_path = ctx.obj.get('config_path')
        integration = ProjectPProtectionIntegration(config_path=config_path)
        
        # Analyze data first
        with click.progressbar(length=100, label='Analyzing data') as bar:
            bar.update(30)
            report = integration.analyze_projectp_data(data, target_column=target)
            bar.update(70)
        
        # Apply fixes
        with click.progressbar(length=100, label='Applying fixes') as bar:
            bar.update(20)
            fixed_data, fix_summary = integration.apply_projectp_fixes(data, target)
            bar.update(80)
        
        # Save cleaned data
        output_path = output or f"{Path(data_file).stem}_cleaned.csv"
        fixed_data.to_csv(output_path, index=False)
        
        # Display results
        click.echo("\n" + "="*60)
        click.echo("🧹 DATA CLEANING RESULTS")
        click.echo("="*60)
        
        click.echo(f"Original data shape: {original_shape}")
        click.echo(f"Cleaned data shape: {fixed_data.shape}")
        
        samples_removed = original_shape[0] - fixed_data.shape[0]
        features_removed = original_shape[1] - fixed_data.shape[1]
        
        if samples_removed > 0:
            click.echo(f"Samples removed: {samples_removed}")
        
        if features_removed > 0:
            click.echo(f"Features removed: {features_removed}")
        
        # Show fixes applied
        if fix_summary['projectp_fixes']:
            click.echo(f"\n🔧 Fixes Applied ({len(fix_summary['projectp_fixes'])}):")
            for fix in fix_summary['projectp_fixes']:
                click.echo(f"  • {fix}")
        
        click.echo(f"\n💾 Cleaned data saved to: {output_path}")
        
    except Exception as e:
        click.echo(f"❌ Error during cleaning: {e}", err=True)
        logger.error(f"Cleaning error: {e}", exc_info=True)

@cli.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--target', '-t', default='target', help='Target column name')
@click.option('--quick', is_flag=True, help='Quick validation (less comprehensive)')
@click.pass_context
def validate(ctx, data_file, target, quick):
    """
    Validate data file for ML pipeline readiness
    
    DATA_FILE: Path to CSV file to validate
    """
    try:
        click.echo(f"✅ Validating data file: {data_file}")
        
        # Load data
        data = pd.read_csv(data_file)
        click.echo(f"📊 Data shape: {data.shape}")
        
        # Initialize protection system
        config_path = ctx.obj.get('config_path')
        config = load_protection_config(config_path)
        protection_system = AdvancedMLProtectionSystem(config)
        
        # Run validation
        validation_label = 'Quick validation' if quick else 'Full validation'
        with click.progressbar(length=100, label=validation_label) as bar:
            bar.update(25)
            
            # Basic checks
            basic_checks = run_basic_validation(data, target)
            bar.update(25)
            
            # Advanced checks (skip if quick mode)
            if not quick:
                report = protection_system.analyze_data_comprehensive(data, target)
                bar.update(50)
            else:
                bar.update(50)
        
        # Display validation results
        click.echo("\n" + "="*60)
        click.echo("✅ VALIDATION RESULTS")
        click.echo("="*60)
        
        # Basic validation results
        display_basic_validation_results(basic_checks)
        
        # Advanced validation results (if not quick mode)
        if not quick:
            display_advanced_validation_results(report)
        
        # Overall validation status
        if basic_checks['all_passed'] and (quick or report.overall_protection_score > 0.6):
            click.echo(f"\n{click.style('✅ VALIDATION PASSED', fg='green', bold=True)}")
        else:
            click.echo(f"\n{click.style('❌ VALIDATION FAILED', fg='red', bold=True)}")
        
    except Exception as e:
        click.echo(f"❌ Error during validation: {e}", err=True)
        logger.error(f"Validation error: {e}", exc_info=True)

@cli.command()
@click.option('--config-file', type=click.Path(), help='Configuration file to create/edit')
@click.option('--template', type=click.Choice(['basic', 'advanced', 'trading']), default='advanced', help='Configuration template')
def config(config_file, template):
    """
    Create or edit protection system configuration
    """
    try:
        config_path = config_file or 'advanced_ml_protection_config.yaml'
        
        click.echo(f"⚙️  Creating configuration: {config_path}")
        click.echo(f"📄 Template: {template}")
        
        # Generate configuration based on template
        if template == 'basic':
            config_dict = generate_basic_config()
        elif template == 'trading':
            config_dict = generate_trading_config()
        else:  # advanced
            config_dict = generate_advanced_config()
        
        # Save configuration
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        click.echo(f"✅ Configuration created: {config_path}")
        click.echo(f"\n📝 Edit the file to customize settings for your specific needs.")
        
        # Show key configuration sections
        click.echo(f"\n🔧 Key Configuration Sections:")
        for section in config_dict.keys():
            click.echo(f"  • {section}")
        
    except Exception as e:
        click.echo(f"❌ Error creating configuration: {e}", err=True)

@cli.command()
@click.pass_context
def status(ctx):
    """
    Show protection system status and recent activity
    """
    try:
        click.echo("📊 Protection System Status")
        click.echo("="*40)
        
        # Check for recent reports
        reports_dir = Path('protection_reports')
        if reports_dir.exists():
            report_files = list(reports_dir.glob('*.json'))
            report_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            if report_files:
                click.echo(f"📄 Recent Reports: {len(report_files)}")
                
                # Show latest report summary
                latest_report_path = report_files[0]
                try:
                    with open(latest_report_path, 'r') as f:
                        latest_report = json.load(f)
                    
                    click.echo(f"\n📋 Latest Analysis:")
                    click.echo(f"  • Timestamp: {latest_report.get('timestamp', 'Unknown')}")
                    click.echo(f"  • Protection Score: {latest_report.get('overall_protection_score', 'N/A')}")
                    click.echo(f"  • Risk Level: {latest_report.get('risk_level', 'Unknown')}")
                    
                except Exception as e:
                    click.echo(f"  ⚠️  Could not read latest report: {e}")
            else:
                click.echo("📄 No recent reports found")
        else:
            click.echo("📄 No reports directory found")
        
        # Check configuration
        config_path = ctx.obj.get('config_path') or 'advanced_ml_protection_config.yaml'
        if Path(config_path).exists():
            click.echo(f"⚙️  Configuration: {config_path} ✅")
        else:
            click.echo(f"⚙️  Configuration: Not found ❌")
        
        # Check system health
        click.echo(f"\n🔧 System Health:")
        
        try:
            # Test imports
            from advanced_ml_protection_system import AdvancedMLProtectionSystem
            click.echo("  • Core protection system: ✅")
        except ImportError:
            click.echo("  • Core protection system: ❌")
        
        try:
            from projectp_advanced_protection_integration import ProjectPProtectionIntegration
            click.echo("  • ProjectP integration: ✅")
        except ImportError:
            click.echo("  • ProjectP integration: ❌")
        
        # Check dependencies
        required_packages = ['pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn', 'yaml']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            click.echo(f"  • Missing packages: {', '.join(missing_packages)} ❌")
        else:
            click.echo("  • Required packages: ✅")
        
        click.echo(f"\n🚀 Protection System Ready!")
        
    except Exception as e:
        click.echo(f"❌ Error checking status: {e}", err=True)

@cli.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--target', '-t', default='target', help='Target column name')
@click.option('--max-issues', default=5, help='Maximum issues to check')
def quick_check(data_file, target, max_issues):
    """
    Quick health check for data file
    
    DATA_FILE: Path to CSV file to check
    """
    try:
        click.echo(f"⚡ Quick check: {data_file}")
        
        # Load data sample for quick check
        data = pd.read_csv(data_file, nrows=1000)  # Sample first 1000 rows
        
        issues = []
        
        # Check 1: Missing data
        missing_pct = data.isnull().mean().mean()
        if missing_pct > 0.1:
            issues.append(f"High missing data: {missing_pct:.1%}")
        
        # Check 2: Target column exists
        if target not in data.columns:
            issues.append(f"Target column '{target}' not found")
        
        # Check 3: Sufficient samples
        if len(data) < 100:
            issues.append(f"Too few samples: {len(data)}")
        
        # Check 4: Constant features
        if target in data.columns:
            feature_cols = [col for col in data.columns if col != target]
            numeric_cols = data[feature_cols].select_dtypes(include=[np.number]).columns
            constant_features = []
            for col in numeric_cols:
                if data[col].nunique() <= 1:
                    constant_features.append(col)
            
            if constant_features:
                issues.append(f"Constant features: {len(constant_features)}")
        
        # Check 5: Target distribution
        if target in data.columns:
            target_dist = data[target].value_counts()
            if len(target_dist) == 1:
                issues.append("Target has only one class")
            elif len(target_dist) > 1:
                min_class_ratio = target_dist.min() / len(data)
                if min_class_ratio < 0.01:
                    issues.append(f"Severe class imbalance: {min_class_ratio:.1%}")
        
        # Display results
        click.echo(f"📊 Samples checked: {len(data)}")
        click.echo(f"📈 Features: {len([col for col in data.columns if col != target])}")
        
        if issues:
            click.echo(f"\n⚠️  Issues Found ({len(issues)}):")
            for i, issue in enumerate(issues[:max_issues], 1):
                click.echo(f"  {i}. {issue}")
            
            if len(issues) > max_issues:
                click.echo(f"  ... and {len(issues) - max_issues} more issues")
            
            click.echo(f"\n💡 Run 'analyze' command for detailed analysis")
        else:
            click.echo(f"\n✅ No major issues detected in quick check")
        
    except Exception as e:
        click.echo(f"❌ Error in quick check: {e}", err=True)

@cli.command()
@click.option('--target-dir', type=click.Path(), default='.', help='Target directory for integration')
@click.option('--example-data', is_flag=True, help='Generate example data for testing')
def projectp_integrate(target_dir, example_data):
    """
    Set up ProjectP integration with protection system
    """
    try:
        target_path = Path(target_dir)
        click.echo(f"🔗 Setting up ProjectP integration in: {target_path}")
        
        # Create integration script
        integration_script = target_path / 'projectp_protection_setup.py'
        
        integration_code = generate_integration_script()
        
        with open(integration_script, 'w') as f:
            f.write(integration_code)
        
        click.echo(f"✅ Integration script created: {integration_script}")
        
        # Create example configuration
        example_config = target_path / 'protection_config_example.yaml'
        config_dict = generate_trading_config()
        
        with open(example_config, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        click.echo(f"✅ Example configuration created: {example_config}")
        
        # Generate example data if requested
        if example_data:
            example_data_path = target_path / 'example_trading_data.csv'
            example_df = generate_example_trading_data()
            example_df.to_csv(example_data_path, index=False)
            click.echo(f"✅ Example data created: {example_data_path}")
        
        # Create usage instructions
        instructions_path = target_path / 'PROJECTP_PROTECTION_USAGE.md'
        instructions = generate_usage_instructions()
        
        with open(instructions_path, 'w') as f:
            f.write(instructions)
        
        click.echo(f"✅ Usage instructions created: {instructions_path}")
        
        click.echo(f"\n🚀 ProjectP integration setup complete!")
        click.echo(f"📖 Read {instructions_path} for usage instructions")
        
    except Exception as e:
        click.echo(f"❌ Error setting up integration: {e}", err=True)

# Helper functions

def save_report(report, output_path, format_type):
    """Save protection report in specified format"""
    if format_type == 'json':
        report_dict = {
            'timestamp': report.timestamp.isoformat(),
            'overall_protection_score': report.overall_protection_score,
            'risk_level': report.risk_level,
            'data_quality_score': report.data_quality_score,
            'overfitting_risk': report.overfitting_risk,
            'noise_level': report.noise_level,
            'signal_to_noise_ratio': report.signal_to_noise_ratio,
            'missing_data_issues': report.missing_data_issues,
            'correlation_issues': report.correlation_issues,
            'feature_leakage_issues': report.feature_leakage_issues,
            'recommendations': report.recommendations,
            'processing_time_seconds': report.processing_time_seconds,
            'memory_usage_mb': report.memory_usage_mb
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
    
    elif format_type == 'yaml':
        # Similar to JSON but YAML format
        report_dict = {
            'timestamp': report.timestamp.isoformat(),
            'scores': {
                'overall_protection_score': report.overall_protection_score,
                'data_quality_score': report.data_quality_score,
                'signal_to_noise_ratio': report.signal_to_noise_ratio
            },
            'risk_assessment': {
                'risk_level': report.risk_level,
                'overfitting_risk': report.overfitting_risk,
                'noise_level': report.noise_level
            },
            'issues': {
                'missing_data_issues': report.missing_data_issues,
                'correlation_issues': report.correlation_issues,
                'feature_leakage_issues': report.feature_leakage_issues
            },
            'recommendations': report.recommendations,
            'performance': {
                'processing_time_seconds': report.processing_time_seconds,
                'memory_usage_mb': report.memory_usage_mb
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(report_dict, f, default_flow_style=False)
    
    else:  # text format
        with open(output_path, 'w') as f:
            f.write("ADVANCED ML PROTECTION SYSTEM REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Timestamp: {report.timestamp}\n")
            f.write(f"Overall Protection Score: {report.overall_protection_score:.3f}\n")
            f.write(f"Risk Level: {report.risk_level}\n")
            f.write(f"Data Quality Score: {report.data_quality_score:.3f}\n")
            f.write(f"Overfitting Risk: {report.overfitting_risk}\n")
            f.write(f"Noise Level: {report.noise_level}\n")
            f.write(f"Signal-to-Noise Ratio: {report.signal_to_noise_ratio:.2f}\n\n")
            
            if report.missing_data_issues:
                f.write("Missing Data Issues:\n")
                for issue in report.missing_data_issues:
                    f.write(f"  - {issue}\n")
                f.write("\n")
            
            if report.correlation_issues:
                f.write("Correlation Issues:\n")
                for issue in report.correlation_issues:
                    f.write(f"  - {issue}\n")
                f.write("\n")
            
            if report.feature_leakage_issues:
                f.write("Feature Leakage Issues:\n")
                for issue in report.feature_leakage_issues:
                    f.write(f"  - {issue}\n")
                f.write("\n")
            
            if report.recommendations:
                f.write("Recommendations:\n")
                for i, rec in enumerate(report.recommendations, 1):
                    f.write(f"  {i}. {rec}\n")

def load_protection_config(config_path):
    """Load protection configuration from file"""
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            return ProtectionConfig(
                max_missing_percentage=config_dict.get('data_quality', {}).get('max_missing_percentage', 0.05),
                temporal_validation_enabled=config_dict.get('temporal_validation', {}).get('enabled', True),
                noise_detection_enabled=config_dict.get('noise_reduction', {}).get('enabled', True)
            )
        except Exception:
            pass
    
    return ProtectionConfig()

def run_basic_validation(data, target):
    """Run basic validation checks"""
    checks = {
        'target_exists': target in data.columns,
        'sufficient_samples': len(data) >= 100,
        'no_all_nan_columns': not data.isnull().all().any(),
        'target_has_variation': data[target].nunique() > 1 if target in data.columns else False,
        'no_inf_values': not np.isinf(data.select_dtypes(include=[np.number])).any().any()
    }
    
    checks['all_passed'] = all(checks.values())
    return checks

def display_basic_validation_results(checks):
    """Display basic validation results"""
    click.echo("🔍 Basic Validation:")
    
    for check_name, passed in checks.items():
        if check_name == 'all_passed':
            continue
        
        status = "✅" if passed else "❌"
        readable_name = check_name.replace('_', ' ').title()
        click.echo(f"  {status} {readable_name}")

def display_advanced_validation_results(report):
    """Display advanced validation results"""
    click.echo("\n🔬 Advanced Validation:")
    click.echo(f"  • Protection Score: {report.overall_protection_score:.3f}")
    click.echo(f"  • Data Quality: {report.data_quality_score:.3f}")
    click.echo(f"  • Overfitting Risk: {report.overfitting_risk}")
    click.echo(f"  • Noise Level: {report.noise_level}")

def generate_basic_config():
    """Generate basic configuration template"""
    return {
        'data_quality': {
            'max_missing_percentage': 0.1,
            'min_variance_threshold': 1e-8,
            'max_correlation_threshold': 0.95
        },
        'overfitting_protection': {
            'cross_validation_folds': 5,
            'feature_selection_enabled': True
        },
        'monitoring': {
            'performance_tracking': True
        }
    }

def generate_advanced_config():
    """Generate advanced configuration template"""
    return {
        'data_quality': {
            'max_missing_percentage': 0.05,
            'min_variance_threshold': 1e-6,
            'max_correlation_threshold': 0.9,
            'outlier_contamination': 0.05
        },
        'temporal_validation': {
            'enabled': True,
            'min_temporal_window': 50
        },
        'overfitting_protection': {
            'cross_validation_folds': 10,
            'feature_selection_enabled': True,
            'max_features_ratio': 0.3,
            'ensemble_validation': True
        },
        'noise_reduction': {
            'enabled': True,
            'signal_to_noise_threshold': 3.0,
            'denoising_method': 'robust_scaler'
        },
        'advanced_features': {
            'market_regime_detection': True,
            'volatility_clustering_check': True
        },
        'monitoring': {
            'performance_tracking': True,
            'alert_threshold_auc': 0.6
        }
    }

def generate_trading_config():
    """Generate trading-specific configuration template"""
    return {
        'data_quality': {
            'max_missing_percentage': 0.02,
            'max_correlation_threshold': 0.85,
            'outlier_contamination': 0.03
        },
        'temporal_validation': {
            'enabled': True,
            'min_temporal_window': 100
        },
        'leakage_protection': {
            'future_data_check': True,
            'target_leakage_check': True,
            'correlation_threshold': 0.95
        },
        'overfitting_protection': {
            'cross_validation_folds': 10,
            'max_features_ratio': 0.2,
            'regularization_strength': 0.01
        },
        'noise_reduction': {
            'enabled': True,
            'signal_to_noise_threshold': 2.5,
            'smoothing_window': 3
        },
        'trading_specific': {
            'market_hours_only': False,
            'handle_weekends': True,
            'volatility_regime_detection': True
        },
        'monitoring': {
            'performance_tracking': True,
            'alert_threshold_auc': 0.65,
            'monitoring_window_days': 30
        }
    }

def generate_integration_script():
    """Generate ProjectP integration script"""
    return '''#!/usr/bin/env python3
"""
ProjectP Protection Integration Script
Generated by Advanced ML Protection CLI
"""

from projectp_advanced_protection_integration import ProjectPProtectionIntegration
import pandas as pd

def setup_projectp_protection():
    """Setup protection for ProjectP pipeline"""
    
    # Initialize protection integration
    integration = ProjectPProtectionIntegration(
        config_path='protection_config_example.yaml'
    )
    
    print("🛡️  ProjectP Protection System initialized")
    return integration

def protect_projectp_data(data_file, target_column='target'):
    """Protect ProjectP data with comprehensive analysis"""
    
    # Load data
    data = pd.read_csv(data_file)
    
    # Initialize protection
    integration = setup_projectp_protection()
    
    # Analyze data
    report = integration.analyze_projectp_data(
        data=data,
        target_column=target_column,
        timeframe='M15'  # Adjust as needed
    )
    
    # Apply fixes if needed
    if report.overall_protection_score < 0.7:
        print("⚠️  Low protection score, applying fixes...")
        fixed_data, fix_summary = integration.apply_projectp_fixes(data, target_column)
        
        # Save fixed data
        output_file = data_file.replace('.csv', '_protected.csv')
        fixed_data.to_csv(output_file, index=False)
        print(f"💾 Protected data saved to: {output_file}")
        
        return fixed_data, report
    
    return data, report

if __name__ == "__main__":
    # Example usage
    print("🚀 Testing ProjectP Protection Integration")
    
    # Replace with your actual data file
    data_file = "example_trading_data.csv"
    
    if Path(data_file).exists():
        protected_data, analysis_report = protect_projectp_data(data_file)
        print(f"✅ Protection complete. Score: {analysis_report.overall_protection_score:.3f}")
    else:
        print(f"❌ Data file not found: {data_file}")
        print("💡 Run CLI with --example-data to generate sample data")
'''

def generate_example_trading_data():
    """Generate example trading data for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate timestamps
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='15T')
    
    # Create DataFrame
    data = pd.DataFrame()
    data['timestamp'] = dates
    
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
    data['bb_upper'] = data['close'] + 2
    data['bb_lower'] = data['close'] - 2
    
    # Returns and target
    data['returns'] = data['close'].pct_change()
    data['target'] = (data['returns'].shift(-1) > 0).astype(int)
    
    # Remove NaN values
    data = data.dropna()
    
    return data

def generate_usage_instructions():
    """Generate usage instructions for ProjectP integration"""
    return '''# ProjectP Advanced ML Protection System

## Quick Start

1. **Analyze your data:**
```bash
python -m advanced_ml_protection_cli analyze your_data.csv --target target_column
```

2. **Clean your data:**
```bash
python -m advanced_ml_protection_cli clean your_data.csv --output cleaned_data.csv
```

3. **Quick health check:**
```bash
python -m advanced_ml_protection_cli quick-check your_data.csv
```

## Integration with ProjectP

Use the generated `projectp_protection_setup.py` script:

```python
from projectp_protection_setup import protect_projectp_data

# Protect your ProjectP data
protected_data, report = protect_projectp_data('your_data.csv', 'target')

# Check protection score
print(f"Protection Score: {report.overall_protection_score:.3f}")
```

## Configuration

Edit `protection_config_example.yaml` to customize protection settings:

- Data quality thresholds
- Overfitting protection parameters
- Noise reduction settings
- Trading-specific configurations

## CLI Commands

- `analyze`: Comprehensive data analysis
- `clean`: Apply automated fixes
- `validate`: Validate data for ML readiness
- `config`: Create/edit configuration
- `status`: Show system status
- `quick-check`: Fast health check
- `projectp-integrate`: Setup integration

## Best Practices

1. Always run analysis before training models
2. Use clean command to fix common issues
3. Monitor protection scores over time
4. Customize configuration for your specific use case
5. Regular validation of new data

For more information, run: `python -m advanced_ml_protection_cli --help`
'''

if __name__ == '__main__':
    cli()
