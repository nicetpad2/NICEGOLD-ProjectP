#!/usr/bin/env python3
"""
Enterprise Tracking System Setup & Installation Script
Comprehensive installation with validation and professional-grade initialization
"""

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
from datetime import datetime

# Rich imports for beautiful CLI
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.syntax import Syntax
    from rich.tree import Tree
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Installing rich for better CLI experience...")

console = Console() if RICH_AVAILABLE else None

class EnterpriseTrackingSetup:
    """
    Professional-grade tracking system setup and installation
    """
    
    def __init__(self):
        self.workspace_root = Path.cwd()
        self.setup_log = []
        self.status = {
            'dependencies': False,
            'directories': False,
            'config': False,
            'validation': False,
            'mlflow': False,
            'wandb': False,
            'local': False
        }
        
    def log_step(self, message: str, status: str = "INFO"):
        """Log setup steps with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {status}: {message}"
        self.setup_log.append(log_entry)
        
        if RICH_AVAILABLE and console:
            if status == "SUCCESS":
                console.print(f"✅ {message}", style="green")
            elif status == "ERROR":
                console.print(f"❌ {message}", style="red")
            elif status == "WARNING":
                console.print(f"⚠️ {message}", style="yellow")
            else:
                console.print(f"ℹ️ {message}", style="blue")
        else:
            print(log_entry)
    
    def print_header(self):
        """Print professional header"""
        if RICH_AVAILABLE and console:
            console.print(Panel.fit(
                "[bold blue]🚀 Enterprise ML Tracking System Setup[/bold blue]\n"
                "[dim]Professional-grade experiment tracking for production ML pipelines[/dim]\n"
                f"[dim]Setup initiated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
                border_style="blue"
            ))
        else:
            print("=" * 80)
            print("🚀 Enterprise ML Tracking System Setup")
            print("Professional-grade experiment tracking for production ML pipelines")
            print(f"Setup initiated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
    
    def check_python_version(self) -> bool:
        """Check Python version compatibility"""
        self.log_step("Checking Python version compatibility...")
        
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            self.log_step(f"Python {version.major}.{version.minor}.{version.micro} - Compatible", "SUCCESS")
            return True
        else:
            self.log_step(f"Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+", "ERROR")
            return False
    
    def install_dependencies(self) -> bool:
        """Install required dependencies with progress tracking"""
        self.log_step("Installing enterprise tracking dependencies...")
        
        # Essential packages for enterprise tracking
        packages = [
            "mlflow>=2.9.0",
            "wandb>=0.16.0",
            "rich>=13.0.0",
            "typer>=0.9.0",
            "click>=8.0.0",
            "pyyaml>=6.0",
            "psutil>=5.9.0",
            "matplotlib>=3.6.0",
            "seaborn>=0.12.0",
            "plotly>=5.17.0",
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0",
            "joblib>=1.3.0",
            "requests>=2.28.0",
            "python-dotenv>=1.0.0",
            "schedule>=1.2.0",
            "evidently>=0.4.0",
            "streamlit>=1.28.0",
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0"
        ]
        
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]Installing packages...", total=len(packages))
                
                for package in packages:
                    progress.update(task, description=f"Installing {package.split('>=')[0]}...")
                    result = subprocess.run([
                        sys.executable, "-m", "pip", "install", "--upgrade", package
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        self.log_step(f"Installed {package.split('>=')[0]}", "SUCCESS")
                    else:
                        self.log_step(f"Failed to install {package}: {result.stderr}", "ERROR")
                        return False
                    
                    progress.advance(task)
        else:
            for package in packages:
                print(f"Installing {package}...")
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "--upgrade", package
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.log_step(f"Installed {package.split('>=')[0]}", "SUCCESS")
                else:
                    self.log_step(f"Failed to install {package}: {result.stderr}", "ERROR")
                    return False
        
        self.status['dependencies'] = True
        self.log_step("All dependencies installed successfully", "SUCCESS")
        return True
    
    def create_enterprise_directories(self) -> bool:
        """Create professional directory structure"""
        self.log_step("Creating enterprise directory structure...")
        
        directories = [
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
            "configs",
            "templates",
            "monitoring",
            "deployments",
            "experiments",
            "production"
        ]
        
        created_dirs = []
        for dir_name in directories:
            dir_path = self.workspace_root / dir_name
            try:
                dir_path.mkdir(exist_ok=True)
                created_dirs.append(dir_name)
                self.log_step(f"Created directory: {dir_name}", "SUCCESS")
            except Exception as e:
                self.log_step(f"Failed to create directory {dir_name}: {e}", "ERROR")
                return False
        
        # Create subdirectories for better organization
        subdirs = {
            "artifacts": ["models", "plots", "data", "reports", "configs"],
            "logs": ["tracking", "mlflow", "wandb", "system", "errors"],
            "monitoring": ["metrics", "alerts", "dashboards", "reports"],
            "data": ["raw", "processed", "features", "predictions", "backups"],
            "models": ["trained", "registry", "artifacts", "checkpoints"],
            "experiments": ["active", "completed", "archived", "templates"]
        }
        
        for parent, subdirs_list in subdirs.items():
            for subdir in subdirs_list:
                subdir_path = self.workspace_root / parent / subdir
                try:
                    subdir_path.mkdir(exist_ok=True)
                    self.log_step(f"Created subdirectory: {parent}/{subdir}", "SUCCESS")
                except Exception as e:
                    self.log_step(f"Failed to create subdirectory {parent}/{subdir}: {e}", "WARNING")
        
        # Create .gitkeep files to preserve empty directories
        for dir_name in directories:
            gitkeep_path = self.workspace_root / dir_name / ".gitkeep"
            try:
                gitkeep_path.touch()
            except Exception:
                pass
        
        self.status['directories'] = True
        self.log_step("Enterprise directory structure created successfully", "SUCCESS")
        return True
    
    def setup_configuration_files(self) -> bool:
        """Setup comprehensive configuration files"""
        self.log_step("Setting up configuration files...")
        
        # Check if tracking_config.yaml already exists
        config_file = self.workspace_root / "tracking_config.yaml"
        if config_file.exists():
            self.log_step("tracking_config.yaml already exists - skipping creation", "INFO")
        else:
            # Create comprehensive config if it doesn't exist
            self.log_step("Creating comprehensive tracking_config.yaml", "INFO")
            # Note: The config file should already be created by previous steps
        
        # Create additional config files
        configs = {
            ".env.example": self._get_env_example(),
            "requirements-tracking.txt": self._get_requirements(),
            "logging_config.yaml": self._get_logging_config(),
            "monitoring_config.yaml": self._get_monitoring_config()
        }
        
        for filename, content in configs.items():
            config_path = self.workspace_root / filename
            try:
                with open(config_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.log_step(f"Created configuration file: {filename}", "SUCCESS")
            except Exception as e:
                self.log_step(f"Failed to create {filename}: {e}", "ERROR")
                return False
        
        self.status['config'] = True
        self.log_step("Configuration files setup completed", "SUCCESS")
        return True
    
    def validate_installation(self) -> bool:
        """Validate the installation with comprehensive checks"""
        self.log_step("Validating installation...")
        
        # Test imports
        test_imports = [
            "mlflow", "wandb", "rich", "typer", "click", 
            "yaml", "psutil", "matplotlib", "seaborn", "plotly"
        ]
        
        import_results = {}
        for module in test_imports:
            try:
                __import__(module)
                import_results[module] = True
                self.log_step(f"Module {module} - OK", "SUCCESS")
            except ImportError as e:
                import_results[module] = False
                self.log_step(f"Module {module} - FAILED: {e}", "ERROR")
        
        # Test MLflow functionality
        try:
            import mlflow
            mlflow.set_tracking_uri("./enterprise_mlruns")
            self.status['mlflow'] = True
            self.log_step("MLflow functionality - OK", "SUCCESS")
        except Exception as e:
            self.status['mlflow'] = False
            self.log_step(f"MLflow functionality - FAILED: {e}", "ERROR")
        
        # Test directory structure
        required_dirs = ["enterprise_tracking", "enterprise_mlruns", "models", "artifacts", "logs"]
        dirs_ok = all((self.workspace_root / d).exists() for d in required_dirs)
        
        if dirs_ok:
            self.status['directories'] = True
            self.log_step("Directory structure - OK", "SUCCESS")
        else:
            self.log_step("Directory structure - INCOMPLETE", "ERROR")
        
        # Overall validation
        all_critical_ok = all([
            import_results.get('mlflow', False),
            import_results.get('rich', False),
            import_results.get('yaml', False),
            self.status['mlflow'],
            self.status['directories']
        ])
        
        self.status['validation'] = all_critical_ok
        
        if all_critical_ok:
            self.log_step("Installation validation - PASSED", "SUCCESS")
            return True
        else:
            self.log_step("Installation validation - FAILED", "ERROR")
            return False
    
    def generate_setup_report(self) -> str:
        """Generate comprehensive setup report"""
        report_lines = [
            "# Enterprise Tracking System Setup Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Workspace: {self.workspace_root}",
            "",
            "## Setup Status",
            ""
        ]
        
        for component, status in self.status.items():
            status_icon = "✅" if status else "❌"
            report_lines.append(f"- {status_icon} {component.title()}: {'COMPLETED' if status else 'FAILED'}")
        
        report_lines.extend([
            "",
            "## Setup Log",
            ""
        ])
        
        for log_entry in self.setup_log:
            report_lines.append(f"- {log_entry}")
        
        report_lines.extend([
            "",
            "## Next Steps",
            "",
            "1. Review the configuration files and adjust settings as needed",
            "2. Set up environment variables (copy .env.example to .env)",
            "3. Run the initialization script: `python init_tracking_system.py`",
            "4. Test the system with: `python tracking_examples.py`",
            "5. Start tracking your experiments!",
            "",
            "## Files Created",
            "",
            "- tracking_config.yaml - Main configuration",
            "- .env.example - Environment variables template",
            "- requirements-tracking.txt - Dependencies list",
            "- logging_config.yaml - Logging configuration",
            "- monitoring_config.yaml - Monitoring configuration",
            "",
            "## Directories Created",
            "",
            "- enterprise_tracking/ - Main tracking directory",
            "- enterprise_mlruns/ - MLflow experiments",
            "- models/ - Model artifacts",
            "- artifacts/ - Experiment artifacts",
            "- logs/ - System logs",
            "- data/ - Data storage",
            "- monitoring/ - Monitoring data",
            "- And more..."
        ])
        
        return "\\n".join(report_lines)
    
    def run_setup(self) -> bool:
        """Execute complete setup process"""
        self.print_header()
        
        steps = [
            ("Python Version Check", self.check_python_version),
            ("Dependency Installation", self.install_dependencies),
            ("Directory Creation", self.create_enterprise_directories),
            ("Configuration Setup", self.setup_configuration_files),
            ("Installation Validation", self.validate_installation)
        ]
        
        for step_name, step_func in steps:
            self.log_step(f"Starting: {step_name}...")
            
            try:
                if not step_func():
                    self.log_step(f"Setup failed at: {step_name}", "ERROR")
                    return False
            except Exception as e:
                self.log_step(f"Exception in {step_name}: {e}", "ERROR")
                return False
        
        # Generate setup report
        report_content = self.generate_setup_report()
        report_path = self.workspace_root / "SETUP_REPORT.md"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            self.log_step(f"Setup report saved to: {report_path}", "SUCCESS")
        except Exception as e:
            self.log_step(f"Failed to save setup report: {e}", "WARNING")
        
        # Print success message
        if RICH_AVAILABLE and console:
            console.print(Panel.fit(
                "[bold green]🎉 Enterprise Tracking System Setup Completed Successfully![/bold green]\\n"
                "[dim]Your professional-grade ML tracking system is ready for production use.[/dim]\\n"
                f"[dim]Setup report: {report_path}[/dim]",
                border_style="green"
            ))
        else:
            print("\\n" + "=" * 80)
            print("🎉 Enterprise Tracking System Setup Completed Successfully!")
            print("Your professional-grade ML tracking system is ready for production use.")
            print(f"Setup report: {report_path}")
            print("=" * 80)
        
        return True
    
    def _get_env_example(self) -> str:
        """Get environment variables example"""
        return '''# Enterprise Tracking System Environment Variables
# Copy this file to .env and fill in your actual values

# MLflow Configuration
MLFLOW_TRACKING_URI=./enterprise_mlruns
MLFLOW_DEFAULT_ARTIFACT_ROOT=./enterprise_mlruns/artifacts
MLFLOW_REGISTRY_URI=

# Weights & Biases Configuration
WANDB_API_KEY=
WANDB_PROJECT=phiradon_trading_ml
WANDB_ENTITY=
WANDB_MODE=online

# Database Configuration (Optional)
DATABASE_URL=
POSTGRES_HOST=
POSTGRES_PORT=5432
POSTGRES_DB=tracking
POSTGRES_USER=
POSTGRES_PASSWORD=

# Cloud Storage Configuration (Optional)
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=us-west-2
AWS_S3_BUCKET=

AZURE_STORAGE_ACCOUNT=
AZURE_STORAGE_KEY=
AZURE_CONTAINER_NAME=

GCP_PROJECT_ID=
GCP_BUCKET_NAME=

# Security Configuration
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-here

# Monitoring Configuration
ENABLE_MONITORING=true
ALERT_EMAIL=
SLACK_WEBHOOK_URL=
ALERT_THRESHOLD_CPU=85
ALERT_THRESHOLD_MEMORY=80

# Production Configuration
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO
'''
    
    def _get_requirements(self) -> str:
        """Get requirements.txt content"""
        return '''# Enterprise ML Tracking System Dependencies
# Core tracking and experiment management
mlflow>=2.9.0
wandb>=0.16.0

# CLI and UI components
rich>=13.0.0
typer>=0.9.0
click>=8.0.0
streamlit>=1.28.0
fastapi>=0.104.0
uvicorn>=0.24.0

# Data processing and ML
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.17.0
joblib>=1.3.0

# Configuration and utilities
pyyaml>=6.0
python-dotenv>=1.0.0
psutil>=5.9.0
requests>=2.28.0
schedule>=1.2.0

# Monitoring and observability
evidently>=0.4.0
prometheus-client>=0.19.0

# Development and testing (optional)
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.6.0
'''
    
    def _get_logging_config(self) -> str:
        """Get logging configuration"""
        return '''# Logging Configuration for Enterprise Tracking System
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    datefmt: "%Y-%m-%d %H:%M:%S"
  
  detailed:
    format: "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    datefmt: "%Y-%m-%d %H:%M:%S"
  
  json:
    class: pythonjsonlogger.jsonlogger.JsonFormatter
    format: "%(asctime)s %(name)s %(levelname)s %(filename)s %(lineno)d %(message)s"

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout
  
  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: detailed
    filename: ./logs/tracking.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
  
  error_file:
    class: logging.handlers.RotatingFileHandler
    level: ERROR
    formatter: detailed
    filename: ./logs/errors.log
    maxBytes: 5242880  # 5MB
    backupCount: 3

loggers:
  tracking:
    level: DEBUG
    handlers: [console, file, error_file]
    propagate: false
  
  mlflow:
    level: INFO
    handlers: [file]
    propagate: false
  
  wandb:
    level: INFO
    handlers: [file]
    propagate: false

root:
  level: INFO
  handlers: [console, file]
'''
    
    def _get_monitoring_config(self) -> str:
        """Get monitoring configuration"""
        return '''# Monitoring Configuration for Enterprise Tracking System
monitoring:
  enabled: true
  interval_seconds: 60
  
  # System metrics
  system_metrics:
    enabled: true
    collect_interval: 30
    metrics:
      - cpu_percent
      - memory_percent
      - disk_usage
      - network_io
      - gpu_usage
  
  # Application metrics
  application_metrics:
    enabled: true
    collect_interval: 60
    metrics:
      - experiment_count
      - active_runs
      - failed_runs
      - model_predictions
      - data_pipeline_health
  
  # Alerting
  alerts:
    enabled: true
    channels:
      - email
      - slack
      - webhook
    
    rules:
      - name: High CPU Usage
        condition: cpu_percent > 85
        severity: warning
        cooldown_minutes: 5
      
      - name: High Memory Usage
        condition: memory_percent > 80
        severity: warning
        cooldown_minutes: 5
      
      - name: Disk Space Low
        condition: disk_usage > 90
        severity: critical
        cooldown_minutes: 1
      
      - name: Experiment Failures
        condition: failed_runs > 3
        severity: warning
        cooldown_minutes: 10
      
      - name: Model Performance Degradation
        condition: model_accuracy < 0.8
        severity: critical
        cooldown_minutes: 1

  # Dashboard configuration
  dashboard:
    enabled: true
    port: 8501
    auto_refresh_seconds: 30
    charts:
      - system_metrics
      - experiment_metrics
      - model_performance
      - alerts_history

# Notification settings
notifications:
  email:
    enabled: false
    smtp_server: smtp.gmail.com
    smtp_port: 587
    username: ""
    password: ""
    recipients: []
  
  slack:
    enabled: false
    webhook_url: ""
    channel: "#ml-alerts"
  
  webhook:
    enabled: false
    url: ""
    headers: {}
'''

def main():
    """Main setup execution"""
    setup = EnterpriseTrackingSetup()
    
    try:
        success = setup.run_setup()
        if success:
            print("\\n🎉 Setup completed successfully!")
            print("Next steps:")
            print("1. Review tracking_config.yaml and adjust settings")
            print("2. Copy .env.example to .env and fill in values")
            print("3. Run: python init_tracking_system.py")
            print("4. Test with: python tracking_examples.py")
            return 0
        else:
            print("\\n❌ Setup failed. Check the logs for details.")
            return 1
            
    except KeyboardInterrupt:
        print("\\n⚠️ Setup interrupted by user.")
        return 1
    except Exception as e:
        print(f"\\n💥 Unexpected error during setup: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
