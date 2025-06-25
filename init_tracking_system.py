# Enterprise Tracking System Initialization
# init_tracking_system.py
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from tracking import start_experiment
            from tracking import start_experiment, tracker
from tracking_integration import production_tracker
from typing import Dict, List, Any, Optional
import json
import logging
                    import mlflow
import os
import platform
import psutil
import sys
import time
                    import wandb
import warnings
import yaml
"""
Professional initialization script for the enterprise tracking system
Sets up all components, validates configuration, and ensures system readiness
"""


# Configure console
console = Console()

class TrackingSystemInitializer:
    """
    Professional tracking system initializer with comprehensive validation
    """

    def __init__(self, config_path: str = "tracking_config.yaml"):
        self.config_path = Path(config_path)
        self.project_root = Path.cwd()
        self.config = None
        self.validation_results = {}
        self.setup_logging()

    def setup_logging(self):
        """Setup professional logging"""
        logging.basicConfig(
            level = logging.INFO, 
            format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
            handlers = [
                RichHandler(rich_tracebacks = True), 
                logging.FileHandler("logs/tracking_init.log", mode = 'a', encoding = 'utf - 8')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def initialize_system(self) -> bool:
        """
        Complete system initialization with professional validation
        """
        console.print(Panel(
            "🚀 Enterprise ML Tracking System Initialization\n"
            "Professional setup and validation process", 
            title = "System Initialization", 
            border_style = "bold blue"
        ))

        with Progress(
            SpinnerColumn(), 
            TextColumn("[progress.description]{task.description}"), 
            BarColumn(), 
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), 
            console = console
        ) as progress:

            # Define initialization tasks
            tasks = [
                ("Loading configuration", self.load_configuration), 
                ("Validating directories", self.validate_directories), 
                ("Checking dependencies", self.check_dependencies), 
                ("Validating backends", self.validate_backends), 
                ("Setting up logging", self.setup_system_logging), 
                ("Initializing MLflow", self.initialize_mlflow), 
                ("Testing system", self.test_system_functionality), 
                ("Creating templates", self.create_templates), 
                ("Setting up monitoring", self.setup_monitoring), 
                ("Finalizing setup", self.finalize_setup)
            ]

            main_task = progress.add_task("Overall Progress", total = len(tasks))

            for task_name, task_func in tasks:
                current_task = progress.add_task(task_name, total = 100)

                try:
                    progress.update(current_task, advance = 20)
                    result = task_func()
                    progress.update(current_task, advance = 80)

                    if result:
                        self.logger.info(f"✅ {task_name} completed successfully")
                        status = "✅"
                    else:
                        self.logger.warning(f"⚠️ {task_name} completed with warnings")
                        status = "⚠️"

                except Exception as e:
                    self.logger.error(f"❌ {task_name} failed: {str(e)}")
                    progress.update(current_task, description = f"❌ {task_name}")
                    status = "❌"

                progress.update(current_task, completed = 100)
                progress.update(main_task, advance = 1)
                self.validation_results[task_name] = status

        # Display results
        self.display_initialization_results()
        return self.is_system_ready()

    def load_configuration(self) -> bool:
        """Load and validate configuration file"""
        try:
            if not self.config_path.exists():
                self.logger.error(f"Configuration file not found: {self.config_path}")
                return False

            with open(self.config_path, 'r', encoding = 'utf - 8') as f:
                self.config = yaml.safe_load(f)

            # Validate required sections
            required_sections = ['mlflow', 'local', 'logging', 'tracking_dir']
            for section in required_sections:
                if section not in self.config:
                    self.logger.warning(f"Missing configuration section: {section}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return False

    def validate_directories(self) -> bool:
        """Validate and create all required directories"""
        try:
            directories = [
                self.config.get('tracking_dir', './enterprise_tracking'), 
                self.config.get('models_dir', './models'), 
                self.config.get('artifacts_dir', './artifacts'), 
                self.config.get('logs_dir', './logs'), 
                self.config.get('data_dir', './data'), 
                self.config.get('notebooks_dir', './notebooks'), 
                self.config.get('scripts_dir', './scripts'), 
                self.config.get('reports_dir', './reports'), 
                self.config.get('backup_dir', './backups'), 
                self.config.get('mlflow', {}).get('tracking_uri', './enterprise_mlruns')
            ]

            for dir_path in directories:
                if dir_path:
                    dir_path = Path(dir_path)
                    dir_path.mkdir(parents = True, exist_ok = True)

                    # Test write permissions
                    test_file = dir_path / '.test_write'
                    try:
                        test_file.write_text('test')
                        test_file.unlink()
                    except Exception as e:
                        self.logger.warning(f"Write permission issue in {dir_path}: {e}")

            return True

        except Exception as e:
            self.logger.error(f"Directory validation failed: {e}")
            return False

    def check_dependencies(self) -> bool:
        """Check all required dependencies"""
        try:
            required_packages = [
                'mlflow', 'rich', 'click', 'yaml', 'pandas', 'numpy', 
                'matplotlib', 'seaborn', 'psutil', 'joblib'
            ]

            missing_packages = []
            installed_packages = {}

            for package in required_packages:
                try:
                    module = __import__(package.replace(' - ', '_'))
                    version = getattr(module, '__version__', 'unknown')
                    installed_packages[package] = version
                except ImportError:
                    missing_packages.append(package)

            if missing_packages:
                self.logger.warning(f"Missing packages: {', '.join(missing_packages)}")
                console.print(f"⚠️ Missing packages: {', '.join(missing_packages)}")
                console.print("💡 Install with: pip install " + " ".join(missing_packages))

            # Log installed packages
            self.logger.info(f"Installed packages: {installed_packages}")

            return len(missing_packages) == 0

        except Exception as e:
            self.logger.error(f"Dependency check failed: {e}")
            return False

    def validate_backends(self) -> bool:
        """Validate tracking backends"""
        try:
            backends_status = {}

            # MLflow validation
            if self.config.get('mlflow', {}).get('enabled', False):
                try:
                    tracking_uri = self.config['mlflow']['tracking_uri']
                    mlflow.set_tracking_uri(tracking_uri)
                    backends_status['mlflow'] = '✅ Available'
                except Exception as e:
                    backends_status['mlflow'] = f'❌ Error: {e}'
            else:
                backends_status['mlflow'] = '⚠️ Disabled'

            # WandB validation
            if self.config.get('wandb', {}).get('enabled', False):
                try:
                    backends_status['wandb'] = '✅ Available'
                except ImportError:
                    backends_status['wandb'] = '❌ Not installed'
            else:
                backends_status['wandb'] = '⚠️ Disabled'

            # Local backend
            backends_status['local'] = '✅ Available' if self.config.get('local', {}).get('enabled', True) else '⚠️ Disabled'

            self.logger.info(f"Backend status: {backends_status}")
            return True

        except Exception as e:
            self.logger.error(f"Backend validation failed: {e}")
            return False

    def setup_system_logging(self) -> bool:
        """Setup comprehensive system logging"""
        try:
            log_config = self.config.get('logging', {})

            # Create log directory
            log_dir = Path(self.config.get('logs_dir', './logs'))
            log_dir.mkdir(exist_ok = True)

            # Setup log files
            log_files = {
                'tracking.log': 'Main tracking log', 
                'performance.log': 'Performance metrics log', 
                'errors.log': 'Error log', 
                'audit.log': 'Audit trail log'
            }

            for log_file, description in log_files.items():
                log_path = log_dir / log_file
                if not log_path.exists():
                    log_path.write_text(f"# {description}\n# Created: {datetime.now()}\n")

            return True

        except Exception as e:
            self.logger.error(f"Logging setup failed: {e}")
            return False

    def initialize_mlflow(self) -> bool:
        """Initialize MLflow tracking server"""
        try:
            if not self.config.get('mlflow', {}).get('enabled', False):
                return True


            # Set tracking URI
            tracking_uri = self.config['mlflow']['tracking_uri']
            mlflow.set_tracking_uri(tracking_uri)

            # Create default experiment
            experiment_name = self.config['mlflow']['experiment_name']
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    mlflow.create_experiment(experiment_name)
                    self.logger.info(f"Created MLflow experiment: {experiment_name}")
                else:
                    self.logger.info(f"MLflow experiment exists: {experiment_name}")
            except Exception as e:
                self.logger.warning(f"MLflow experiment setup warning: {e}")

            return True

        except Exception as e:
            self.logger.error(f"MLflow initialization failed: {e}")
            return False

    def test_system_functionality(self) -> bool:
        """Test basic system functionality"""
        try:
            # Test tracking import

            # Test basic experiment
            with start_experiment("system_test", "initialization_test") as exp:
                exp.log_params({"test_param": "initialization"})
                exp.log_metric("test_metric", 1.0)

            self.logger.info("System functionality test passed")
            return True

        except Exception as e:
            self.logger.error(f"System functionality test failed: {e}")
            return False

    def create_templates(self) -> bool:
        """Create useful templates and examples"""
        try:
            templates_dir = Path("templates")
            templates_dir.mkdir(exist_ok = True)

            # Create experiment template
            experiment_template = '''# Experiment Template

def run_experiment():
    """Template for running ML experiments"""
    with start_experiment("experiment_name", "run_name") as exp:
        # Log parameters
        exp.log_params({
            "param1": "value1", 
            "param2": "value2"
        })

        # Your ML code here
        # model = train_model()
        # results = evaluate_model(model)

        # Log metrics
        exp.log_metrics({
            "accuracy": 0.95, 
            "loss": 0.05
        })

        # Log model
        # exp.log_model(model, "my_model")

        print("Experiment completed!")

if __name__ == "__main__":
    run_experiment()
'''

            (templates_dir / "experiment_template.py").write_text(experiment_template, encoding = 'utf - 8')

            # Create configuration template
            config_template = '''# Configuration Template
# Copy this to your project and customize

# Project Settings
project_name: "my_ml_project"
experiment_prefix: "exp"

# Model Parameters
model:
  type: "RandomForest"
  n_estimators: 100
  max_depth: 10

# Data Settings
data:
  train_path: "data/train.csv"
  test_path: "data/test.csv"
  target_column: "target"

# Training Settings
training:
  test_size: 0.2
  random_state: 42
  cross_validation: 5
'''

            (templates_dir / "config_template.yaml").write_text(config_template, encoding = 'utf - 8')

            self.logger.info("Templates created successfully")
            return True

        except Exception as e:
            self.logger.error(f"Template creation failed: {e}")
            return False

    def setup_monitoring(self) -> bool:
        """Setup monitoring and alerting"""
        try:
            monitoring_config = self.config.get('monitoring', {})

            if not monitoring_config.get('enabled', True):
                return True

            # Create monitoring configuration file
            monitoring_dir = Path("monitoring")
            monitoring_dir.mkdir(exist_ok = True)

            # Setup monitoring script
            monitoring_script = '''# Monitoring Script

def monitor_system():
    """Basic system monitoring"""
    while True:
        cpu_percent = psutil.cpu_percent(interval = 1)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent

        print(f"{datetime.now()}: CPU: {cpu_percent}%, Memory: {memory_percent}%, Disk: {disk_percent}%")

        # Add your alerting logic here
        if cpu_percent > 90:
            print("🚨 High CPU usage alert!")

        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    monitor_system()
'''

            (monitoring_dir / "system_monitor.py").write_text(monitoring_script, encoding = 'utf - 8')

            self.logger.info("Monitoring setup completed")
            return True

        except Exception as e:
            self.logger.error(f"Monitoring setup failed: {e}")
            return False

    def finalize_setup(self) -> bool:
        """Finalize system setup"""
        try:
            # Create system info file
            system_info = {
                "setup_date": datetime.now().isoformat(), 
                "python_version": platform.python_version(), 
                "platform": platform.platform(), 
                "project_root": str(self.project_root), 
                "config_path": str(self.config_path), 
                "validation_results": self.validation_results
            }

            info_file = Path("system_info.json")
            with open(info_file, 'w', encoding = 'utf - 8') as f:
                json.dump(system_info, f, indent = 2)

            # Create quick start guide
            quick_start = '''# Quick Start Guide

## 🚀 Your tracking system is ready!

### Basic Usage:
```python

with start_experiment("my_experiment", "test_run") as exp:
    exp.log_params({"lr": 0.01})
    exp.log_metric("accuracy", 0.95)
```

### CLI Commands:
```bash
python tracking_cli.py list - experiments
python tracking_cli.py best - runs - - metric accuracy
python tracking_cli.py generate - report - - days 7
```

### Next Steps:
1. Check examples: `python tracking_examples.py all`
2. Read documentation: `TRACKING_DOCUMENTATION.md`
3. Customize config: `tracking_config.yaml`

Happy tracking! 🎉
'''

            Path("QUICK_START.md").write_text(quick_start, encoding = 'utf - 8')

            self.logger.info("System setup finalized")
            return True

        except Exception as e:
            self.logger.error(f"Setup finalization failed: {e}")
            return False

    def display_initialization_results(self):
        """Display comprehensive initialization results"""
        table = Table(title = "🔍 System Initialization Results")
        table.add_column("Component", style = "cyan", no_wrap = True)
        table.add_column("Status", style = "green")
        table.add_column("Description", style = "yellow")

        descriptions = {
            "Loading configuration": "YAML config file parsing and validation", 
            "Validating directories": "Directory structure and permissions", 
            "Checking dependencies": "Python packages and imports", 
            "Validating backends": "MLflow, WandB, and local storage", 
            "Setting up logging": "Log files and configuration", 
            "Initializing MLflow": "MLflow server and experiments", 
            "Testing system": "Basic functionality tests", 
            "Creating templates": "Example scripts and configs", 
            "Setting up monitoring": "System monitoring and alerts", 
            "Finalizing setup": "System info and documentation"
        }

        for component, status in self.validation_results.items():
            description = descriptions.get(component, "System component")
            table.add_row(component, status, description)

        console.print(table)

        # System summary
        total_tasks = len(self.validation_results)
        successful_tasks = sum(1 for status in self.validation_results.values() if status == "✅")
        warning_tasks = sum(1 for status in self.validation_results.values() if status == "⚠️")
        failed_tasks = sum(1 for status in self.validation_results.values() if status == "❌")

        summary_panel = Panel(
            f"📊 **Initialization Summary**\n\n"
            f"✅ Successful: {successful_tasks}/{total_tasks}\n"
            f"⚠️ Warnings: {warning_tasks}/{total_tasks}\n"
            f"❌ Failed: {failed_tasks}/{total_tasks}\n\n"
            f"**System Status**: {'🟢 Ready' if failed_tasks == 0 else '🟡 Partial' if warning_tasks > 0 else '🔴 Issues'}", 
            title = "System Status", 
            border_style = "green" if failed_tasks == 0 else "yellow" if warning_tasks > 0 else "red"
        )

        console.print(summary_panel)

    def is_system_ready(self) -> bool:
        """Check if system is ready for use"""
        failed_tasks = sum(1 for status in self.validation_results.values() if status == "❌")
        return failed_tasks == 0

def main():
    """Main initialization function"""
    console.print("🔧 Starting Enterprise ML Tracking System Initialization...")

    initializer = TrackingSystemInitializer()
    success = initializer.initialize_system()

    if success:
        console.print(Panel(
            "🎉 **System Initialization Complete!**\n\n"
            "Your enterprise tracking system is ready for use.\n\n"
            "**Next Steps:**\n"
            "1. Run examples: `python tracking_examples.py all`\n"
            "2. Start tracking: `from tracking import start_experiment`\n"
            "3. Use CLI tools: `python tracking_cli.py - - help`\n"
            "4. Read docs: `TRACKING_DOCUMENTATION.md`\n\n"
            "**Pro Tip:** Check `QUICK_START.md` for immediate usage!", 
            title = "🚀 Ready to Track!", 
            border_style = "bold green"
        ))
        return 0
    else:
        console.print(Panel(
            "⚠️ **System Initialization Completed with Issues**\n\n"
            "Some components have warnings or errors.\n"
            "Check the logs for details and resolve issues.\n\n"
            "**Troubleshooting:**\n"
            "1. Check `logs/tracking_init.log`\n"
            "2. Verify configuration in `tracking_config.yaml`\n"
            "3. Install missing packages\n"
            "4. Check directory permissions", 
            title = "⚠️ Attention Required", 
            border_style = "yellow"
        ))
        return 1

if __name__ == "__main__":
    sys.exit(main())