# Enterprise Tracking Setup Script
# setup_tracking.py
"""
Setup script for enterprise tracking system
Handles installation, configuration, and initialization
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, List
import yaml

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.table import Table

console = Console()

class TrackingSetup:
    """
    Enterprise tracking system setup and configuration
    """
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.config_file = self.project_root / "tracking_config.yaml"
        self.requirements_file = self.project_root / "tracking_requirements.txt"
        
    def run_setup(self):
        """Run complete setup process"""
        console.print(Panel(
            "🚀 Enterprise ML Tracking System Setup",
            title="Setup Wizard",
            border_style="blue"
        ))
        
        # Check Python version
        if not self._check_python_version():
            return False
        
        # Install requirements
        if not self._install_requirements():
            return False
        
        # Setup configuration
        if not self._setup_configuration():
            return False
        
        # Initialize directories
        if not self._initialize_directories():
            return False
        
        # Test tracking system
        if not self._test_tracking_system():
            return False
        
        self._display_success_message()
        return True
    
    def _check_python_version(self) -> bool:
        """Check Python version compatibility"""
        console.print("🐍 Checking Python version...")
        
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            console.print(Panel(
                f"❌ Python {version.major}.{version.minor} detected\n"
                f"Minimum required: Python 3.8+",
                title="Python Version Error",
                border_style="red"
            ))
            return False
        
        console.print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    
    def _install_requirements(self) -> bool:
        """Install required packages"""
        if not self.requirements_file.exists():
            console.print("❌ tracking_requirements.txt not found")
            return False
        
        install = Confirm.ask("📦 Install required packages?", default=True)
        if not install:
            console.print("⏭️  Skipping package installation")
            return True
        
        console.print("📦 Installing required packages...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Installing packages...", total=None)
            
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)
                ], capture_output=True, text=True, check=True)
                
                progress.update(task, description="✅ Packages installed successfully")
                return True
                
            except subprocess.CalledProcessError as e:
                console.print(Panel(
                    f"❌ Package installation failed:\n{e.stderr}",
                    title="Installation Error",
                    border_style="red"
                ))
                return False
    
    def _setup_configuration(self) -> bool:
        """Setup tracking configuration"""
        console.print("⚙️  Setting up configuration...")
        
        if self.config_file.exists():
            overwrite = Confirm.ask(
                f"Configuration file exists at {self.config_file}. Overwrite?",
                default=False
            )
            if not overwrite:
                console.print("⏭️  Using existing configuration")
                return True
        
        # Gather configuration from user
        config = self._gather_user_config()
        
        # Save configuration
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            console.print(f"✅ Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            console.print(Panel(
                f"❌ Failed to save configuration: {e}",
                title="Configuration Error",
                border_style="red"
            ))
            return False
    
    def _gather_user_config(self) -> Dict[str, Any]:
        """Gather configuration from user input"""
        console.print("📝 Please provide configuration details:")
        
        # Basic settings
        experiment_name = Prompt.ask(
            "Default experiment name",
            default="trading_ml_production"
        )
        
        tracking_dir = Prompt.ask(
            "Tracking directory",
            default="./experiment_tracking"
        )
        
        # MLflow settings
        use_mlflow = Confirm.ask("Enable MLflow tracking?", default=True)
        mlflow_uri = "./mlruns"
        if use_mlflow:
            mlflow_uri = Prompt.ask(
                "MLflow tracking URI",
                default="./mlruns"
            )
        
        # WandB settings
        use_wandb = Confirm.ask("Enable Weights & Biases?", default=False)
        wandb_project = None
        wandb_entity = None
        if use_wandb:
            wandb_project = Prompt.ask("WandB project name", default="trading_ml")
            wandb_entity = Prompt.ask("WandB entity (username/org)", default="")
            if not wandb_entity:
                wandb_entity = None
        
        # Build configuration
        config = {
            "mlflow": {
                "enabled": use_mlflow,
                "tracking_uri": mlflow_uri,
                "experiment_name": experiment_name
            },
            "wandb": {
                "enabled": use_wandb,
                "project": wandb_project,
                "entity": wandb_entity
            },
            "local": {
                "enabled": True,
                "save_models": True,
                "save_plots": True
            },
            "tracking_dir": tracking_dir,
            "auto_log": {
                "enabled": True,
                "log_system_info": True,
                "log_git_info": True
            },
            "logging": {
                "level": "INFO",
                "file_logging": True
            }
        }
        
        return config
    
    def _initialize_directories(self) -> bool:
        """Initialize required directories"""
        console.print("📁 Creating directories...")
        
        # Load config to get directory paths
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            console.print(f"❌ Could not load config: {e}")
            return False
        
        directories = [
            config.get("tracking_dir", "./experiment_tracking"),
            "./models",
            "./artifacts", 
            "./logs",
            "./data",
            "./notebooks",
            "./scripts"
        ]
        
        for dir_path in directories:
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                console.print(f"✅ Created/verified: {dir_path}")
            except Exception as e:
                console.print(f"❌ Failed to create {dir_path}: {e}")
                return False
        
        return True
    
    def _test_tracking_system(self) -> bool:
        """Test the tracking system"""
        console.print("🧪 Testing tracking system...")
        
        try:
            # Import and test tracking
            from tracking import tracker, start_experiment
            
            with start_experiment("setup_test", "test_run") as exp:
                exp.log_params({"test_param": "setup_test"})
                exp.log_metric("test_metric", 1.0)
            
            console.print("✅ Tracking system test passed")
            return True
            
        except Exception as e:
            console.print(Panel(
                f"❌ Tracking system test failed: {e}",
                title="Test Error",
                border_style="red"
            ))
            return False
    
    def _display_success_message(self):
        """Display setup success message"""
        table = Table(title="🎉 Setup Complete!")
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Status", style="green")
        table.add_column("Location", style="yellow")
        
        table.add_row("Configuration", "✅ Ready", str(self.config_file))
        table.add_row("Tracking System", "✅ Active", "./tracking.py")
        table.add_row("Integration", "✅ Available", "./tracking_integration.py")
        table.add_row("Directories", "✅ Created", "Multiple locations")
        
        console.print(table)
        
        console.print(Panel(
            "🚀 Your enterprise tracking system is ready!\n\n"
            "Quick Start:\n"
            "```python\n"
            "from tracking import start_experiment\n\n"
            "with start_experiment('my_experiment') as exp:\n"
            "    exp.log_params({'lr': 0.01})\n"
            "    exp.log_metric('accuracy', 0.95)\n"
            "```\n\n"
            "For production monitoring:\n"
            "```python\n"
            "from tracking_integration import start_production_monitoring\n"
            "start_production_monitoring('my_model', 'deployment_1')\n"
            "```",
            title="Next Steps",
            border_style="green"
        ))

def main():
    """Main setup function"""
    setup = TrackingSetup()
    success = setup.run_setup()
    
    if success:
        console.print("🎊 Enterprise tracking setup completed successfully!")
        return 0
    else:
        console.print("💥 Setup failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
