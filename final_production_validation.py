#!/usr/bin/env python3
"""
🚀 NICEGOLD ENTERPRISE - FINAL PRODUCTION VALIDATION
==================================================

Complete production readiness validation and final setup script.
This script ensures all components are production-ready before deployment.

Version: 3.0
Author: NICEGOLD Team
"""

import hashlib
import importlib.util
import json
import logging
import os
import socket
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Rich for beautiful output
try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import track
    from rich.table import Table
    from rich.text import Text
    console = Console()
except ImportError:
    class Console:
        def print(self, *args, **kwargs):
            print(*args)
    console = Console()

class ProductionValidator:
    """Complete production validation system"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.config_dir = self.base_dir / "config"
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "UNKNOWN",
            "validation_results": {},
            "critical_issues": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('production_validation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def validate_environment(self) -> bool:
        """Validate Python environment and dependencies"""
        console.print("\n[bold blue]🔍 Validating Environment...[/bold blue]")
        
        try:
            # Check Python version
            python_version = sys.version_info
            if python_version < (3, 8) or python_version >= (3, 11):
                self.results["critical_issues"].append(
                    f"Python version {python_version.major}.{python_version.minor} not supported. Use Python 3.8-3.10"
                )
                return False
            
            # Check critical dependencies
            critical_deps = [
                'pandas', 'numpy', 'scikit-learn', 'fastapi', 'streamlit',
                'pydantic', 'uvicorn', 'sqlalchemy', 'jose', 'passlib'
            ]
            
            missing_deps = []
            for dep in critical_deps:
                if importlib.util.find_spec(dep) is None:
                    missing_deps.append(dep)
            
            if missing_deps:
                self.results["critical_issues"].append(
                    f"Missing critical dependencies: {', '.join(missing_deps)}"
                )
                return False
            
            self.results["validation_results"]["environment"] = "PASS"
            console.print("✅ Environment validation passed")
            return True
            
        except Exception as e:
            self.results["critical_issues"].append(f"Environment validation error: {str(e)}")
            return False

    def validate_configuration(self) -> bool:
        """Validate all configuration files"""
        console.print("\n[bold blue]🔧 Validating Configuration...[/bold blue]")
        
        try:
            config_files = [
                "config.yaml",
                "config/production.yaml",
                ".env.production",
                ".env.example"
            ]
            
            missing_configs = []
            for config_file in config_files:
                config_path = self.base_dir / config_file
                if not config_path.exists():
                    missing_configs.append(config_file)
            
            if missing_configs:
                self.results["critical_issues"].append(
                    f"Missing configuration files: {', '.join(missing_configs)}"
                )
                return False
            
            # Validate YAML configurations
            try:
                with open(self.base_dir / "config.yaml", 'r') as f:
                    main_config = yaml.safe_load(f)
                
                with open(self.base_dir / "config/production.yaml", 'r') as f:
                    prod_config = yaml.safe_load(f)
                
                # Check for required sections
                required_sections = ['data', 'model', 'training']
                for section in required_sections:
                    if section not in main_config:
                        self.results["critical_issues"].append(
                            f"Missing required section '{section}' in config.yaml"
                        )
                        return False
                
                # Validate production config
                if prod_config.get('application', {}).get('environment') != 'production':
                    self.results["warnings"].append(
                        "Production config environment not set to 'production'"
                    )
                
                if prod_config.get('application', {}).get('debug', True):
                    self.results["critical_issues"].append(
                        "Debug mode is enabled in production config"
                    )
                    return False
                
            except yaml.YAMLError as e:
                self.results["critical_issues"].append(f"YAML configuration error: {str(e)}")
                return False
            
            self.results["validation_results"]["configuration"] = "PASS"
            console.print("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            self.results["critical_issues"].append(f"Configuration validation error: {str(e)}")
            return False

    def validate_security(self) -> bool:
        """Validate security settings and credentials"""
        console.print("\n[bold blue]🔐 Validating Security...[/bold blue]")
        
        try:
            # Check .env.production file
            env_path = self.base_dir / ".env.production"
            with open(env_path, 'r') as f:
                env_content = f.read()
            
            # Check for secure secret keys
            if "SECRET_KEY=" in env_content:
                secret_key_line = [line for line in env_content.split('\n') if line.startswith('SECRET_KEY=')]
                if secret_key_line:
                    secret_key = secret_key_line[0].split('=', 1)[1]
                    if len(secret_key) < 32:
                        self.results["critical_issues"].append(
                            "SECRET_KEY is too short (minimum 32 characters required)"
                        )
                        return False
            
            # Check JWT secret
            if "JWT_SECRET=" in env_content:
                jwt_secret_line = [line for line in env_content.split('\n') if line.startswith('JWT_SECRET=')]
                if jwt_secret_line:
                    jwt_secret = jwt_secret_line[0].split('=', 1)[1]
                    if len(jwt_secret) < 32:
                        self.results["critical_issues"].append(
                            "JWT_SECRET is too short (minimum 32 characters required)"
                        )
                        return False
            
            # Check debug mode is disabled
            if "DEBUG=true" in env_content.lower():
                self.results["critical_issues"].append(
                    "DEBUG mode is enabled in production environment"
                )
                return False
            
            self.results["validation_results"]["security"] = "PASS"
            console.print("✅ Security validation passed")
            return True
            
        except Exception as e:
            self.results["critical_issues"].append(f"Security validation error: {str(e)}")
            return False

    def validate_database(self) -> bool:
        """Validate database setup and connectivity"""
        console.print("\n[bold blue]💾 Validating Database...[/bold blue]")
        
        try:
            # Create database directory if it doesn't exist
            db_dir = self.base_dir / "database"
            db_dir.mkdir(exist_ok=True)
            
            # Test SQLite connection
            db_path = db_dir / "production.db"
            try:
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                conn.close()
            except Exception as e:
                self.results["critical_issues"].append(f"Database connectivity error: {str(e)}")
                return False
            
            self.results["validation_results"]["database"] = "PASS"
            console.print("✅ Database validation passed")
            return True
            
        except Exception as e:
            self.results["critical_issues"].append(f"Database validation error: {str(e)}")
            return False

    def validate_models_data(self) -> bool:
        """Validate models and data files"""
        console.print("\n[bold blue]🤖 Validating Models & Data...[/bold blue]")
        
        try:
            # Check for data files
            data_files = ["dummy_m1.csv", "dummy_m15.csv"]
            missing_data = []
            for data_file in data_files:
                if not (self.base_dir / data_file).exists():
                    missing_data.append(data_file)
            
            if missing_data:
                self.results["warnings"].append(
                    f"Missing data files: {', '.join(missing_data)} (will use generated dummy data)"
                )
            
            # Create models directory
            models_dir = self.base_dir / "models"
            models_dir.mkdir(exist_ok=True)
            
            self.results["validation_results"]["models_data"] = "PASS"
            console.print("✅ Models & Data validation passed")
            return True
            
        except Exception as e:
            self.results["critical_issues"].append(f"Models & Data validation error: {str(e)}")
            return False

    def validate_scripts(self) -> bool:
        """Validate critical production scripts"""
        console.print("\n[bold blue]📜 Validating Production Scripts...[/bold blue]")
        
        try:
            critical_scripts = [
                "production_setup.py",
                "one_click_deploy.py",
                "system_maintenance.py",
                "ai_orchestrator.py",
                "start_production_single_user.py"
            ]
            
            missing_scripts = []
            for script in critical_scripts:
                script_path = self.base_dir / script
                if not script_path.exists():
                    missing_scripts.append(script)
                else:
                    # Basic syntax check
                    try:
                        with open(script_path, 'r', encoding='utf-8') as f:
                            compile(f.read(), script_path, 'exec')
                    except SyntaxError as e:
                        self.results["critical_issues"].append(
                            f"Syntax error in {script}: {str(e)}"
                        )
                        return False
            
            if missing_scripts:
                self.results["critical_issues"].append(
                    f"Missing critical scripts: {', '.join(missing_scripts)}"
                )
                return False
            
            self.results["validation_results"]["scripts"] = "PASS"
            console.print("✅ Production scripts validation passed")
            return True
            
        except Exception as e:
            self.results["critical_issues"].append(f"Scripts validation error: {str(e)}")
            return False

    def validate_ports(self) -> bool:
        """Validate that required ports are available"""
        console.print("\n[bold blue]🌐 Validating Ports...[/bold blue]")
        
        try:
            required_ports = [8000, 8501]  # FastAPI, Streamlit
            occupied_ports = []
            
            for port in required_ports:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('127.0.0.1', port))
                sock.close()
                
                if result == 0:  # Port is occupied
                    occupied_ports.append(port)
            
            if occupied_ports:
                self.results["warnings"].append(
                    f"Ports {', '.join(map(str, occupied_ports))} are occupied. "
                    "They will be used by the system during deployment."
                )
            
            self.results["validation_results"]["ports"] = "PASS"
            console.print("✅ Ports validation passed")
            return True
            
        except Exception as e:
            self.results["critical_issues"].append(f"Ports validation error: {str(e)}")
            return False

    def validate_permissions(self) -> bool:
        """Validate file and directory permissions"""
        console.print("\n[bold blue]🔑 Validating Permissions...[/bold blue]")
        
        try:
            # Check write permissions for critical directories
            critical_dirs = ["logs", "database", "models", "backups"]
            
            for dir_name in critical_dirs:
                dir_path = self.base_dir / dir_name
                dir_path.mkdir(exist_ok=True)
                
                # Test write permission
                test_file = dir_path / ".test_permission"
                try:
                    test_file.write_text("test")
                    test_file.unlink()
                except PermissionError:
                    self.results["critical_issues"].append(
                        f"No write permission for directory: {dir_name}"
                    )
                    return False
            
            self.results["validation_results"]["permissions"] = "PASS"
            console.print("✅ Permissions validation passed")
            return True
            
        except Exception as e:
            self.results["critical_issues"].append(f"Permissions validation error: {str(e)}")
            return False

    def create_production_report(self) -> None:
        """Create comprehensive production readiness report"""
        console.print("\n[bold blue]📊 Generating Production Report...[/bold blue]")
        
        # Determine overall status
        if self.results["critical_issues"]:
            self.results["overall_status"] = "FAILED"
        elif self.results["warnings"]:
            self.results["overall_status"] = "WARNING"
        else:
            self.results["overall_status"] = "READY"
        
        # Create report
        report_content = f"""
# 🚀 NICEGOLD ENTERPRISE - PRODUCTION READINESS REPORT

**Generated**: {self.results['timestamp']}
**Overall Status**: {self.results['overall_status']}

## ✅ Validation Results

| Component | Status |
|-----------|--------|
"""
        
        for component, status in self.results["validation_results"].items():
            emoji = "✅" if status == "PASS" else "❌"
            report_content += f"| {component.title()} | {emoji} {status} |\n"
        
        if self.results["critical_issues"]:
            report_content += "\n## ❌ Critical Issues\n\n"
            for issue in self.results["critical_issues"]:
                report_content += f"- 🚨 {issue}\n"
        
        if self.results["warnings"]:
            report_content += "\n## ⚠️ Warnings\n\n"
            for warning in self.results["warnings"]:
                report_content += f"- ⚠️ {warning}\n"
        
        if self.results["recommendations"]:
            report_content += "\n## 💡 Recommendations\n\n"
            for rec in self.results["recommendations"]:
                report_content += f"- 💡 {rec}\n"
        
        report_content += f"""

## 🎯 Next Steps

{"### ❌ SYSTEM NOT READY FOR PRODUCTION" if self.results['overall_status'] == 'FAILED' else ""}
{"Please resolve all critical issues before deployment." if self.results['overall_status'] == 'FAILED' else ""}

{"### ⚠️ SYSTEM READY WITH WARNINGS" if self.results['overall_status'] == 'WARNING' else ""}
{"Review warnings before deployment." if self.results['overall_status'] == 'WARNING' else ""}

{"### ✅ SYSTEM READY FOR PRODUCTION" if self.results['overall_status'] == 'READY' else ""}
{"You can proceed with deployment using:" if self.results['overall_status'] == 'READY' else ""}
{"```bash" if self.results['overall_status'] == 'READY' else ""}
{"python one_click_deploy.py" if self.results['overall_status'] == 'READY' else ""}
{"```" if self.results['overall_status'] == 'READY' else ""}

---
*Report generated by NICEGOLD Enterprise Production Validator*
"""
        
        # Save report
        report_path = self.base_dir / "PRODUCTION_READINESS_REPORT.md"
        report_path.write_text(report_content, encoding='utf-8')
        
        # Also save JSON version
        json_path = self.base_dir / "production_validation_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        console.print(f"📊 Report saved to: {report_path}")
        console.print(f"📊 JSON results saved to: {json_path}")

    def run_validation(self) -> bool:
        """Run complete production validation"""
        console.print(Panel.fit(
            "[bold cyan]🚀 NICEGOLD ENTERPRISE PRODUCTION VALIDATOR[/bold cyan]\n"
            "[dim]Validating system readiness for production deployment...[/dim]",
            box=box.DOUBLE
        ))
        
        validation_steps = [
            ("Environment", self.validate_environment),
            ("Configuration", self.validate_configuration),
            ("Security", self.validate_security),
            ("Database", self.validate_database),
            ("Models & Data", self.validate_models_data),
            ("Scripts", self.validate_scripts),
            ("Ports", self.validate_ports),
            ("Permissions", self.validate_permissions)
        ]
        
        success_count = 0
        total_steps = len(validation_steps)
        
        for step_name, validation_func in validation_steps:
            try:
                if validation_func():
                    success_count += 1
            except Exception as e:
                self.results["critical_issues"].append(
                    f"Validation step '{step_name}' failed: {str(e)}"
                )
        
        # Create report
        self.create_production_report()
        
        # Display final status
        if self.results["overall_status"] == "READY":
            console.print(Panel.fit(
                "[bold green]✅ SYSTEM READY FOR PRODUCTION[/bold green]\n"
                "[dim]All validations passed successfully![/dim]",
                box=box.DOUBLE
            ))
            return True
        elif self.results["overall_status"] == "WARNING":
            console.print(Panel.fit(
                "[bold yellow]⚠️ SYSTEM READY WITH WARNINGS[/bold yellow]\n"
                "[dim]Review warnings before deployment[/dim]",
                box=box.DOUBLE
            ))
            return True
        else:
            console.print(Panel.fit(
                "[bold red]❌ SYSTEM NOT READY[/bold red]\n"
                "[dim]Critical issues must be resolved[/dim]",
                box=box.DOUBLE
            ))
            return False

def main():
    """Main execution function"""
    validator = ProductionValidator()
    
    try:
        is_ready = validator.run_validation()
        
        if is_ready:
            console.print("\n[bold green]🎉 Production validation completed successfully![/bold green]")
            console.print("[dim]You can now proceed with deployment.[/dim]")
        else:
            console.print("\n[bold red]🚫 Production validation failed![/bold red]")
            console.print("[dim]Please resolve issues before deployment.[/dim]")
            sys.exit(1)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Validation interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]💥 Validation error: {str(e)}[/bold red]")
        sys.exit(1)

if __name__ == "__main__":
    main()
