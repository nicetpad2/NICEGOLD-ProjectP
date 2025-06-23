#!/usr/bin/env python3
"""
🚀 NICEGOLD ENTERPRISE - SIMPLE PRODUCTION VALIDATOR
==================================================

Simple production readiness validation script without external dependencies.

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
import sys
from datetime import datetime
from pathlib import Path


class SimpleProductionValidator:
    """Simple production validation system"""
    
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
        
        # Setup basic logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def print_status(self, message, status="INFO"):
        """Print status message"""
        symbols = {"INFO": "ℹ️", "PASS": "✅", "FAIL": "❌", "WARN": "⚠️"}
        symbol = symbols.get(status, "ℹ️")
        print(f"{symbol} {message}")

    def validate_environment(self) -> bool:
        """Validate Python environment and dependencies"""
        self.print_status("Validating Environment...", "INFO")
        
        try:
            # Check Python version - more flexible for various environments
            python_version = sys.version_info
            if python_version < (3, 8):
                self.results["critical_issues"].append(
                    f"Python version {python_version.major}.{python_version.minor} is too old. Use Python 3.8+"
                )
                self.print_status("Python version check failed", "FAIL")
                return False
            elif python_version >= (3, 12):
                self.results["warnings"].append(
                    f"Python version {python_version.major}.{python_version.minor} is newer than tested versions. Some packages may have compatibility issues."
                )
                self.print_status(f"Python {python_version.major}.{python_version.minor} - newer than tested versions", "WARN")
            else:
                self.print_status(f"Python {python_version.major}.{python_version.minor} - compatible", "PASS")
            
            # Check critical dependencies
            critical_deps = [
                'pandas', 'numpy', 'scikit-learn', 'yaml'
            ]
            
            # Optional but recommended dependencies
            optional_deps = [
                'fastapi', 'streamlit', 'pydantic', 'uvicorn', 'sqlalchemy'
            ]
            
            missing_critical = []
            missing_optional = []
            
            for dep in critical_deps:
                if importlib.util.find_spec(dep) is None:
                    missing_critical.append(dep)
            
            for dep in optional_deps:
                if importlib.util.find_spec(dep) is None:
                    missing_optional.append(dep)
            
            if missing_critical:
                self.results["critical_issues"].append(
                    f"Missing critical dependencies: {', '.join(missing_critical)}"
                )
                self.print_status(f"Missing critical dependencies: {', '.join(missing_critical)}", "FAIL")
                return False
            
            if missing_optional:
                self.results["warnings"].append(
                    f"Missing optional dependencies: {', '.join(missing_optional)} (can be installed as needed)"
                )
                self.print_status(f"Missing optional dependencies: {', '.join(missing_optional)}", "WARN")
            
            self.results["validation_results"]["environment"] = "PASS"
            self.print_status("Environment validation passed", "PASS")
            return True
            
        except Exception as e:
            self.results["critical_issues"].append(f"Environment validation error: {str(e)}")
            self.print_status(f"Environment validation error: {str(e)}", "FAIL")
            return False

    def validate_configuration(self) -> bool:
        """Validate all configuration files"""
        self.print_status("Validating Configuration...", "INFO")
        
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
                self.print_status(f"Missing config files: {', '.join(missing_configs)}", "FAIL")
                return False
            
            # Basic check for .env.production content
            env_path = self.base_dir / ".env.production"
            with open(env_path, 'r') as f:
                env_content = f.read()
            
            if "DEBUG=true" in env_content.lower():
                self.results["critical_issues"].append(
                    "DEBUG mode is enabled in production environment"
                )
                self.print_status("DEBUG mode enabled in production", "FAIL")
                return False
            
            self.results["validation_results"]["configuration"] = "PASS"
            self.print_status("Configuration validation passed", "PASS")
            return True
            
        except Exception as e:
            self.results["critical_issues"].append(f"Configuration validation error: {str(e)}")
            self.print_status(f"Configuration validation error: {str(e)}", "FAIL")
            return False

    def validate_database(self) -> bool:
        """Validate database setup and connectivity"""
        self.print_status("Validating Database...", "INFO")
        
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
                self.print_status(f"Database connectivity error: {str(e)}", "FAIL")
                return False
            
            self.results["validation_results"]["database"] = "PASS"
            self.print_status("Database validation passed", "PASS")
            return True
            
        except Exception as e:
            self.results["critical_issues"].append(f"Database validation error: {str(e)}")
            self.print_status(f"Database validation error: {str(e)}", "FAIL")
            return False

    def validate_scripts(self) -> bool:
        """Validate critical production scripts"""
        self.print_status("Validating Production Scripts...", "INFO")
        
        try:
            critical_scripts = [
                "production_setup.py",
                "one_click_deploy.py",
                "system_maintenance.py",
                "ai_orchestrator.py",
                "start_production_single_user.py"
            ]
            
            missing_scripts = []
            syntax_errors = []
            
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
                        syntax_errors.append(f"{script}: {str(e)}")
            
            if missing_scripts:
                self.results["critical_issues"].append(
                    f"Missing critical scripts: {', '.join(missing_scripts)}"
                )
                self.print_status(f"Missing scripts: {', '.join(missing_scripts)}", "FAIL")
                return False
            
            if syntax_errors:
                self.results["critical_issues"].append(
                    f"Syntax errors in scripts: {'; '.join(syntax_errors)}"
                )
                self.print_status(f"Syntax errors found", "FAIL")
                return False
            
            self.results["validation_results"]["scripts"] = "PASS"
            self.print_status("Production scripts validation passed", "PASS")
            return True
            
        except Exception as e:
            self.results["critical_issues"].append(f"Scripts validation error: {str(e)}")
            self.print_status(f"Scripts validation error: {str(e)}", "FAIL")
            return False

    def validate_permissions(self) -> bool:
        """Validate file and directory permissions"""
        self.print_status("Validating Permissions...", "INFO")
        
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
                    self.print_status(f"No write permission for {dir_name}", "FAIL")
                    return False
            
            self.results["validation_results"]["permissions"] = "PASS"
            self.print_status("Permissions validation passed", "PASS")
            return True
            
        except Exception as e:
            self.results["critical_issues"].append(f"Permissions validation error: {str(e)}")
            self.print_status(f"Permissions validation error: {str(e)}", "FAIL")
            return False

    def create_production_report(self) -> None:
        """Create comprehensive production readiness report"""
        self.print_status("Generating Production Report...", "INFO")
        
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
        
        self.print_status(f"Report saved to: {report_path}", "PASS")
        self.print_status(f"JSON results saved to: {json_path}", "PASS")

    def run_validation(self) -> bool:
        """Run complete production validation"""
        print("=" * 60)
        print("🚀 NICEGOLD ENTERPRISE PRODUCTION VALIDATOR")
        print("Validating system readiness for production deployment...")
        print("=" * 60)
        
        validation_steps = [
            ("Environment", self.validate_environment),
            ("Configuration", self.validate_configuration),
            ("Database", self.validate_database),
            ("Scripts", self.validate_scripts),
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
                self.print_status(f"Validation step '{step_name}' failed: {str(e)}", "FAIL")
        
        # Create report
        self.create_production_report()
        
        print("\n" + "=" * 60)
        # Display final status
        if self.results["overall_status"] == "READY":
            print("✅ SYSTEM READY FOR PRODUCTION")
            print("All validations passed successfully!")
            return True
        elif self.results["overall_status"] == "WARNING":
            print("⚠️ SYSTEM READY WITH WARNINGS")
            print("Review warnings before deployment")
            return True
        else:
            print("❌ SYSTEM NOT READY")
            print("Critical issues must be resolved")
            return False

def main():
    """Main execution function"""
    validator = SimpleProductionValidator()
    
    try:
        is_ready = validator.run_validation()
        
        if is_ready:
            print("\n🎉 Production validation completed successfully!")
            print("You can now proceed with deployment.")
        else:
            print("\n🚫 Production validation failed!")
            print("Please resolve issues before deployment.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Validation error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
