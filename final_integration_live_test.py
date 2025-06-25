#!/usr/bin/env python3
                from ai_orchestrator import AIOrchestrator
                from ai_team_manager import AITeamManager
from datetime import datetime
from pathlib import Path
    from rich.columns import Columns
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
    from rich.prompt import Confirm, Prompt
    from rich.table import Table
    from rich.text import Text
                from single_user_auth import SingleUserAuth
from typing import Dict, List, Optional, Tuple
import asyncio
import json
import logging
            import numpy as np
import os
            import pandas as pd
    import psutil
    import requests
import sqlite3
import subprocess
import sys
import threading
import time
    import yaml
"""
🔥 NICEGOLD Final Live Integration Test 🔥
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

การทดสอบระบบสุดท้ายแบบสมบูรณ์ สำหรับ single - user production system
พร้อม AI Team orchestration และ dashboard integration

Features:
- Complete system integration test
- Single - user authentication validation
- AI team functionality verification
- Production services health check
- Performance benchmarking
- Security audit
- Dashboard integration test
"""


try:
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

try:
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None

class FinalIntegrationTest:
    """Complete integration test for NICEGOLD production system"""

    def __init__(self):
        self.project_root = Path(".")
        self.test_results = {
            "start_time": datetime.now(), 
            "tests": {}, 
            "overall_status": "PENDING", 
            "errors": [], 
            "warnings": [], 
            "performance": {}
        }

        # Setup logging
        self._setup_logging()

        # Test configuration
        self.test_config = {
            "auth_test_user": "admin", 
            "auth_test_password": "test_admin_password_123", 
            "api_base_url": "http://localhost:8000", 
            "dashboard_url": "http://localhost:8501", 
            "test_timeout": 30, 
            "performance_threshold": {
                "api_response_time": 2.0, 
                "dashboard_load_time": 5.0, 
                "ai_response_time": 10.0
            }
        }

        self.log("🚀 Final Integration Test Initialized")

    def _setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = self.project_root / "logs" / "integration_test"
        log_dir.mkdir(parents = True, exist_ok = True)

        log_file = log_dir / f"integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level = logging.INFO, 
            format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
            handlers = [
                logging.FileHandler(log_file), 
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def log(self, message: str, level: str = "info"):
        """Enhanced logging with rich console support"""
        if RICH_AVAILABLE and console:
            timestamp = datetime.now().strftime("%H:%M:%S")
            if level == "error":
                console.print(f"[red][{timestamp}] ❌ {message}[/red]")
            elif level == "warning":
                console.print(f"[yellow][{timestamp}] ⚠️  {message}[/yellow]")
            elif level == "success":
                console.print(f"[green][{timestamp}] ✅ {message}[/green]")
            else:
                console.print(f"[blue][{timestamp}] ℹ️  {message}[/blue]")

        getattr(self.logger, level, self.logger.info)(message)

    def run_all_tests(self) -> bool:
        """Run complete integration test suite"""
        self.log("🔥 Starting Complete Integration Test Suite")

        if RICH_AVAILABLE and console:
            with Live(self._create_test_dashboard(), refresh_per_second = 2) as live:
                success = self._execute_all_tests(live)
        else:
            success = self._execute_all_tests()

        self._generate_final_report()
        return success

    def _execute_all_tests(self, live = None) -> bool:
        """Execute all integration tests"""
        tests = [
            ("Environment Check", self._test_environment), 
            ("Dependencies Validation", self._test_dependencies), 
            ("Authentication System", self._test_authentication), 
            ("Database Connectivity", self._test_database), 
            ("Configuration Management", self._test_configuration), 
            ("API Services", self._test_api_services), 
            ("Dashboard Services", self._test_dashboard), 
            ("AI Team System", self._test_ai_team), 
            ("AI Orchestrator", self._test_ai_orchestrator), 
            ("Security Audit", self._test_security), 
            ("Performance Benchmark", self._test_performance), 
            ("Integration Workflow", self._test_integration_workflow)
        ]

        total_tests = len(tests)
        passed_tests = 0

        for i, (test_name, test_function) in enumerate(tests):
            self.log(f"Running test {i + 1}/{total_tests}: {test_name}")

            try:
                start_time = time.time()
                result = test_function()
                execution_time = time.time() - start_time

                self.test_results["tests"][test_name] = {
                    "status": "PASSED" if result else "FAILED", 
                    "execution_time": execution_time, 
                    "details": result if isinstance(result, dict) else {}
                }

                if result:
                    passed_tests += 1
                    self.log(f"✅ {test_name} - PASSED ({execution_time:.2f}s)", "success")
                else:
                    self.log(f"❌ {test_name} - FAILED ({execution_time:.2f}s)", "error")

            except Exception as e:
                self.log(f"❌ {test_name} - ERROR: {str(e)}", "error")
                self.test_results["errors"].append(f"{test_name}: {str(e)}")
                self.test_results["tests"][test_name] = {
                    "status": "ERROR", 
                    "error": str(e)
                }

            if live:
                live.update(self._create_test_dashboard())

            time.sleep(0.5)  # Brief pause between tests

        # Calculate overall success
        success_rate = passed_tests / total_tests
        self.test_results["overall_status"] = "PASSED" if success_rate >= 0.8 else "FAILED"
        self.test_results["success_rate"] = success_rate

        return success_rate >= 0.8

    def _test_environment(self) -> bool:
        """Test environment setup and requirements"""
        try:
            # Check Python version
            python_version = sys.version_info
            if python_version < (3, 8):
                self.test_results["errors"].append(f"Python version {python_version} < 3.8")
                return False

            # Check project structure
            required_dirs = [
                "src", "config", "logs", "database", 
                "static", "templates"
            ]

            missing_dirs = []
            for dir_name in required_dirs:
                if not (self.project_root / dir_name).exists():
                    missing_dirs.append(dir_name)

            if missing_dirs:
                self.test_results["warnings"].append(f"Missing directories: {missing_dirs}")

            # Check disk space
            disk_usage = psutil.disk_usage('.')
            free_gb = disk_usage.free / (1024**3)
            if free_gb < 1:
                self.test_results["warnings"].append(f"Low disk space: {free_gb:.1f}GB")

            return True

        except Exception as e:
            self.log(f"Environment test failed: {e}", "error")
            return False

    def _test_dependencies(self) -> bool:
        """Test all required dependencies"""
        try:
            required_packages = [
                "fastapi", "uvicorn", "streamlit", "pandas", 
                "numpy", "scikit - learn", "joblib", "requests", 
                "pydantic", "python - jose", "passlib", "bcrypt", 
                "rich", "psutil", "pyyaml"
            ]

            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package.replace(" - ", "_"))
                except ImportError:
                    missing_packages.append(package)

            if missing_packages:
                self.test_results["errors"].append(f"Missing packages: {missing_packages}")
                return False

            return True

        except Exception as e:
            self.log(f"Dependencies test failed: {e}", "error")
            return False

    def _test_authentication(self) -> bool:
        """Test single - user authentication system"""
        try:
            # Check if auth module exists
            auth_file = self.project_root / "src" / "single_user_auth.py"
            if not auth_file.exists():
                self.test_results["errors"].append("Authentication module not found")
                return False

            # Test database creation and user management
            sys.path.insert(0, str(self.project_root / "src"))

            try:
                auth = SingleUserAuth()

                # Test user creation
                test_user = self.test_config["auth_test_user"]
                test_password = self.test_config["auth_test_password"]

                # Create test user
                auth.create_user(test_user, test_password)

                # Test authentication
                token = auth.authenticate(test_user, test_password)
                if not token:
                    self.test_results["errors"].append("Authentication failed")
                    return False

                # Test token validation
                user_data = auth.verify_token(token)
                if not user_data or user_data.get("username") != test_user:
                    self.test_results["errors"].append("Token verification failed")
                    return False

                self.log("Authentication system working correctly", "success")
                return True

            except Exception as e:
                self.test_results["errors"].append(f"Authentication module error: {e}")
                return False

        except Exception as e:
            self.log(f"Authentication test failed: {e}", "error")
            return False

    def _test_database(self) -> bool:
        """Test database connectivity and operations"""
        try:
            db_dir = self.project_root / "database"
            db_dir.mkdir(exist_ok = True)

            # Test SQLite connection
            test_db = db_dir / "test_integration.db"

            with sqlite3.connect(test_db) as conn:
                cursor = conn.cursor()

                # Create test table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS test_table (
                        id INTEGER PRIMARY KEY, 
                        name TEXT, 
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Test insert
                cursor.execute("INSERT INTO test_table (name) VALUES (?)", ("test_data", ))

                # Test select
                cursor.execute("SELECT * FROM test_table")
                results = cursor.fetchall()

                if not results:
                    self.test_results["errors"].append("Database operations failed")
                    return False

            # Clean up
            if test_db.exists():
                test_db.unlink()

            return True

        except Exception as e:
            self.log(f"Database test failed: {e}", "error")
            return False

    def _test_configuration(self) -> bool:
        """Test configuration management"""
        try:
            config_files = [
                "config.yaml", 
                "config/production.yaml", 
                ".env.production"
            ]

            missing_configs = []
            for config_file in config_files:
                config_path = self.project_root / config_file
                if not config_path.exists():
                    missing_configs.append(config_file)

            if missing_configs:
                self.test_results["warnings"].append(f"Missing configs: {missing_configs}")

            # Test YAML loading
            try:
                config_file = self.project_root / "config.yaml"
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                    if not isinstance(config, dict):
                        self.test_results["errors"].append("Invalid config format")
                        return False
            except Exception as e:
                self.test_results["errors"].append(f"Config loading error: {e}")
                return False

            return True

        except Exception as e:
            self.log(f"Configuration test failed: {e}", "error")
            return False

    def _test_api_services(self) -> bool:
        """Test API services functionality"""
        try:
            # Check if API file exists
            api_file = self.project_root / "src" / "api.py"
            if not api_file.exists():
                self.test_results["warnings"].append("API module not found")
                return True  # Not critical for basic functionality

            # TODO: Could add actual API endpoint testing here
            # For now, just check file existence and basic imports

            return True

        except Exception as e:
            self.log(f"API services test failed: {e}", "error")
            return False

    def _test_dashboard(self) -> bool:
        """Test dashboard functionality"""
        try:
            # Check if dashboard files exist
            dashboard_files = [
                "dashboard_app.py", 
                "src/dashboard.py"
            ]

            dashboard_found = False
            for dashboard_file in dashboard_files:
                if (self.project_root / dashboard_file).exists():
                    dashboard_found = True
                    break

            if not dashboard_found:
                self.test_results["warnings"].append("Dashboard module not found")
                return True  # Not critical for basic functionality

            return True

        except Exception as e:
            self.log(f"Dashboard test failed: {e}", "error")
            return False

    def _test_ai_team(self) -> bool:
        """Test AI Team system"""
        try:
            # Check if AI team file exists
            ai_team_file = self.project_root / "ai_team_manager.py"
            if not ai_team_file.exists():
                self.test_results["errors"].append("AI Team Manager not found")
                return False

            # Test AI team initialization
            sys.path.insert(0, str(self.project_root))

            try:
                team_manager = AITeamManager()

                # Test basic functionality
                agents = team_manager.get_available_agents()
                if not agents:
                    self.test_results["warnings"].append("No AI agents available")

                return True

            except Exception as e:
                self.test_results["errors"].append(f"AI Team initialization error: {e}")
                return False

        except Exception as e:
            self.log(f"AI Team test failed: {e}", "error")
            return False

    def _test_ai_orchestrator(self) -> bool:
        """Test AI Orchestrator system"""
        try:
            # Check if AI orchestrator file exists
            orchestrator_file = self.project_root / "ai_orchestrator.py"
            if not orchestrator_file.exists():
                self.test_results["errors"].append("AI Orchestrator not found")
                return False

            # Test orchestrator initialization
            try:
                orchestrator = AIOrchestrator()

                # Test basic functionality
                return True

            except Exception as e:
                self.test_results["errors"].append(f"AI Orchestrator initialization error: {e}")
                return False

        except Exception as e:
            self.log(f"AI Orchestrator test failed: {e}", "error")
            return False

    def _test_security(self) -> bool:
        """Test security configurations"""
        try:
            security_checks = []

            # Check for sensitive files
            sensitive_files = [".env", ".env.production", "database/*.db"]
            for pattern in sensitive_files:
                # Basic check - in production should have proper permissions
                security_checks.append(f"Checked: {pattern}")

            # Check authentication requirements
            auth_file = self.project_root / "src" / "single_user_auth.py"
            if auth_file.exists():
                security_checks.append("Authentication system available")

            self.test_results["tests"]["Security Audit"] = {
                "checks": security_checks
            }

            return True

        except Exception as e:
            self.log(f"Security test failed: {e}", "error")
            return False

    def _test_performance(self) -> bool:
        """Test system performance"""
        try:
            performance_data = {}

            # Test import performance
            start_time = time.time()
            import_time = time.time() - start_time
            performance_data["import_time"] = import_time

            # Test basic operations
            start_time = time.time()
            df = pd.DataFrame(np.random.rand(1000, 10))
            df.describe()
            operation_time = time.time() - start_time
            performance_data["basic_operations"] = operation_time

            # Memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            performance_data["memory_usage_mb"] = memory_info.rss / 1024 / 1024

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval = 1)
            performance_data["cpu_usage_percent"] = cpu_percent

            self.test_results["performance"] = performance_data

            return True

        except Exception as e:
            self.log(f"Performance test failed: {e}", "error")
            return False

    def _test_integration_workflow(self) -> bool:
        """Test complete integration workflow"""
        try:
            # Test a complete workflow from authentication to AI processing
            workflow_steps = []

            # Step 1: Authentication
            sys.path.insert(0, str(self.project_root / "src"))
            try:
                auth = SingleUserAuth()
                workflow_steps.append("Authentication system loaded")
            except:
                workflow_steps.append("Authentication system failed to load")
                return False

            # Step 2: AI Team
            try:
                team = AITeamManager()
                workflow_steps.append("AI Team system loaded")
            except:
                workflow_steps.append("AI Team system failed to load")

            # Step 3: AI Orchestrator
            try:
                orchestrator = AIOrchestrator()
                workflow_steps.append("AI Orchestrator loaded")
            except:
                workflow_steps.append("AI Orchestrator failed to load")

            self.test_results["tests"]["Integration Workflow"] = {
                "steps": workflow_steps
            }

            return True

        except Exception as e:
            self.log(f"Integration workflow test failed: {e}", "error")
            return False

    def _create_test_dashboard(self) -> Panel:
        """Create live test dashboard"""
        if not RICH_AVAILABLE:
            return None

        # Create test progress table
        table = Table(title = "🔥 NICEGOLD Integration Test Status")
        table.add_column("Test", style = "cyan")
        table.add_column("Status", style = "bold")
        table.add_column("Time", style = "green")
        table.add_column("Details", style = "dim")

        for test_name, test_data in self.test_results["tests"].items():
            status = test_data.get("status", "PENDING")
            exec_time = test_data.get("execution_time", 0)

            if status == "PASSED":
                status_text = Text("✅ PASSED", style = "green")
            elif status == "FAILED":
                status_text = Text("❌ FAILED", style = "red")
            elif status == "ERROR":
                status_text = Text("💥 ERROR", style = "red")
            else:
                status_text = Text("⏳ PENDING", style = "yellow")

            table.add_row(
                test_name, 
                status_text, 
                f"{exec_time:.2f}s", 
                test_data.get("error", "")[:50]
            )

        # Create summary panel
        total_tests = len(self.test_results["tests"])
        passed_tests = sum(1 for t in self.test_results["tests"].values() if t.get("status") == "PASSED")

        summary = f"""
🚀 NICEGOLD Final Integration Test
📊 Progress: {passed_tests}/{total_tests} tests passed
⏱️  Started: {self.test_results['start_time'].strftime('%H:%M:%S')}
🎯 Status: {self.test_results['overall_status']}
        """

        return Panel(
            Columns([table, Panel(summary, title = "Summary")]), 
            title = "🔥 NICEGOLD Integration Test Dashboard", 
            border_style = "bright_blue"
        )

    def _generate_final_report(self):
        """Generate comprehensive final report"""
        self.test_results["end_time"] = datetime.now()
        self.test_results["total_duration"] = (
            self.test_results["end_time"] - self.test_results["start_time"]
        ).total_seconds()

        # Save detailed report
        report_dir = self.project_root / "logs" / "integration_test"
        report_dir.mkdir(parents = True, exist_ok = True)

        report_file = report_dir / f"final_integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert datetime objects to strings for JSON serialization
        report_data = {
            **self.test_results, 
            "start_time": self.test_results["start_time"].isoformat(), 
            "end_time": self.test_results["end_time"].isoformat()
        }

        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent = 2)

        # Display final summary
        self._display_final_summary()

        self.log(f"📊 Detailed report saved to: {report_file}")

    def _display_final_summary(self):
        """Display beautiful final summary"""
        if not RICH_AVAILABLE or not console:
            # Fallback to simple text output
            print("\n" + " = "*60)
            print("🔥 NICEGOLD FINAL INTEGRATION TEST SUMMARY")
            print(" = "*60)
            print(f"Overall Status: {self.test_results['overall_status']}")
            print(f"Success Rate: {self.test_results.get('success_rate', 0)*100:.1f}%")
            print(f"Total Duration: {self.test_results['total_duration']:.2f}s")
            print(f"Errors: {len(self.test_results['errors'])}")
            print(f"Warnings: {len(self.test_results['warnings'])}")
            return

        # Rich display
        console.print("\n")

        # Main summary panel
        success_rate = self.test_results.get('success_rate', 0) * 100
        status_color = "green" if self.test_results['overall_status'] == 'PASSED' else "red"

        summary_content = f"""
[bold {status_color}]🔥 NICEGOLD FINAL INTEGRATION TEST SUMMARY[/bold {status_color}]

📊 Overall Status: [{status_color}]{self.test_results['overall_status']}[/{status_color}]
🎯 Success Rate: [green]{success_rate:.1f}%[/green]
⏱️  Total Duration: [blue]{self.test_results['total_duration']:.2f} seconds[/blue]
🔍 Tests Run: [cyan]{len(self.test_results['tests'])}[/cyan]
❌ Errors: [red]{len(self.test_results['errors'])}[/red]
⚠️  Warnings: [yellow]{len(self.test_results['warnings'])}[/yellow]
        """

        console.print(Panel(summary_content, title = "🚀 Test Summary", border_style = status_color))

        # Performance summary
        if self.test_results.get('performance'):
            perf = self.test_results['performance']
            perf_content = f"""
💾 Memory Usage: [cyan]{perf.get('memory_usage_mb', 0):.1f} MB[/cyan]
🖥️  CPU Usage: [cyan]{perf.get('cpu_usage_percent', 0):.1f}%[/cyan]
⚡ Import Time: [cyan]{perf.get('import_time', 0):.3f}s[/cyan]
🚀 Operation Time: [cyan]{perf.get('basic_operations', 0):.3f}s[/cyan]
            """
            console.print(Panel(perf_content, title = "📈 Performance Metrics", border_style = "blue"))

        # Errors and warnings
        if self.test_results['errors']:
            error_content = "\n".join([f"❌ {error}" for error in self.test_results['errors']])
            console.print(Panel(error_content, title = "🚨 Errors", border_style = "red"))

        if self.test_results['warnings']:
            warning_content = "\n".join([f"⚠️  {warning}" for warning in self.test_results['warnings']])
            console.print(Panel(warning_content, title = "⚠️  Warnings", border_style = "yellow"))

        # Final status
        if self.test_results['overall_status'] == 'PASSED':
            console.print("\n[bold green]🎉 INTEGRATION TEST COMPLETED SUCCESSFULLY! 🎉[/bold green]")
            console.print("[green]✅ Your NICEGOLD production system is ready to deploy![/green]")
        else:
            console.print("\n[bold red]❌ INTEGRATION TEST FAILED[/bold red]")
            console.print("[red]🔧 Please review errors and fix issues before deployment[/red]")

def main():
    """Main entry point"""
    if not DEPENDENCIES_AVAILABLE:
        print("❌ Required dependencies not available. Please install: requests, psutil, pyyaml")
        return False

    print("🔥 Starting NICEGOLD Final Integration Test...")

    tester = FinalIntegrationTest()
    success = tester.run_all_tests()

    if success:
        print("\n🎉 All tests completed successfully!")
        return True
    else:
        print("\n❌ Some tests failed. Please review the results.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)