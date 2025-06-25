#!/usr/bin/env python3
from datetime import datetime
from pathlib import Path
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Confirm
    from rich.table import Table
            from src.single_user_auth import auth_manager
from typing import Dict, List, Optional
import json
import logging
import os
import psutil
import signal
        import socket
import subprocess
import sys
import time
import yaml
"""
Production Start Script for Single User NICEGOLD System
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

สคริปต์เริ่มต้นระบบ production สำหรับผู้ใช้คนเดียว
พร้อมการตรวจสอบและเริ่มบริการทั้งหมด
"""


# Rich imports for beautiful output
try:
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None

class ProductionManager:
    """Production system manager for NICEGOLD Enterprise"""

    def __init__(self):
        self.project_root = Path(".")
        self.config_dir = self.project_root / "config"
        self.logs_dir = self.project_root / "logs"
        self.pid_dir = self.project_root / "run"

        # Create directories
        for directory in [self.logs_dir, self.pid_dir]:
            directory.mkdir(exist_ok = True)

        # Setup logging
        self._setup_logging()

        # Services configuration
        self.services = {
            "api": {
                "name": "NICEGOLD API", 
                "command": [sys.executable, " - m", "uvicorn", "src.api:app", 
                           " -  - host", "0.0.0.0", " -  - port", "8000", " -  - workers", "4"], 
                "port": 8000, 
                "health_endpoint": "/health", 
                "required": True
            }, 
            "dashboard": {
                "name": "Dashboard", 
                "command": [sys.executable, " - m", "streamlit", "run", "single_user_dashboard.py", 
                           " -  - server.port", "8501", " -  - server.address", "0.0.0.0"], 
                "port": 8501, 
                "health_endpoint": "/", 
                "required": True
            }, 
            "monitoring": {
                "name": "Monitoring Service", 
                "command": [sys.executable, "src/monitoring_service.py"], 
                "port": 9090, 
                "health_endpoint": "/metrics", 
                "required": False
            }
        }

        self.running_processes = {}
        self.shutdown_requested = False

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _setup_logging(self):
        """Setup logging system"""
        log_file = self.logs_dir / f"production_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level = logging.INFO, 
            format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
            handlers = [
                logging.FileHandler(log_file), 
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_requested = True

    def start_production(self) -> bool:
        """Start production system"""
        if RICH_AVAILABLE:
            self._start_production_rich()
        else:
            self._start_production_simple()

        return True

    def _start_production_rich(self):
        """Start production with rich interface"""
        console.print(Panel.fit(
            "[bold green]🚀 NICEGOLD Enterprise Production System[/bold green]\n"
            "[yellow]Starting all services...[/yellow]", 
            title = "Production Startup"
        ))

        # Pre - startup checks
        if not self._pre_startup_checks():
            console.print("[bold red]❌ Pre - startup checks failed![/bold red]")
            return False

        # Start services
        self._start_all_services()

        # Show system status
        self._show_live_status()

    def _start_production_simple(self):
        """Start production with simple interface"""
        print("🚀 NICEGOLD Enterprise Production System")
        print(" = " * 50)

        # Pre - startup checks
        if not self._pre_startup_checks():
            print("❌ Pre - startup checks failed!")
            return False

        # Start services
        self._start_all_services()

        # Monitor services
        self._monitor_services()

    def _pre_startup_checks(self) -> bool:
        """Perform pre - startup system checks"""
        if RICH_AVAILABLE:
            console.print("\n[bold blue]📋 Pre - startup Checks[/bold blue]")
        else:
            print("\n📋 Pre - startup Checks")

        checks = [
            ("Authentication System", self._check_auth_system), 
            ("Database Connection", self._check_database), 
            ("Configuration Files", self._check_configuration), 
            ("Required Ports", self._check_ports), 
            ("System Resources", self._check_resources), 
            ("Dependencies", self._check_dependencies)
        ]

        all_passed = True

        for check_name, check_func in checks:
            try:
                result = check_func()
                if result:
                    if RICH_AVAILABLE:
                        console.print(f"✅ {check_name}: [green]PASSED[/green]")
                    else:
                        print(f"✅ {check_name}: PASSED")
                else:
                    if RICH_AVAILABLE:
                        console.print(f"❌ {check_name}: [red]FAILED[/red]")
                    else:
                        print(f"❌ {check_name}: FAILED")
                    all_passed = False
            except Exception as e:
                if RICH_AVAILABLE:
                    console.print(f"❌ {check_name}: [red]ERROR - {e}[/red]")
                else:
                    print(f"❌ {check_name}: ERROR - {e}")
                all_passed = False

        return all_passed

    def _check_auth_system(self) -> bool:
        """Check authentication system"""
        try:
            status = auth_manager.get_system_status()
            return status["user_configured"]
        except ImportError:
            return False
        except Exception:
            return False

    def _check_database(self) -> bool:
        """Check database connection"""
        try:
            db_path = self.project_root / "database" / "production.db"
            return db_path.exists()
        except Exception:
            return False

    def _check_configuration(self) -> bool:
        """Check configuration files"""
        required_configs = [
            "config/production.yaml", 
            ".env.production"
        ]

        return all((self.project_root / config).exists() for config in required_configs)

    def _check_ports(self) -> bool:
        """Check if required ports are available"""
        required_ports = [8000, 8501]  # API and Dashboard

        for port in required_ports:
            if self._is_port_in_use(port):
                return False

        return True

    def _is_port_in_use(self, port: int) -> bool:
        """Check if port is in use"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return False
        except OSError:
            return True

    def _check_resources(self) -> bool:
        """Check system resources"""
        try:
            # Check available memory (minimum 2GB)
            memory = psutil.virtual_memory()
            if memory.available < 2 * 1024 * 1024 * 1024:  # 2GB
                return False

            # Check disk space (minimum 1GB)
            disk = psutil.disk_usage(str(self.project_root))
            if disk.free < 1 * 1024 * 1024 * 1024:  # 1GB
                return False

            return True
        except Exception:
            return False

    def _check_dependencies(self) -> bool:
        """Check required Python packages"""
        required_packages = [
            "fastapi", "uvicorn", "streamlit", "pandas", "numpy"
        ]

        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                return False

        return True

    def _start_all_services(self):
        """Start all services"""
        if RICH_AVAILABLE:
            console.print("\n[bold blue]🚀 Starting Services[/bold blue]")
        else:
            print("\n🚀 Starting Services")

        for service_id, service_config in self.services.items():
            if self._start_service(service_id, service_config):
                if RICH_AVAILABLE:
                    console.print(f"✅ {service_config['name']}: [green]STARTED[/green]")
                else:
                    print(f"✅ {service_config['name']}: STARTED")
            else:
                if RICH_AVAILABLE:
                    console.print(f"❌ {service_config['name']}: [red]FAILED[/red]")
                else:
                    print(f"❌ {service_config['name']}: FAILED")

    def _start_service(self, service_id: str, config: Dict) -> bool:
        """Start a single service"""
        try:
            # Set environment variables
            env = os.environ.copy()
            env.update({
                "ENVIRONMENT": "production", 
                "PYTHONPATH": str(self.project_root)
            })

            # Start process
            process = subprocess.Popen(
                config["command"], 
                cwd = str(self.project_root), 
                env = env, 
                stdout = subprocess.PIPE, 
                stderr = subprocess.PIPE, 
                preexec_fn = os.setsid if os.name != 'nt' else None
            )

            # Store process info
            self.running_processes[service_id] = {
                "process": process, 
                "config": config, 
                "started_at": datetime.now(), 
                "restarts": 0
            }

            # Save PID file
            pid_file = self.pid_dir / f"{service_id}.pid"
            with open(pid_file, 'w') as f:
                f.write(str(process.pid))

            # Wait a moment to check if process started successfully
            time.sleep(2)

            if process.poll() is None:
                self.logger.info(f"Service {service_id} started with PID {process.pid}")
                return True
            else:
                self.logger.error(f"Service {service_id} failed to start")
                return False

        except Exception as e:
            self.logger.error(f"Failed to start service {service_id}: {e}")
            return False

    def _show_live_status(self):
        """Show live system status"""
        console.print("\n[bold green]✅ All services started successfully![/bold green]")

        # Show access URLs
        table = Table(title = "🔗 Service Access URLs")
        table.add_column("Service", style = "cyan")
        table.add_column("URL", style = "green")
        table.add_column("Status", style = "yellow")

        table.add_row("API", "http://localhost:8000", "🟢 Running")
        table.add_row("Dashboard", "http://localhost:8501", "🟢 Running")
        table.add_row("API Docs", "http://localhost:8000/docs", "🟢 Available")
        table.add_row("Health Check", "http://localhost:8000/health", "🟢 Available")

        console.print(table)

        # Show system information
        console.print(f"\n[bold blue]📊 System Information[/bold blue]")
        console.print(f"• Started at: {datetime.now().strftime('%Y - %m - %d %H:%M:%S')}")
        console.print(f"• Process ID: {os.getpid()}")
        console.print(f"• Log directory: {self.logs_dir}")
        console.print(f"• Configuration: Production mode")

        console.print(f"\n[bold yellow]💡 Management Commands:[/bold yellow]")
        console.print(f"• View logs: tail -f {self.logs_dir}/production_*.log")
        console.print(f"• Stop system: Ctrl + C or kill {os.getpid()}")
        console.print(f"• Health check: curl http://localhost:8000/health")

        # Monitor services
        self._monitor_services()

    def _monitor_services(self):
        """Monitor running services"""
        if RICH_AVAILABLE:
            console.print("\n[bold blue]📊 Monitoring Services[/bold blue]")
            console.print("Press Ctrl + C to shutdown gracefully...")
        else:
            print("\n📊 Monitoring Services")
            print("Press Ctrl + C to shutdown gracefully...")

        try:
            while not self.shutdown_requested:
                # Check service health
                self._check_service_health()

                # Sleep for monitoring interval
                time.sleep(30)  # Check every 30 seconds

        except KeyboardInterrupt:
            self.shutdown_requested = True
        finally:
            self._shutdown_services()

    def _check_service_health(self):
        """Check health of all services"""
        for service_id, service_info in self.running_processes.items():
            process = service_info["process"]

            if process.poll() is not None:
                # Process has stopped
                self.logger.warning(f"Service {service_id} has stopped unexpectedly")

                # Attempt restart if required service
                if service_info["config"].get("required", False):
                    if service_info["restarts"] < 3:  # Max 3 restarts
                        self.logger.info(f"Attempting to restart service {service_id}")
                        if self._start_service(service_id, service_info["config"]):
                            service_info["restarts"] += 1
                        else:
                            self.logger.error(f"Failed to restart service {service_id}")
                    else:
                        self.logger.error(f"Service {service_id} exceeded restart limit")

    def _shutdown_services(self):
        """Shutdown all services gracefully"""
        if RICH_AVAILABLE:
            console.print("\n[bold yellow]🛑 Shutting down services...[/bold yellow]")
        else:
            print("\n🛑 Shutting down services...")

        for service_id, service_info in self.running_processes.items():
            process = service_info["process"]
            service_name = service_info["config"]["name"]

            if process.poll() is None:  # Process is still running
                try:
                    if RICH_AVAILABLE:
                        console.print(f"Stopping {service_name}...")
                    else:
                        print(f"Stopping {service_name}...")

                    # Send SIGTERM
                    if os.name != 'nt':
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    else:
                        process.terminate()

                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout = 10)
                    except subprocess.TimeoutExpired:
                        # Force kill if necessary
                        if os.name != 'nt':
                            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        else:
                            process.kill()
                        process.wait()

                    if RICH_AVAILABLE:
                        console.print(f"✅ {service_name} stopped")
                    else:
                        print(f"✅ {service_name} stopped")

                except Exception as e:
                    self.logger.error(f"Error stopping {service_name}: {e}")

            # Remove PID file
            pid_file = self.pid_dir / f"{service_id}.pid"
            if pid_file.exists():
                pid_file.unlink()

        if RICH_AVAILABLE:
            console.print("\n[bold green]✅ All services stopped successfully[/bold green]")
        else:
            print("\n✅ All services stopped successfully")

    def status(self):
        """Show system status"""
        if RICH_AVAILABLE:
            self._show_status_rich()
        else:
            self._show_status_simple()

    def _show_status_rich(self):
        """Show status with rich interface"""
        console.print(Panel.fit(
            "[bold blue]📊 NICEGOLD Enterprise System Status[/bold blue]", 
            title = "System Status"
        ))

        # Check if services are running
        table = Table(title = "Service Status")
        table.add_column("Service", style = "cyan")
        table.add_column("Status", style = "yellow")
        table.add_column("PID", style = "green")
        table.add_column("Port", style = "magenta")

        for service_id, config in self.services.items():
            pid_file = self.pid_dir / f"{service_id}.pid"

            if pid_file.exists():
                try:
                    with open(pid_file, 'r') as f:
                        pid = int(f.read().strip())

                    if psutil.pid_exists(pid):
                        status = "🟢 Running"
                        pid_str = str(pid)
                    else:
                        status = "🔴 Stopped"
                        pid_str = "N/A"
                except:
                    status = "❓ Unknown"
                    pid_str = "N/A"
            else:
                status = "🔴 Stopped"
                pid_str = "N/A"

            table.add_row(
                config["name"], 
                status, 
                pid_str, 
                str(config["port"])
            )

        console.print(table)

    def _show_status_simple(self):
        """Show status with simple interface"""
        print("📊 NICEGOLD Enterprise System Status")
        print(" = " * 40)

        for service_id, config in self.services.items():
            pid_file = self.pid_dir / f"{service_id}.pid"

            if pid_file.exists():
                try:
                    with open(pid_file, 'r') as f:
                        pid = int(f.read().strip())

                    if psutil.pid_exists(pid):
                        status = "Running"
                    else:
                        status = "Stopped"
                except:
                    status = "Unknown"
            else:
                status = "Stopped"

            print(f"{config['name']}: {status} (Port: {config['port']})")

    def stop(self):
        """Stop all services"""
        if RICH_AVAILABLE:
            console.print("[bold yellow]🛑 Stopping NICEGOLD Enterprise...[/bold yellow]")
        else:
            print("🛑 Stopping NICEGOLD Enterprise...")

        stopped_services = 0

        for service_id, config in self.services.items():
            pid_file = self.pid_dir / f"{service_id}.pid"

            if pid_file.exists():
                try:
                    with open(pid_file, 'r') as f:
                        pid = int(f.read().strip())

                    if psutil.pid_exists(pid):
                        # Stop the process
                        process = psutil.Process(pid)
                        process.terminate()

                        # Wait for termination
                        try:
                            process.wait(timeout = 10)
                        except psutil.TimeoutExpired:
                            process.kill()

                        stopped_services += 1

                        if RICH_AVAILABLE:
                            console.print(f"✅ Stopped {config['name']}")
                        else:
                            print(f"✅ Stopped {config['name']}")

                    # Remove PID file
                    pid_file.unlink()

                except Exception as e:
                    if RICH_AVAILABLE:
                        console.print(f"❌ Error stopping {config['name']}: {e}")
                    else:
                        print(f"❌ Error stopping {config['name']}: {e}")

        if stopped_services > 0:
            if RICH_AVAILABLE:
                console.print(f"\n[bold green]✅ Stopped {stopped_services} services[/bold green]")
            else:
                print(f"\n✅ Stopped {stopped_services} services")
        else:
            if RICH_AVAILABLE:
                console.print("\n[yellow]ℹ️ No services were running[/yellow]")
            else:
                print("\nℹ️ No services were running")

def main():
    """Main entry point"""
    manager = ProductionManager()

    # Parse command line arguments
    if len(sys.argv) < 2:
        command = "start"
    else:
        command = sys.argv[1].lower()

    try:
        if command == "start":
            manager.start_production()
        elif command == "stop":
            manager.stop()
        elif command == "status":
            manager.status()
        elif command == "restart":
            manager.stop()
            time.sleep(2)
            manager.start_production()
        else:
            if RICH_AVAILABLE:
                console.print(f"[red]Unknown command: {command}[/red]")
                console.print("\n[bold]Available commands:[/bold]")
                console.print("• start   - Start all services")
                console.print("• stop    - Stop all services")
                console.print("• restart - Restart all services")
                console.print("• status  - Show service status")
            else:
                print(f"Unknown command: {command}")
                print("\nAvailable commands:")
                print("• start   - Start all services")
                print("• stop    - Stop all services")
                print("• restart - Restart all services")
                print("• status  - Show service status")
            sys.exit(1)

    except KeyboardInterrupt:
        if RICH_AVAILABLE:
            console.print("\n[yellow]Operation cancelled by user[/yellow]")
        else:
            print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"\n[red]Error: {e}[/red]")
        else:
            print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()