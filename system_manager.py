#!/usr/bin/env python3
"""
🔧 NICEGOLD Enterprise System Manager 🔧
=======================================

ระบบจัดการและควบคุมเซอร์วิสของ NICEGOLD Enterprise
พร้อมการตรวจสอบสุขภาพระบบและการจัดการผู้ใช้
"""

import json
import logging
import os
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import yaml

# Rich imports for beautiful output
try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Confirm, Prompt
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None

class SystemManager:
    """NICEGOLD Enterprise System Manager"""
    
    def __init__(self):
        self.project_root = Path(".").resolve()
        self.config_dir = self.project_root / "config"
        self.logs_dir = self.project_root / "logs"
        self.run_dir = self.project_root / "run"
        self.database_dir = self.project_root / "database"
        
        # Ensure directories exist
        for directory in [self.logs_dir, self.run_dir]:
            directory.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Service definitions
        self.services = {
            "api": {
                "name": "NICEGOLD API Server",
                "command": ["python", "-m", "uvicorn", "src.api:app", "--host", "127.0.0.1", "--port", "8000"],
                "pid_file": "api.pid",
                "log_file": "api.log",
                "health_url": "http://127.0.0.1:8000/health"
            },
            "dashboard": {
                "name": "NICEGOLD Dashboard",
                "command": ["streamlit", "run", "single_user_dashboard.py", "--server.address", "127.0.0.1", "--server.port", "8501"],
                "pid_file": "dashboard.pid",
                "log_file": "dashboard.log",
                "health_url": "http://127.0.0.1:8501"
            }
        }
        
    def _setup_logging(self):
        """Setup logging system"""
        log_file = self.logs_dir / "system" / f"manager_{datetime.now().strftime('%Y%m%d')}.log"
        log_file.parent.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def start_all_services(self) -> bool:
        """Start all NICEGOLD services"""
        if RICH_AVAILABLE:
            console.print(Panel.fit(
                "[bold green]🚀 Starting NICEGOLD Enterprise Services[/bold green]",
                title="System Startup"
            ))
        else:
            print("🚀 Starting NICEGOLD Enterprise Services")
        
        success = True
        
        for service_id, service_config in self.services.items():
            if self.start_service(service_id):
                if RICH_AVAILABLE:
                    console.print(f"[green]✅ {service_config['name']} started successfully[/green]")
                else:
                    print(f"✅ {service_config['name']} started successfully")
            else:
                if RICH_AVAILABLE:
                    console.print(f"[red]❌ Failed to start {service_config['name']}[/red]")
                else:
                    print(f"❌ Failed to start {service_config['name']}")
                success = False
        
        if success:
            self._show_service_urls()
        
        return success
    
    def start_service(self, service_id: str) -> bool:
        """Start a specific service"""
        if service_id not in self.services:
            self.logger.error(f"Unknown service: {service_id}")
            return False
        
        service_config = self.services[service_id]
        
        # Check if already running
        if self.is_service_running(service_id):
            self.logger.warning(f"Service {service_id} is already running")
            return True
        
        try:
            # Prepare log file
            log_file = self.logs_dir / "application" / service_config["log_file"]
            log_file.parent.mkdir(exist_ok=True)
            
            # Start service
            with open(log_file, 'a') as log:
                process = subprocess.Popen(
                    service_config["command"],
                    stdout=log,
                    stderr=log,
                    cwd=self.project_root
                )
            
            # Save PID
            pid_file = self.run_dir / service_config["pid_file"]
            with open(pid_file, 'w') as f:
                f.write(str(process.pid))
            
            # Wait a moment for service to start
            time.sleep(2)
            
            # Verify service is running
            if self.is_service_running(service_id):
                self.logger.info(f"Service {service_id} started successfully (PID: {process.pid})")
                return True
            else:
                self.logger.error(f"Service {service_id} failed to start")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start service {service_id}: {e}")
            return False
    
    def stop_all_services(self) -> bool:
        """Stop all NICEGOLD services"""
        if RICH_AVAILABLE:
            console.print(Panel.fit(
                "[bold red]🛑 Stopping NICEGOLD Enterprise Services[/bold red]",
                title="System Shutdown"
            ))
        else:
            print("🛑 Stopping NICEGOLD Enterprise Services")
        
        success = True
        
        for service_id, service_config in self.services.items():
            if self.stop_service(service_id):
                if RICH_AVAILABLE:
                    console.print(f"[green]✅ {service_config['name']} stopped successfully[/green]")
                else:
                    print(f"✅ {service_config['name']} stopped successfully")
            else:
                if RICH_AVAILABLE:
                    console.print(f"[yellow]⚠️ {service_config['name']} was not running[/yellow]")
                else:
                    print(f"⚠️ {service_config['name']} was not running")
        
        return success
    
    def stop_service(self, service_id: str) -> bool:
        """Stop a specific service"""
        if service_id not in self.services:
            self.logger.error(f"Unknown service: {service_id}")
            return False
        
        service_config = self.services[service_id]
        pid_file = self.run_dir / service_config["pid_file"]
        
        if not pid_file.exists():
            self.logger.warning(f"PID file not found for service {service_id}")
            return False
        
        try:
            # Read PID
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # Check if process exists
            if psutil.pid_exists(pid):
                # Graceful shutdown
                os.kill(pid, signal.SIGTERM)
                
                # Wait for graceful shutdown
                for _ in range(10):
                    if not psutil.pid_exists(pid):
                        break
                    time.sleep(1)
                
                # Force kill if still running
                if psutil.pid_exists(pid):
                    os.kill(pid, signal.SIGKILL)
                    self.logger.warning(f"Force killed service {service_id} (PID: {pid})")
                
                self.logger.info(f"Service {service_id} stopped successfully")
            
            # Remove PID file
            pid_file.unlink()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop service {service_id}: {e}")
            return False
    
    def restart_service(self, service_id: str) -> bool:
        """Restart a specific service"""
        if RICH_AVAILABLE:
            console.print(f"[yellow]🔄 Restarting {self.services[service_id]['name']}...[/yellow]")
        
        self.stop_service(service_id)
        time.sleep(2)
        return self.start_service(service_id)
    
    def is_service_running(self, service_id: str) -> bool:
        """Check if a service is running"""
        service_config = self.services[service_id]
        pid_file = self.run_dir / service_config["pid_file"]
        
        if not pid_file.exists():
            return False
        
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            return psutil.pid_exists(pid)
        except:
            return False
    
    def get_service_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all services"""
        status = {}
        
        for service_id, service_config in self.services.items():
            is_running = self.is_service_running(service_id)
            pid = None
            uptime = None
            
            if is_running:
                pid_file = self.run_dir / service_config["pid_file"]
                try:
                    with open(pid_file, 'r') as f:
                        pid = int(f.read().strip())
                    
                    # Get process start time
                    process = psutil.Process(pid)
                    start_time = datetime.fromtimestamp(process.create_time())
                    uptime = datetime.now() - start_time
                except:
                    pass
            
            status[service_id] = {
                "name": service_config["name"],
                "running": is_running,
                "pid": pid,
                "uptime": str(uptime) if uptime else None,
                "health_url": service_config.get("health_url")
            }
        
        return status
    
    def show_status(self):
        """Show system status"""
        status = self.get_service_status()
        
        if RICH_AVAILABLE:
            console.print(Panel.fit(
                "[bold blue]📊 NICEGOLD Enterprise System Status[/bold blue]",
                title="System Status"
            ))
            
            # Services table
            table = Table(show_header=True, header_style="bold blue")
            table.add_column("Service", style="cyan")
            table.add_column("Status")
            table.add_column("PID")
            table.add_column("Uptime")
            table.add_column("Health URL")
            
            for service_id, service_info in status.items():
                status_icon = "✅ Running" if service_info["running"] else "❌ Stopped"
                pid_str = str(service_info["pid"]) if service_info["pid"] else "-"
                uptime_str = service_info["uptime"] if service_info["uptime"] else "-"
                health_url = service_info.get("health_url", "-")
                
                table.add_row(
                    service_info["name"],
                    status_icon,
                    pid_str,
                    uptime_str,
                    health_url
                )
            
            console.print(table)
            
            # System information
            self._show_system_info()
            
            # Authentication status
            self._show_auth_status()
            
        else:
            print("📊 NICEGOLD Enterprise System Status")
            print("="*50)
            
            for service_id, service_info in status.items():
                status_str = "✅ Running" if service_info["running"] else "❌ Stopped"
                print(f"{service_info['name']}: {status_str}")
                if service_info["pid"]:
                    print(f"  PID: {service_info['pid']}")
                if service_info["uptime"]:
                    print(f"  Uptime: {service_info['uptime']}")
    
    def _show_service_urls(self):
        """Show service URLs after startup"""
        if RICH_AVAILABLE:
            console.print("\n[bold green]🌐 Service Access URLs[/bold green]")
            urls_table = Table(show_header=True, header_style="bold blue")
            urls_table.add_column("Service", style="cyan")
            urls_table.add_column("URL", style="green")
            urls_table.add_column("Description")
            
            urls_table.add_row("API Server", "http://127.0.0.1:8000", "REST API & Health Check")
            urls_table.add_row("Dashboard", "http://127.0.0.1:8501", "Trading Dashboard")
            urls_table.add_row("API Docs", "http://127.0.0.1:8000/docs", "Interactive API Documentation")
            
            console.print(urls_table)
        else:
            print("\n🌐 Service Access URLs:")
            print("API Server: http://127.0.0.1:8000")
            print("Dashboard: http://127.0.0.1:8501")
            print("API Docs: http://127.0.0.1:8000/docs")
    
    def _show_system_info(self):
        """Show system information"""
        try:
            # Memory info
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            memory_percent = memory.percent
            
            # Disk info
            disk = psutil.disk_usage('/')
            disk_used_gb = disk.used / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            disk_percent = (disk.used / disk.total) * 100
            
            # CPU info
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if RICH_AVAILABLE:
                console.print("\n[bold blue]💻 System Information[/bold blue]")
                
                sys_table = Table(show_header=True, header_style="bold blue")
                sys_table.add_column("Resource", style="cyan")
                sys_table.add_column("Usage")
                sys_table.add_column("Details")
                
                sys_table.add_row(
                    "CPU",
                    f"{cpu_percent:.1f}%",
                    f"{psutil.cpu_count()} cores"
                )
                sys_table.add_row(
                    "Memory",
                    f"{memory_percent:.1f}%",
                    f"{memory_used_gb:.1f}GB / {memory_total_gb:.1f}GB"
                )
                sys_table.add_row(
                    "Disk",
                    f"{disk_percent:.1f}%",
                    f"{disk_used_gb:.1f}GB / {disk_total_gb:.1f}GB"
                )
                
                console.print(sys_table)
            else:
                print(f"\n💻 System Information:")
                print(f"CPU: {cpu_percent:.1f}% ({psutil.cpu_count()} cores)")
                print(f"Memory: {memory_percent:.1f}% ({memory_used_gb:.1f}GB / {memory_total_gb:.1f}GB)")
                print(f"Disk: {disk_percent:.1f}% ({disk_used_gb:.1f}GB / {disk_total_gb:.1f}GB)")
                
        except Exception as e:
            self.logger.error(f"Failed to get system info: {e}")
    
    def _show_auth_status(self):
        """Show authentication system status"""
        try:
            from src.single_user_auth import auth_manager
            auth_status = auth_manager.get_system_status()
            
            if RICH_AVAILABLE:
                console.print("\n[bold blue]🔐 Authentication Status[/bold blue]")
                
                auth_table = Table(show_header=True, header_style="bold blue")
                auth_table.add_column("Property", style="cyan")
                auth_table.add_column("Value")
                
                auth_table.add_row("User Configured", "✅ Yes" if auth_status["user_configured"] else "❌ No")
                auth_table.add_row("Username", auth_status["username"])
                auth_table.add_row("Active Sessions", str(auth_status["active_sessions"]))
                auth_table.add_row("Last Login", auth_status["last_login"] or "Never")
                auth_table.add_row("Total Logins", str(auth_status["login_count"]))
                
                console.print(auth_table)
            else:
                print(f"\n🔐 Authentication Status:")
                print(f"User Configured: {'✅ Yes' if auth_status['user_configured'] else '❌ No'}")
                print(f"Username: {auth_status['username']}")
                print(f"Active Sessions: {auth_status['active_sessions']}")
                
        except Exception as e:
            if RICH_AVAILABLE:
                console.print(f"[red]❌ Authentication system error: {e}[/red]")
            else:
                print(f"❌ Authentication system error: {e}")
    
    def health_check(self) -> Dict[str, bool]:
        """Perform health checks on all services"""
        results = {}
        
        if RICH_AVAILABLE:
            console.print("[bold blue]🔍 Performing Health Checks[/bold blue]")
        
        for service_id, service_config in self.services.items():
            is_healthy = False
            
            # Check if service is running
            if self.is_service_running(service_id):
                # Additional health check via HTTP if URL is provided
                health_url = service_config.get("health_url")
                if health_url:
                    try:
                        import requests
                        response = requests.get(health_url, timeout=5)
                        is_healthy = response.status_code == 200
                    except:
                        is_healthy = False
                else:
                    is_healthy = True  # Just check if process is running
            
            results[service_id] = is_healthy
            
            if RICH_AVAILABLE:
                status_icon = "✅" if is_healthy else "❌"
                console.print(f"{status_icon} {service_config['name']}")
            else:
                status_icon = "✅" if is_healthy else "❌"
                print(f"{status_icon} {service_config['name']}")
        
        return results
    
    def create_backup(self) -> bool:
        """Create system backup"""
        if RICH_AVAILABLE:
            console.print("[bold blue]💾 Creating System Backup[/bold blue]")
        
        try:
            backup_dir = self.project_root / "backups"
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"nicegold_backup_{timestamp}"
            backup_path = backup_dir / backup_name
            backup_path.mkdir()
            
            # Backup database
            db_file = self.database_dir / "production.db"
            if db_file.exists():
                shutil.copy2(db_file, backup_path / "production.db")
                if RICH_AVAILABLE:
                    console.print("[green]✅ Database backed up[/green]")
            
            # Backup configuration
            if self.config_dir.exists():
                shutil.copytree(self.config_dir, backup_path / "config")
                if RICH_AVAILABLE:
                    console.print("[green]✅ Configuration backed up[/green]")
            
            # Backup recent logs
            recent_logs = backup_path / "logs"
            recent_logs.mkdir()
            
            # Copy logs from last 7 days
            cutoff_date = datetime.now() - timedelta(days=7)
            for log_file in self.logs_dir.rglob("*.log"):
                if datetime.fromtimestamp(log_file.stat().st_mtime) > cutoff_date:
                    dest_file = recent_logs / log_file.relative_to(self.logs_dir)
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(log_file, dest_file)
            
            # Create compressed archive
            import tarfile
            archive_path = backup_dir / f"{backup_name}.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(backup_path, arcname=backup_name)
            
            # Remove temporary directory
            shutil.rmtree(backup_path)
            
            if RICH_AVAILABLE:
                console.print(f"[green]✅ Backup created: {archive_path}[/green]")
            else:
                print(f"✅ Backup created: {archive_path}")
            
            # Cleanup old backups (keep last 10)
            backup_files = sorted(backup_dir.glob("nicegold_backup_*.tar.gz"))
            if len(backup_files) > 10:
                for old_backup in backup_files[:-10]:
                    old_backup.unlink()
                    if RICH_AVAILABLE:
                        console.print(f"[yellow]🗑️ Removed old backup: {old_backup.name}[/yellow]")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            if RICH_AVAILABLE:
                console.print(f"[red]❌ Backup failed: {e}[/red]")
            return False

def main():
    """Main system manager function"""
    if len(sys.argv) < 2:
        print("NICEGOLD Enterprise System Manager")
        print("Usage:")
        print("  python system_manager.py start       - Start all services")
        print("  python system_manager.py stop        - Stop all services")
        print("  python system_manager.py restart     - Restart all services")
        print("  python system_manager.py status      - Show system status")
        print("  python system_manager.py health      - Run health checks")
        print("  python system_manager.py backup      - Create system backup")
        print("  python system_manager.py service <service_id> <action>")
        print("    Service IDs: api, dashboard")
        print("    Actions: start, stop, restart")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    manager = SystemManager()
    
    try:
        if command == "start":
            success = manager.start_all_services()
            sys.exit(0 if success else 1)
            
        elif command == "stop":
            success = manager.stop_all_services()
            sys.exit(0 if success else 1)
            
        elif command == "restart":
            manager.stop_all_services()
            time.sleep(3)
            success = manager.start_all_services()
            sys.exit(0 if success else 1)
            
        elif command == "status":
            manager.show_status()
            
        elif command == "health":
            results = manager.health_check()
            all_healthy = all(results.values())
            sys.exit(0 if all_healthy else 1)
            
        elif command == "backup":
            success = manager.create_backup()
            sys.exit(0 if success else 1)
            
        elif command == "service" and len(sys.argv) >= 4:
            service_id = sys.argv[2]
            action = sys.argv[3].lower()
            
            if service_id not in manager.services:
                print(f"Unknown service: {service_id}")
                sys.exit(1)
            
            if action == "start":
                success = manager.start_service(service_id)
            elif action == "stop":
                success = manager.stop_service(service_id)
            elif action == "restart":
                success = manager.restart_service(service_id)
            else:
                print(f"Unknown action: {action}")
                sys.exit(1)
            
            sys.exit(0 if success else 1)
            
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        if RICH_AVAILABLE:
            console.print("\n[yellow]⚠️ Operation cancelled by user[/yellow]")
        else:
            print("\n⚠️ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"[red]❌ Error: {e}[/red]")
        else:
            print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
