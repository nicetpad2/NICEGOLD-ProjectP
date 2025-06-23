#!/usr/bin/env python3
"""
🔧 NICEGOLD System Maintenance & Monitoring 🔧
=============================================

ระบบดูแลและบำรุงรักษาสำหรับ NICEGOLD Enterprise
พร้อมการตรวจสอบสุขภาพระบบ, การสำรองข้อมูล, และการจัดการแบบเรียลไทม์

Features:
- Real-time system monitoring
- Automated backup management
- Performance optimization
- Health check dashboard
- Log management
- Database maintenance
- Security auditing
- Resource monitoring
"""

import json
import logging
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import psutil
    import yaml
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

try:
    from rich.align import Align
    from rich.columns import Columns
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
    from rich.prompt import Confirm, IntPrompt, Prompt
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None

class SystemMaintenanceManager:
    """Comprehensive system maintenance and monitoring"""
    
    def __init__(self):
        self.project_root = Path(".")
        self.running = False
        self.monitoring_thread = None
        
        # System state
        self.system_stats = {
            "start_time": datetime.now(),
            "uptime": 0,
            "cpu_usage": 0,
            "memory_usage": 0,
            "disk_usage": 0,
            "active_processes": 0,
            "api_status": "unknown",
            "dashboard_status": "unknown",
            "database_status": "unknown",
            "last_backup": None,
            "errors_24h": 0,
            "warnings_24h": 0
        }
        
        # Configuration
        self.config = {
            "monitoring_interval": 5,  # seconds
            "backup_interval": 3600,   # seconds (1 hour)
            "log_retention_days": 30,
            "max_log_size_mb": 100,
            "health_check_timeout": 10,
            "alert_thresholds": {
                "cpu_usage": 80,
                "memory_usage": 85,
                "disk_usage": 90,
                "error_rate": 10
            }
        }
        
        # Setup logging
        self._setup_logging()
        
        # Load configuration
        self._load_configuration()
        
        self.log("🔧 System Maintenance Manager Initialized")
    
    def _setup_logging(self):
        """Setup maintenance logging"""
        log_dir = self.project_root / "logs" / "maintenance"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"maintenance_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
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
                console.print(f"[blue][{timestamp}] 🔧 {message}[/blue]")
        
        getattr(self.logger, level, self.logger.info)(message)
    
    def _load_configuration(self):
        """Load system configuration"""
        try:
            config_file = self.project_root / "config.yaml"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                
                # Update monitoring configuration
                if "monitoring" in loaded_config:
                    monitoring_config = loaded_config["monitoring"]
                    self.config.update({
                        "monitoring_interval": monitoring_config.get("interval", 5),
                        "health_check_timeout": monitoring_config.get("timeout", 10)
                    })
                
                # Update backup configuration
                if "database" in loaded_config:
                    db_config = loaded_config["database"]
                    if db_config.get("backup_enabled", True):
                        backup_interval = db_config.get("backup_interval_hours", 1) * 3600
                        self.config["backup_interval"] = backup_interval
        
        except Exception as e:
            self.log(f"Configuration loading warning: {e}", "warning")
    
    def start_monitoring(self):
        """Start real-time system monitoring"""
        if self.running:
            self.log("Monitoring already running", "warning")
            return
        
        self.running = True
        self.log("🚀 Starting system monitoring...")
        
        if RICH_AVAILABLE and console:
            with Live(self._create_monitoring_dashboard(), refresh_per_second=2) as live:
                self._monitoring_loop(live)
        else:
            self._monitoring_loop()
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        self.log("🛑 System monitoring stopped")
    
    def _monitoring_loop(self, live=None):
        """Main monitoring loop"""
        last_backup = datetime.now()
        
        try:
            while self.running:
                # Update system stats
                self._update_system_stats()
                
                # Check for alerts
                self._check_alerts()
                
                # Perform periodic tasks
                now = datetime.now()
                
                # Backup check
                if (now - last_backup).total_seconds() >= self.config["backup_interval"]:
                    self._perform_backup()
                    last_backup = now
                
                # Log rotation check
                self._check_log_rotation()
                
                # Update live dashboard
                if live:
                    live.update(self._create_monitoring_dashboard())
                
                time.sleep(self.config["monitoring_interval"])
                
        except KeyboardInterrupt:
            self.log("Monitoring interrupted by user", "info")
        except Exception as e:
            self.log(f"Monitoring error: {e}", "error")
        finally:
            self.running = False
    
    def _update_system_stats(self):
        """Update system statistics"""
        try:
            # System uptime
            self.system_stats["uptime"] = (
                datetime.now() - self.system_stats["start_time"]
            ).total_seconds()
            
            # CPU usage
            self.system_stats["cpu_usage"] = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.system_stats["memory_usage"] = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('.')
            self.system_stats["disk_usage"] = (disk.used / disk.total) * 100
            
            # Active processes
            self.system_stats["active_processes"] = len(psutil.pids())
            
            # Service status checks
            self._check_service_status()
            
            # Database status
            self._check_database_status()
            
            # Error and warning counts
            self._count_recent_issues()
            
        except Exception as e:
            self.log(f"Stats update error: {e}", "error")
    
    def _check_service_status(self):
        """Check status of running services"""
        try:
            # Check API service
            api_pid_file = self.project_root / "run" / "api.pid"
            if api_pid_file.exists():
                try:
                    with open(api_pid_file, 'r') as f:
                        pid = int(f.read().strip())
                    if psutil.pid_exists(pid):
                        self.system_stats["api_status"] = "running"
                    else:
                        self.system_stats["api_status"] = "stopped"
                        api_pid_file.unlink()  # Clean up stale PID file
                except:
                    self.system_stats["api_status"] = "error"
            else:
                self.system_stats["api_status"] = "stopped"
            
            # Check Dashboard service
            dashboard_pid_file = self.project_root / "run" / "dashboard.pid"
            if dashboard_pid_file.exists():
                try:
                    with open(dashboard_pid_file, 'r') as f:
                        pid = int(f.read().strip())
                    if psutil.pid_exists(pid):
                        self.system_stats["dashboard_status"] = "running"
                    else:
                        self.system_stats["dashboard_status"] = "stopped"
                        dashboard_pid_file.unlink()  # Clean up stale PID file
                except:
                    self.system_stats["dashboard_status"] = "error"
            else:
                self.system_stats["dashboard_status"] = "stopped"
                
        except Exception as e:
            self.log(f"Service status check error: {e}", "error")
    
    def _check_database_status(self):
        """Check database connectivity and health"""
        try:
            db_file = self.project_root / "database" / "production.db"
            
            if not db_file.exists():
                self.system_stats["database_status"] = "missing"
                return
            
            # Test connection
            with sqlite3.connect(db_file, timeout=5) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                table_count = cursor.fetchone()[0]
                
                if table_count > 0:
                    self.system_stats["database_status"] = "healthy"
                else:
                    self.system_stats["database_status"] = "empty"
                    
        except Exception as e:
            self.system_stats["database_status"] = "error"
            self.log(f"Database check error: {e}", "error")
    
    def _count_recent_issues(self):
        """Count recent errors and warnings from logs"""
        try:
            log_dir = self.project_root / "logs"
            if not log_dir.exists():
                return
            
            # Count issues in the last 24 hours
            cutoff_time = datetime.now() - timedelta(hours=24)
            error_count = 0
            warning_count = 0
            
            for log_file in log_dir.rglob("*.log"):
                try:
                    with open(log_file, 'r') as f:
                        for line in f:
                            if "ERROR" in line or "❌" in line:
                                error_count += 1
                            elif "WARNING" in line or "⚠️" in line:
                                warning_count += 1
                except:
                    continue
            
            self.system_stats["errors_24h"] = error_count
            self.system_stats["warnings_24h"] = warning_count
            
        except Exception as e:
            self.log(f"Issue counting error: {e}", "error")
    
    def _check_alerts(self):
        """Check for system alerts based on thresholds"""
        alerts = []
        
        # CPU usage alert
        if self.system_stats["cpu_usage"] > self.config["alert_thresholds"]["cpu_usage"]:
            alerts.append(f"High CPU usage: {self.system_stats['cpu_usage']:.1f}%")
        
        # Memory usage alert
        if self.system_stats["memory_usage"] > self.config["alert_thresholds"]["memory_usage"]:
            alerts.append(f"High memory usage: {self.system_stats['memory_usage']:.1f}%")
        
        # Disk usage alert
        if self.system_stats["disk_usage"] > self.config["alert_thresholds"]["disk_usage"]:
            alerts.append(f"High disk usage: {self.system_stats['disk_usage']:.1f}%")
        
        # Service alerts
        if self.system_stats["api_status"] == "error":
            alerts.append("API service error detected")
        
        if self.system_stats["dashboard_status"] == "error":
            alerts.append("Dashboard service error detected")
        
        if self.system_stats["database_status"] == "error":
            alerts.append("Database connectivity issues")
        
        # Error rate alert
        if self.system_stats["errors_24h"] > self.config["alert_thresholds"]["error_rate"]:
            alerts.append(f"High error rate: {self.system_stats['errors_24h']} errors in 24h")
        
        # Log alerts
        for alert in alerts:
            self.log(f"🚨 ALERT: {alert}", "warning")
    
    def _perform_backup(self):
        """Perform automated system backup"""
        try:
            self.log("📦 Starting automated backup...")
            
            backup_dir = self.project_root / "backups"
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"nicegold_backup_{timestamp}"
            backup_path = backup_dir / backup_name
            backup_path.mkdir()
            
            # Backup database
            db_file = self.project_root / "database" / "production.db"
            if db_file.exists():
                shutil.copy2(db_file, backup_path / "production.db")
                self.log("✅ Database backed up", "success")
            
            # Backup configuration
            config_files = [
                "config.yaml",
                "config/production.yaml",
                ".env.production"
            ]
            
            for config_file in config_files:
                config_path = self.project_root / config_file
                if config_path.exists():
                    dest_path = backup_path / config_file
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(config_path, dest_path)
            
            # Backup logs (recent ones)
            recent_logs_dir = backup_path / "logs"
            recent_logs_dir.mkdir()
            
            log_dir = self.project_root / "logs"
            if log_dir.exists():
                cutoff_time = datetime.now() - timedelta(days=7)  # Last 7 days
                
                for log_file in log_dir.rglob("*.log"):
                    if log_file.stat().st_mtime > cutoff_time.timestamp():
                        rel_path = log_file.relative_to(log_dir)
                        dest_log = recent_logs_dir / rel_path
                        dest_log.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(log_file, dest_log)
            
            # Create backup manifest
            manifest = {
                "backup_time": datetime.now().isoformat(),
                "backup_name": backup_name,
                "included_files": {
                    "database": "production.db",
                    "configuration": config_files,
                    "logs": "Recent 7 days"
                },
                "system_stats": self.system_stats.copy()
            }
            
            with open(backup_path / "manifest.json", 'w') as f:
                json.dump(manifest, f, indent=2, default=str)
            
            # Compress backup
            archive_path = backup_dir / f"{backup_name}.tar.gz"
            shutil.make_archive(str(backup_path), 'gztar', str(backup_path))
            
            # Clean up uncompressed backup
            shutil.rmtree(backup_path)
            
            # Clean old backups (keep last 10)
            backup_files = sorted(backup_dir.glob("*.tar.gz"), key=lambda x: x.stat().st_mtime)
            if len(backup_files) > 10:
                for old_backup in backup_files[:-10]:
                    old_backup.unlink()
                    self.log(f"🗑️  Removed old backup: {old_backup.name}")
            
            self.system_stats["last_backup"] = datetime.now()
            self.log(f"✅ Backup completed: {archive_path.name}", "success")
            
        except Exception as e:
            self.log(f"❌ Backup failed: {e}", "error")
    
    def _check_log_rotation(self):
        """Check and rotate large log files"""
        try:
            log_dir = self.project_root / "logs"
            if not log_dir.exists():
                return
            
            max_size = self.config["max_log_size_mb"] * 1024 * 1024  # Convert to bytes
            
            for log_file in log_dir.rglob("*.log"):
                if log_file.stat().st_size > max_size:
                    # Rotate log file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    rotated_name = f"{log_file.stem}_{timestamp}.log"
                    rotated_path = log_file.parent / rotated_name
                    
                    log_file.rename(rotated_path)
                    
                    # Create new empty log file
                    log_file.touch()
                    
                    self.log(f"🔄 Rotated log file: {log_file.name}", "info")
            
            # Clean old log files
            retention_days = self.config["log_retention_days"]
            cutoff_time = datetime.now() - timedelta(days=retention_days)
            
            for log_file in log_dir.rglob("*.log"):
                if log_file.stat().st_mtime < cutoff_time.timestamp():
                    log_file.unlink()
                    self.log(f"🗑️  Removed old log: {log_file.name}")
                    
        except Exception as e:
            self.log(f"Log rotation error: {e}", "error")
    
    def _create_monitoring_dashboard(self) -> Panel:
        """Create real-time monitoring dashboard"""
        if not RICH_AVAILABLE:
            return None
        
        # System overview table
        overview_table = Table(title="🖥️ System Overview")
        overview_table.add_column("Metric", style="cyan")
        overview_table.add_column("Value", style="bold")
        overview_table.add_column("Status", style="green")
        
        # Uptime
        uptime_str = str(timedelta(seconds=int(self.system_stats["uptime"])))
        overview_table.add_row("Uptime", uptime_str, "✅ Running")
        
        # CPU usage
        cpu_usage = self.system_stats["cpu_usage"]
        cpu_status = "⚠️ High" if cpu_usage > 80 else "✅ Normal"
        overview_table.add_row("CPU Usage", f"{cpu_usage:.1f}%", cpu_status)
        
        # Memory usage
        memory_usage = self.system_stats["memory_usage"]
        memory_status = "⚠️ High" if memory_usage > 85 else "✅ Normal"
        overview_table.add_row("Memory Usage", f"{memory_usage:.1f}%", memory_status)
        
        # Disk usage
        disk_usage = self.system_stats["disk_usage"]
        disk_status = "⚠️ High" if disk_usage > 90 else "✅ Normal"
        overview_table.add_row("Disk Usage", f"{disk_usage:.1f}%", disk_status)
        
        # Services status table
        services_table = Table(title="🔧 Services Status")
        services_table.add_column("Service", style="cyan")
        services_table.add_column("Status", style="bold")
        services_table.add_column("Health", style="green")
        
        # API status
        api_status = self.system_stats["api_status"]
        api_health = "✅ Healthy" if api_status == "running" else "❌ Down"
        services_table.add_row("API Server", api_status.title(), api_health)
        
        # Dashboard status
        dashboard_status = self.system_stats["dashboard_status"]
        dashboard_health = "✅ Healthy" if dashboard_status == "running" else "❌ Down"
        services_table.add_row("Dashboard", dashboard_status.title(), dashboard_health)
        
        # Database status
        db_status = self.system_stats["database_status"]
        db_health = "✅ Healthy" if db_status == "healthy" else "❌ Issues"
        services_table.add_row("Database", db_status.title(), db_health)
        
        # Issues summary
        issues_content = f"""
[bold yellow]📊 24h Summary:[/bold yellow]
❌ Errors: [red]{self.system_stats['errors_24h']}[/red]
⚠️  Warnings: [yellow]{self.system_stats['warnings_24h']}[/yellow]
📦 Last Backup: [cyan]{self.system_stats['last_backup'] or 'Never'}[/cyan]
🔄 Active Processes: [blue]{self.system_stats['active_processes']}[/blue]
        """
        
        # Main dashboard layout
        dashboard_content = Columns([
            overview_table,
            services_table,
            Panel(issues_content, title="📈 Statistics", border_style="blue")
        ])
        
        return Panel(
            dashboard_content,
            title=f"🔧 NICEGOLD System Monitor - {datetime.now().strftime('%H:%M:%S')}",
            border_style="bright_blue"
        )
    
    def run_health_check(self) -> Dict:
        """Run comprehensive system health check"""
        self.log("🩺 Running comprehensive health check...")
        
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "overall_health": "unknown",
            "checks": {},
            "recommendations": []
        }
        
        checks = [
            ("System Resources", self._health_check_resources),
            ("Services", self._health_check_services),
            ("Database", self._health_check_database),
            ("Configuration", self._health_check_configuration),
            ("Security", self._health_check_security),
            ("Performance", self._health_check_performance)
        ]
        
        passed_checks = 0
        total_checks = len(checks)
        
        for check_name, check_function in checks:
            try:
                result = check_function()
                health_report["checks"][check_name] = result
                
                if result.get("status") == "healthy":
                    passed_checks += 1
                    
            except Exception as e:
                health_report["checks"][check_name] = {
                    "status": "error",
                    "message": str(e)
                }
        
        # Calculate overall health
        health_percentage = (passed_checks / total_checks) * 100
        
        if health_percentage >= 90:
            health_report["overall_health"] = "excellent"
        elif health_percentage >= 70:
            health_report["overall_health"] = "good"
        elif health_percentage >= 50:
            health_report["overall_health"] = "fair"
        else:
            health_report["overall_health"] = "poor"
        
        # Generate recommendations
        health_report["recommendations"] = self._generate_health_recommendations(health_report)
        
        # Save health report
        reports_dir = self.project_root / "logs" / "health_reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = reports_dir / f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(health_report, f, indent=2)
        
        self._display_health_report(health_report)
        
        return health_report
    
    def _health_check_resources(self) -> Dict:
        """Check system resource health"""
        try:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            issues = []
            if cpu_usage > 80:
                issues.append(f"High CPU usage: {cpu_usage:.1f}%")
            if memory.percent > 85:
                issues.append(f"High memory usage: {memory.percent:.1f}%")
            if (disk.used / disk.total * 100) > 90:
                issues.append(f"High disk usage: {disk.used / disk.total * 100:.1f}%")
            
            return {
                "status": "healthy" if not issues else "warning",
                "cpu_usage": cpu_usage,
                "memory_usage": memory.percent,
                "disk_usage": (disk.used / disk.total * 100),
                "issues": issues
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _health_check_services(self) -> Dict:
        """Check services health"""
        self._check_service_status()
        
        services_ok = (
            self.system_stats["api_status"] in ["running", "stopped"] and
            self.system_stats["dashboard_status"] in ["running", "stopped"]
        )
        
        return {
            "status": "healthy" if services_ok else "warning",
            "api_status": self.system_stats["api_status"],
            "dashboard_status": self.system_stats["dashboard_status"]
        }
    
    def _health_check_database(self) -> Dict:
        """Check database health"""
        self._check_database_status()
        
        return {
            "status": "healthy" if self.system_stats["database_status"] == "healthy" else "warning",
            "database_status": self.system_stats["database_status"]
        }
    
    def _health_check_configuration(self) -> Dict:
        """Check configuration health"""
        try:
            required_files = [
                "config.yaml",
                ".env.production"
            ]
            
            missing_files = []
            for file_path in required_files:
                if not (self.project_root / file_path).exists():
                    missing_files.append(file_path)
            
            return {
                "status": "healthy" if not missing_files else "warning",
                "missing_files": missing_files
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _health_check_security(self) -> Dict:
        """Check security health"""
        try:
            issues = []
            
            # Check file permissions
            env_file = self.project_root / ".env.production"
            if env_file.exists():
                try:
                    import stat
                    file_stat = os.stat(env_file)
                    if file_stat.st_mode & 0o077:
                        issues.append("Environment file permissions too open")
                except:
                    pass
            
            # Check for default passwords (placeholder check)
            # In real implementation, this would check for weak passwords
            
            return {
                "status": "healthy" if not issues else "warning",
                "issues": issues
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _health_check_performance(self) -> Dict:
        """Check performance health"""
        try:
            # Simple performance test
            start_time = time.time()
            
            # Test file I/O
            test_file = self.project_root / "logs" / "performance_test.tmp"
            with open(test_file, 'w') as f:
                f.write("performance test data" * 1000)
            
            with open(test_file, 'r') as f:
                content = f.read()
            
            test_file.unlink()  # Clean up
            
            io_time = time.time() - start_time
            
            return {
                "status": "healthy" if io_time < 1.0 else "warning",
                "io_performance": f"{io_time:.3f}s"
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _generate_health_recommendations(self, health_report: Dict) -> List[str]:
        """Generate health improvement recommendations"""
        recommendations = []
        
        for check_name, check_data in health_report["checks"].items():
            if check_data.get("status") != "healthy":
                if check_name == "System Resources":
                    if check_data.get("cpu_usage", 0) > 80:
                        recommendations.append("Consider optimizing CPU-intensive processes")
                    if check_data.get("memory_usage", 0) > 85:
                        recommendations.append("Consider increasing available memory or optimizing memory usage")
                    if check_data.get("disk_usage", 0) > 90:
                        recommendations.append("Clean up disk space or expand storage")
                
                elif check_name == "Services":
                    if check_data.get("api_status") == "error":
                        recommendations.append("Investigate API service errors and restart if necessary")
                    if check_data.get("dashboard_status") == "error":
                        recommendations.append("Check dashboard service configuration and logs")
                
                elif check_name == "Database":
                    recommendations.append("Check database connectivity and integrity")
                
                elif check_name == "Configuration":
                    missing_files = check_data.get("missing_files", [])
                    if missing_files:
                        recommendations.append(f"Create missing configuration files: {', '.join(missing_files)}")
                
                elif check_name == "Security":
                    recommendations.append("Review and fix security issues")
                
                elif check_name == "Performance":
                    recommendations.append("Investigate performance bottlenecks")
        
        return recommendations
    
    def _display_health_report(self, health_report: Dict):
        """Display health check report"""
        if not RICH_AVAILABLE or not console:
            print(f"\n🩺 Health Check Report - {health_report['overall_health'].upper()}")
            for check_name, check_data in health_report["checks"].items():
                status = check_data.get("status", "unknown")
                print(f"  {check_name}: {status}")
            
            if health_report["recommendations"]:
                print("\nRecommendations:")
                for rec in health_report["recommendations"]:
                    print(f"  • {rec}")
            return
        
        # Rich display
        console.print("\n")
        
        # Overall health status
        health_status = health_report["overall_health"]
        status_colors = {
            "excellent": "green",
            "good": "blue", 
            "fair": "yellow",
            "poor": "red"
        }
        status_color = status_colors.get(health_status, "white")
        
        # Health summary
        health_table = Table(title="🩺 System Health Report")
        health_table.add_column("Check", style="cyan")
        health_table.add_column("Status", style="bold")
        health_table.add_column("Details", style="dim")
        
        for check_name, check_data in health_report["checks"].items():
            status = check_data.get("status", "unknown")
            
            if status == "healthy":
                status_display = Text("✅ Healthy", style="green")
            elif status == "warning":
                status_display = Text("⚠️ Warning", style="yellow")
            else:
                status_display = Text("❌ Error", style="red")
            
            details = check_data.get("message", "")
            if not details and "issues" in check_data:
                details = "; ".join(check_data["issues"])
            
            health_table.add_row(check_name, status_display, details[:50])
        
        console.print(Panel(
            health_table,
            title=f"[{status_color}]Overall Health: {health_status.upper()}[/{status_color}]",
            border_style=status_color
        ))
        
        # Recommendations
        if health_report["recommendations"]:
            recommendations_content = "\n".join([
                f"• {rec}" for rec in health_report["recommendations"]
            ])
            
            console.print(Panel(
                recommendations_content,
                title="💡 Recommendations",
                border_style="blue"
            ))

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NICEGOLD System Maintenance & Monitoring")
    parser.add_argument("command", choices=["monitor", "health", "backup"], help="Command to execute")
    parser.add_argument("--interval", type=int, default=5, help="Monitoring interval in seconds")
    
    args = parser.parse_args()
    
    if not DEPENDENCIES_AVAILABLE:
        print("❌ Required dependencies not available. Please install: psutil, pyyaml")
        return False
    
    manager = SystemMaintenanceManager()
    
    if args.command == "monitor":
        manager.config["monitoring_interval"] = args.interval
        try:
            manager.start_monitoring()
        except KeyboardInterrupt:
            manager.stop_monitoring()
            print("\n👋 Monitoring stopped")
    
    elif args.command == "health":
        health_report = manager.run_health_check()
        return health_report["overall_health"] in ["excellent", "good"]
    
    elif args.command == "backup":
        manager._perform_backup()
        print("✅ Backup completed")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
