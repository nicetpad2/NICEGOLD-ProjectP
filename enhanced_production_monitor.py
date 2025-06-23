#!/usr/bin/env python3
"""
🚀 NICEGOLD ENTERPRISE - PRODUCTION MONITORING SYSTEM
====================================================

Advanced production monitoring, alerting, and health management system.
Provides real-time monitoring, automated recovery, and comprehensive alerting.

Version: 3.0
Author: NICEGOLD Team
"""

import json
import logging
import os
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import psutil
import requests

# Rich for beautiful output
try:
    from rich import box
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.text import Text
    console = Console()
except ImportError:
    class Console:
        def print(self, *args, **kwargs):
            print(*args)
    console = Console()

@dataclass
class SystemMetrics:
    """System metrics data structure"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    api_status: str
    dashboard_status: str
    database_status: str
    ai_system_status: str
    active_connections: int
    response_time: float

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    timestamp: str
    severity: str  # CRITICAL, WARNING, INFO
    component: str
    message: str
    resolved: bool = False
    resolution_time: Optional[str] = None

class ProductionMonitor:
    """Advanced production monitoring system"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.config_dir = self.base_dir / "config"
        self.logs_dir = self.base_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Monitoring configuration
        self.monitoring_interval = 10  # seconds
        self.alert_thresholds = {
            'cpu_percent': 85.0,
            'memory_percent': 90.0,
            'disk_percent': 85.0,
            'response_time': 5.0,  # seconds
            'database_size_mb': 1000.0
        }
        
        # State tracking
        self.running = False
        self.alerts: List[Alert] = []
        self.metrics_history: List[SystemMetrics] = []
        self.recovery_attempts = {}
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.logs_dir / 'production_monitor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def check_service_status(self, host: str, port: int, timeout: int = 5) -> bool:
        """Check if a service is running on the specified host:port"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    def check_api_health(self) -> tuple[str, float]:
        """Check API health and response time"""
        try:
            start_time = time.time()
            response = requests.get(
                "http://127.0.0.1:8000/health",
                timeout=10
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                return "HEALTHY", response_time
            else:
                return "UNHEALTHY", response_time
        except requests.exceptions.RequestException:
            if self.check_service_status("127.0.0.1", 8000):
                return "RUNNING_NO_HEALTH", 0.0
            return "DOWN", 0.0
        except Exception:
            return "ERROR", 0.0
    
    def check_database_health(self) -> str:
        """Check database health and connectivity"""
        try:
            db_path = self.base_dir / "database" / "production.db"
            if not db_path.exists():
                return "NOT_FOUND"
            
            conn = sqlite3.connect(str(db_path), timeout=5)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            conn.close()
            return "HEALTHY"
        except sqlite3.Error:
            return "ERROR"
        except Exception:
            return "UNKNOWN"
    
    def check_ai_system_health(self) -> str:
        """Check AI system components"""
        try:
            ai_scripts = [
                "ai_orchestrator.py",
                "ai_team_manager.py",
                "ai_assistant_brain.py"
            ]
            
            missing_scripts = []
            for script in ai_scripts:
                if not (self.base_dir / script).exists():
                    missing_scripts.append(script)
            
            if missing_scripts:
                return f"INCOMPLETE ({len(missing_scripts)} missing)"
            
            return "AVAILABLE"
        except Exception:
            return "ERROR"
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # Service health checks
            api_status, response_time = self.check_api_health()
            dashboard_status = "RUNNING" if self.check_service_status("127.0.0.1", 8501) else "DOWN"
            database_status = self.check_database_health()
            ai_system_status = self.check_ai_system_health()
            
            # Active connections (approximate)
            connections = len([p for p in psutil.process_iter(['pid', 'name']) 
                             if 'python' in p.info['name'].lower()])
            
            metrics = SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk.percent,
                network_io={
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv
                },
                api_status=api_status,
                dashboard_status=dashboard_status,
                database_status=database_status,
                ai_system_status=ai_system_status,
                active_connections=connections,
                response_time=response_time
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {str(e)}")
            # Return basic metrics on error
            return SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_percent=0.0,
                network_io={'bytes_sent': 0, 'bytes_recv': 0},
                api_status="ERROR",
                dashboard_status="ERROR",
                database_status="ERROR",
                ai_system_status="ERROR",
                active_connections=0,
                response_time=0.0
            )
    
    def check_alerts(self, metrics: SystemMetrics) -> List[Alert]:
        """Check for alert conditions based on metrics"""
        new_alerts = []
        current_time = datetime.now().isoformat()
        
        # CPU alert
        if metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
            alert = Alert(
                id=f"cpu_high_{int(time.time())}",
                timestamp=current_time,
                severity="WARNING",
                component="CPU",
                message=f"High CPU usage: {metrics.cpu_percent:.1f}%"
            )
            new_alerts.append(alert)
        
        # Memory alert
        if metrics.memory_percent > self.alert_thresholds['memory_percent']:
            alert = Alert(
                id=f"memory_high_{int(time.time())}",
                timestamp=current_time,
                severity="CRITICAL",
                component="Memory",
                message=f"High memory usage: {metrics.memory_percent:.1f}%"
            )
            new_alerts.append(alert)
        
        # Disk alert
        if metrics.disk_percent > self.alert_thresholds['disk_percent']:
            alert = Alert(
                id=f"disk_high_{int(time.time())}",
                timestamp=current_time,
                severity="WARNING",
                component="Disk",
                message=f"High disk usage: {metrics.disk_percent:.1f}%"
            )
            new_alerts.append(alert)
        
        # Service alerts
        if metrics.api_status in ["DOWN", "ERROR"]:
            alert = Alert(
                id=f"api_down_{int(time.time())}",
                timestamp=current_time,
                severity="CRITICAL",
                component="API",
                message=f"API service is {metrics.api_status}"
            )
            new_alerts.append(alert)
        
        if metrics.dashboard_status == "DOWN":
            alert = Alert(
                id=f"dashboard_down_{int(time.time())}",
                timestamp=current_time,
                severity="WARNING",
                component="Dashboard",
                message="Dashboard service is down"
            )
            new_alerts.append(alert)
        
        if metrics.database_status not in ["HEALTHY"]:
            alert = Alert(
                id=f"db_issue_{int(time.time())}",
                timestamp=current_time,
                severity="CRITICAL",
                component="Database",
                message=f"Database status: {metrics.database_status}"
            )
            new_alerts.append(alert)
        
        # Response time alert
        if metrics.response_time > self.alert_thresholds['response_time']:
            alert = Alert(
                id=f"slow_response_{int(time.time())}",
                timestamp=current_time,
                severity="WARNING",
                component="Performance",
                message=f"Slow API response: {metrics.response_time:.2f}s"
            )
            new_alerts.append(alert)
        
        return new_alerts
    
    def attempt_recovery(self, alert: Alert) -> bool:
        """Attempt automated recovery for certain types of alerts"""
        try:
            if alert.component == "API" and alert.severity == "CRITICAL":
                self.logger.info("Attempting API recovery...")
                
                # Try to restart API service
                result = subprocess.run([
                    sys.executable, "start_production_single_user.py", "--restart-api"
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    self.logger.info("API recovery successful")
                    return True
                else:
                    self.logger.error(f"API recovery failed: {result.stderr}")
                    return False
            
            elif alert.component == "Dashboard" and alert.severity == "WARNING":
                self.logger.info("Attempting Dashboard recovery...")
                
                # Try to restart Dashboard service
                result = subprocess.run([
                    sys.executable, "start_production_single_user.py", "--restart-dashboard"
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    self.logger.info("Dashboard recovery successful")
                    return True
                else:
                    self.logger.error(f"Dashboard recovery failed: {result.stderr}")
                    return False
            
            return False
            
        except Exception as e:
            self.logger.error(f"Recovery attempt failed: {str(e)}")
            return False
    
    def save_metrics(self, metrics: SystemMetrics) -> None:
        """Save metrics to file for historical analysis"""
        try:
            metrics_file = self.logs_dir / "system_metrics.jsonl"
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(asdict(metrics)) + '\n')
        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")
    
    def create_monitoring_dashboard(self, metrics: SystemMetrics) -> Layout:
        """Create rich monitoring dashboard layout"""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        layout["left"].split_column(
            Layout(name="system"),
            Layout(name="services")
        )
        
        layout["right"].split_column(
            Layout(name="alerts"),
            Layout(name="performance")
        )
        
        # Header
        layout["header"].update(Panel.fit(
            f"[bold cyan]🚀 NICEGOLD ENTERPRISE - PRODUCTION MONITOR[/bold cyan]\n"
            f"[dim]Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
            box=box.DOUBLE
        ))
        
        # System metrics table
        system_table = Table(title="System Resources", box=box.ROUNDED)
        system_table.add_column("Metric", style="cyan")
        system_table.add_column("Value", style="magenta")
        system_table.add_column("Status", style="green")
        
        cpu_status = "🔴 HIGH" if metrics.cpu_percent > 85 else "🟡 MEDIUM" if metrics.cpu_percent > 70 else "🟢 NORMAL"
        memory_status = "🔴 HIGH" if metrics.memory_percent > 90 else "🟡 MEDIUM" if metrics.memory_percent > 75 else "🟢 NORMAL"
        disk_status = "🔴 HIGH" if metrics.disk_percent > 85 else "🟡 MEDIUM" if metrics.disk_percent > 70 else "🟢 NORMAL"
        
        system_table.add_row("CPU Usage", f"{metrics.cpu_percent:.1f}%", cpu_status)
        system_table.add_row("Memory Usage", f"{metrics.memory_percent:.1f}%", memory_status)
        system_table.add_row("Disk Usage", f"{metrics.disk_percent:.1f}%", disk_status)
        system_table.add_row("Active Connections", str(metrics.active_connections), "🟢 NORMAL")
        
        layout["system"].update(Panel(system_table))
        
        # Services status table
        services_table = Table(title="Services Status", box=box.ROUNDED)
        services_table.add_column("Service", style="cyan")
        services_table.add_column("Status", style="magenta")
        services_table.add_column("Details", style="yellow")
        
        api_emoji = "🟢" if metrics.api_status == "HEALTHY" else "🔴"
        dashboard_emoji = "🟢" if metrics.dashboard_status == "RUNNING" else "🔴"
        db_emoji = "🟢" if metrics.database_status == "HEALTHY" else "🔴"
        ai_emoji = "🟢" if metrics.ai_system_status == "AVAILABLE" else "🟡"
        
        services_table.add_row("API Server", f"{api_emoji} {metrics.api_status}", f"Response: {metrics.response_time:.2f}s")
        services_table.add_row("Dashboard", f"{dashboard_emoji} {metrics.dashboard_status}", "Port 8501")
        services_table.add_row("Database", f"{db_emoji} {metrics.database_status}", "SQLite")
        services_table.add_row("AI System", f"{ai_emoji} {metrics.ai_system_status}", "5 Agents")
        
        layout["services"].update(Panel(services_table))
        
        # Recent alerts
        alerts_table = Table(title="Recent Alerts", box=box.ROUNDED)
        alerts_table.add_column("Time", style="cyan")
        alerts_table.add_column("Severity", style="red")
        alerts_table.add_column("Component", style="yellow")
        alerts_table.add_column("Message", style="white")
        
        # Show last 5 alerts
        recent_alerts = sorted(self.alerts, key=lambda x: x.timestamp, reverse=True)[:5]
        for alert in recent_alerts:
            severity_color = "red" if alert.severity == "CRITICAL" else "yellow" if alert.severity == "WARNING" else "green"
            alerts_table.add_row(
                alert.timestamp.split('T')[1][:8],  # Time only
                f"[{severity_color}]{alert.severity}[/{severity_color}]",
                alert.component,
                alert.message[:50] + "..." if len(alert.message) > 50 else alert.message
            )
        
        if not recent_alerts:
            alerts_table.add_row("--", "[green]INFO[/green]", "System", "No recent alerts")
        
        layout["alerts"].update(Panel(alerts_table))
        
        # Performance metrics
        perf_text = f"""[bold]Performance Metrics[/bold]

[cyan]Network I/O:[/cyan]
  Sent: {metrics.network_io['bytes_sent'] / 1024 / 1024:.1f} MB
  Received: {metrics.network_io['bytes_recv'] / 1024 / 1024:.1f} MB

[cyan]Response Time:[/cyan] {metrics.response_time:.3f}s
[cyan]Uptime:[/cyan] {self.get_uptime()}
[cyan]Total Alerts:[/cyan] {len(self.alerts)}
[cyan]Active Alerts:[/cyan] {len([a for a in self.alerts if not a.resolved])}
"""
        
        layout["performance"].update(Panel(perf_text, title="Performance"))
        
        # Footer
        layout["footer"].update(Panel.fit(
            f"[dim]Monitoring Interval: {self.monitoring_interval}s | "
            f"Alerts Threshold: CPU {self.alert_thresholds['cpu_percent']}% | "
            f"Memory {self.alert_thresholds['memory_percent']}% | "
            f"Disk {self.alert_thresholds['disk_percent']}%[/dim]",
            box=box.DOUBLE
        ))
        
        return layout
    
    def get_uptime(self) -> str:
        """Get system uptime"""
        try:
            boot_time = psutil.boot_time()
            uptime_seconds = time.time() - boot_time
            uptime_delta = timedelta(seconds=uptime_seconds)
            return str(uptime_delta).split('.')[0]  # Remove microseconds
        except Exception:
            return "Unknown"
    
    def run_monitoring_loop(self) -> None:
        """Main monitoring loop"""
        self.running = True
        self.logger.info("Production monitoring started")
        
        with Live(console=console, refresh_per_second=1) as live:
            while self.running:
                try:
                    # Collect metrics
                    metrics = self.collect_system_metrics()
                    self.metrics_history.append(metrics)
                    
                    # Keep only last 100 metrics for memory efficiency
                    if len(self.metrics_history) > 100:
                        self.metrics_history = self.metrics_history[-100:]
                    
                    # Check for alerts
                    new_alerts = self.check_alerts(metrics)
                    
                    # Process new alerts
                    for alert in new_alerts:
                        self.alerts.append(alert)
                        self.logger.warning(f"ALERT: {alert.severity} - {alert.component}: {alert.message}")
                        
                        # Attempt recovery for critical alerts
                        if alert.severity == "CRITICAL" and alert.component not in self.recovery_attempts:
                            self.recovery_attempts[alert.component] = time.time()
                            recovery_success = self.attempt_recovery(alert)
                            
                            if recovery_success:
                                alert.resolved = True
                                alert.resolution_time = datetime.now().isoformat()
                                self.logger.info(f"Recovery successful for {alert.component}")
                    
                    # Save metrics
                    self.save_metrics(metrics)
                    
                    # Update dashboard
                    layout = self.create_monitoring_dashboard(metrics)
                    live.update(layout)
                    
                    # Sleep until next iteration
                    time.sleep(self.monitoring_interval)
                    
                except KeyboardInterrupt:
                    self.running = False
                    break
                except Exception as e:
                    self.logger.error(f"Monitoring loop error: {str(e)}")
                    time.sleep(5)  # Wait before retrying
        
        self.logger.info("Production monitoring stopped")
    
    def start_monitoring(self) -> None:
        """Start the monitoring system"""
        console.print(Panel.fit(
            "[bold cyan]🚀 NICEGOLD ENTERPRISE PRODUCTION MONITOR[/bold cyan]\n"
            "[dim]Starting real-time monitoring system...[/dim]",
            box=box.DOUBLE
        ))
        
        try:
            self.run_monitoring_loop()
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️ Monitoring interrupted by user[/yellow]")
        except Exception as e:
            console.print(f"\n[bold red]💥 Monitoring error: {str(e)}[/bold red]")
            self.logger.error(f"Fatal monitoring error: {str(e)}")

def main():
    """Main execution function"""
    monitor = ProductionMonitor()
    
    console.print("[bold green]🎯 Starting NICEGOLD Enterprise Production Monitor...[/bold green]")
    console.print("[dim]Press Ctrl+C to stop monitoring[/dim]\n")
    
    try:
        monitor.start_monitoring()
    except Exception as e:
        console.print(f"[bold red]❌ Failed to start monitoring: {str(e)}[/bold red]")
        sys.exit(1)

if __name__ == "__main__":
    main()
