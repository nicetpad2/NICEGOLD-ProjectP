#!/usr/bin/env python3
"""
📋 NICEGOLD Enterprise System Status & Summary 📋
================================================

สคริปต์แสดงสถานะและสรุปของระบบ NICEGOLD Enterprise
พร้อมข้อมูลการใช้งานและคำแนะนำสำหรับผู้ดูแลระบบ

Features:
- System status overview
- File structure validation
- Service availability check
- Quick start guide
- Troubleshooting tips
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.tree import Tree
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None

class NICEGOLDSystemStatus:
    """System status and summary generator"""
    
    def __init__(self):
        self.project_root = Path(".")
        self.system_info = {
            "timestamp": datetime.now(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "files_status": {},
            "services_status": {},
            "system_stats": {},
            "recommendations": []
        }
    
    def generate_status_report(self) -> Dict:
        """Generate comprehensive system status report"""
        print("🔍 Generating NICEGOLD Enterprise system status...")
        
        # Check file structure
        self._check_file_structure()
        
        # Check services
        self._check_services_status()
        
        # Get system stats
        self._get_system_stats()
        
        # Generate recommendations
        self._generate_recommendations()
        
        # Display report
        self._display_status_report()
        
        return self.system_info
    
    def _check_file_structure(self):
        """Check project file structure"""
        required_files = {
            # Core system files
            "one_click_deploy.py": "One-click deployment script",
            "start_production_single_user.py": "Production system manager",
            "system_maintenance.py": "System maintenance & monitoring",
            "final_integration_live_test.py": "Integration testing",
            "system_demo.py": "System demonstration",
            
            # AI system files
            "ai_team_manager.py": "AI team management",
            "ai_orchestrator.py": "AI workflow orchestrator",
            "ai_assistant_brain.py": "AI assistant brain",
            
            # Authentication & API
            "src/single_user_auth.py": "Single-user authentication",
            "src/api.py": "FastAPI backend",
            
            # Configuration
            "config.yaml": "Main configuration",
            "ADMIN_GUIDE.md": "Administrator guide",
            "README.md": "Project documentation"
        }
        
        optional_files = {
            "dashboard_app.py": "Streamlit dashboard",
            "src/dashboard.py": "Dashboard module",
            "config/production.yaml": "Production configuration",
            ".env.production": "Production environment",
            "requirements.txt": "Python dependencies"
        }
        
        # Check required files
        missing_required = []
        existing_required = []
        
        for file_path, description in required_files.items():
            full_path = self.project_root / file_path
            if full_path.exists():
                existing_required.append((file_path, description))
            else:
                missing_required.append((file_path, description))
        
        # Check optional files
        existing_optional = []
        missing_optional = []
        
        for file_path, description in optional_files.items():
            full_path = self.project_root / file_path
            if full_path.exists():
                existing_optional.append((file_path, description))
            else:
                missing_optional.append((file_path, description))
        
        self.system_info["files_status"] = {
            "required_files": {
                "existing": existing_required,
                "missing": missing_required,
                "total": len(required_files),
                "completion_rate": len(existing_required) / len(required_files) * 100
            },
            "optional_files": {
                "existing": existing_optional,
                "missing": missing_optional,
                "total": len(optional_files),
                "completion_rate": len(existing_optional) / len(optional_files) * 100
            }
        }
    
    def _check_services_status(self):
        """Check service availability"""
        services = {
            "API Server": {"port": 8000, "pid_file": "run/api.pid"},
            "Dashboard": {"port": 8501, "pid_file": "run/dashboard.pid"},
            "Database": {"file": "database/production.db"},
        }
        
        service_status = {}
        
        for service_name, config in services.items():
            status = {"name": service_name, "status": "unknown", "details": ""}
            
            if "pid_file" in config:
                pid_file = self.project_root / config["pid_file"]
                if pid_file.exists():
                    try:
                        with open(pid_file, 'r') as f:
                            pid = int(f.read().strip())
                        
                        if DEPENDENCIES_AVAILABLE and psutil.pid_exists(pid):
                            status["status"] = "running"
                            status["details"] = f"PID: {pid}"
                        else:
                            status["status"] = "stopped"
                            status["details"] = "PID file exists but process not found"
                    except:
                        status["status"] = "error"
                        status["details"] = "Invalid PID file"
                else:
                    status["status"] = "stopped"
                    status["details"] = "No PID file found"
            
            elif "file" in config:
                file_path = self.project_root / config["file"]
                if file_path.exists():
                    status["status"] = "available"
                    status["details"] = f"Size: {file_path.stat().st_size} bytes"
                else:
                    status["status"] = "missing"
                    status["details"] = "Database file not found"
            
            service_status[service_name] = status
        
        self.system_info["services_status"] = service_status
    
    def _get_system_stats(self):
        """Get system statistics"""
        stats = {}
        
        if DEPENDENCIES_AVAILABLE:
            # CPU usage
            stats["cpu_usage"] = psutil.cpu_percent()
            
            # Memory usage
            memory = psutil.virtual_memory()
            stats["memory"] = {
                "total_gb": round(memory.total / 1024**3, 2),
                "used_percent": memory.percent,
                "available_gb": round(memory.available / 1024**3, 2)
            }
            
            # Disk usage
            disk = psutil.disk_usage('.')
            stats["disk"] = {
                "total_gb": round(disk.total / 1024**3, 2),
                "used_percent": round((disk.used / disk.total) * 100, 1),
                "free_gb": round(disk.free / 1024**3, 2)
            }
            
            # Process count
            stats["processes"] = len(psutil.pids())
        else:
            stats["error"] = "psutil not available"
        
        # Project statistics
        project_stats = {}
        
        # Count Python files
        python_files = list(self.project_root.rglob("*.py"))
        project_stats["python_files"] = len(python_files)
        
        # Count log files
        log_dir = self.project_root / "logs"
        if log_dir.exists():
            log_files = list(log_dir.rglob("*.log"))
            project_stats["log_files"] = len(log_files)
        else:
            project_stats["log_files"] = 0
        
        # Count configuration files
        config_files = list(self.project_root.rglob("*.yaml")) + list(self.project_root.rglob("*.yml"))
        project_stats["config_files"] = len(config_files)
        
        stats["project"] = project_stats
        self.system_info["system_stats"] = stats
    
    def _generate_recommendations(self):
        """Generate system recommendations"""
        recommendations = []
        
        # File structure recommendations
        files_status = self.system_info["files_status"]
        if files_status["required_files"]["completion_rate"] < 100:
            missing_count = len(files_status["required_files"]["missing"])
            recommendations.append(f"⚠️  {missing_count} required files are missing. Run deployment script to create them.")
        
        if files_status["optional_files"]["completion_rate"] < 50:
            recommendations.append("ℹ️  Consider creating optional configuration files for better functionality.")
        
        # Service recommendations
        services_status = self.system_info["services_status"]
        stopped_services = [name for name, status in services_status.items() if status["status"] == "stopped"]
        if stopped_services:
            recommendations.append(f"🚀 Start stopped services: {', '.join(stopped_services)}")
        
        error_services = [name for name, status in services_status.items() if status["status"] == "error"]
        if error_services:
            recommendations.append(f"🔧 Fix service errors: {', '.join(error_services)}")
        
        # System stats recommendations
        stats = self.system_info["system_stats"]
        if "cpu_usage" in stats and stats["cpu_usage"] > 80:
            recommendations.append("⚡ High CPU usage detected. Consider optimizing or scaling.")
        
        if "memory" in stats and stats["memory"]["used_percent"] > 85:
            recommendations.append("🧠 High memory usage detected. Consider adding more RAM.")
        
        if "disk" in stats and stats["disk"]["used_percent"] > 90:
            recommendations.append("💾 Low disk space. Clean up old files or expand storage.")
        
        # General recommendations
        if not (self.project_root / "database").exists():
            recommendations.append("💾 Create database directory for data storage.")
        
        if not (self.project_root / "logs").exists():
            recommendations.append("📝 Create logs directory for system logging.")
        
        if not (self.project_root / "backups").exists():
            recommendations.append("📦 Create backups directory for automated backups.")
        
        # If no issues found
        if not recommendations:
            recommendations.append("✅ System appears to be in good condition!")
        
        self.system_info["recommendations"] = recommendations
    
    def _display_status_report(self):
        """Display formatted status report"""
        if not RICH_AVAILABLE or not console:
            self._display_simple_report()
            return
        
        # Main header
        header_content = f"""
[bold blue]🚀 NICEGOLD Enterprise System Status[/bold blue]

[yellow]📅 Report Generated:[/yellow] [cyan]{self.system_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}[/cyan]
[yellow]🐍 Python Version:[/yellow] [cyan]{self.system_info['python_version']}[/cyan]
[yellow]📂 Project Root:[/yellow] [cyan]{self.project_root.absolute()}[/cyan]
        """
        
        console.print(Panel(header_content, title="📋 System Overview", border_style="blue"))
        
        # File structure status
        self._display_file_status()
        
        # Services status
        self._display_services_status()
        
        # System statistics
        self._display_system_stats()
        
        # Recommendations
        self._display_recommendations()
        
        # Quick start guide
        self._display_quick_start()
    
    def _display_simple_report(self):
        """Simple text-based report fallback"""
        print("\n" + "="*60)
        print("🚀 NICEGOLD Enterprise System Status")
        print("="*60)
        
        print(f"📅 Report Generated: {self.system_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🐍 Python Version: {self.system_info['python_version']}")
        
        # File structure
        files_status = self.system_info["files_status"]
        print(f"\n📁 File Structure:")
        print(f"   Required files: {len(files_status['required_files']['existing'])}/{files_status['required_files']['total']}")
        print(f"   Optional files: {len(files_status['optional_files']['existing'])}/{files_status['optional_files']['total']}")
        
        # Services
        print(f"\n🔧 Services:")
        for name, status in self.system_info["services_status"].items():
            print(f"   {name}: {status['status']}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        for rec in self.system_info["recommendations"]:
            print(f"   {rec}")
    
    def _display_file_status(self):
        """Display file structure status"""
        files_status = self.system_info["files_status"]
        
        # Create file status table
        file_table = Table(title="📁 File Structure Status")
        file_table.add_column("Category", style="cyan")
        file_table.add_column("Status", style="bold")
        file_table.add_column("Count", style="green")
        file_table.add_column("Completion", style="yellow")
        
        # Required files
        req_rate = files_status["required_files"]["completion_rate"]
        req_status = "✅ Complete" if req_rate == 100 else "⚠️ Incomplete"
        file_table.add_row(
            "Required Files",
            req_status,
            f"{len(files_status['required_files']['existing'])}/{files_status['required_files']['total']}",
            f"{req_rate:.1f}%"
        )
        
        # Optional files
        opt_rate = files_status["optional_files"]["completion_rate"]
        opt_status = "✅ Complete" if opt_rate == 100 else "ℹ️ Partial" if opt_rate > 0 else "❌ None"
        file_table.add_row(
            "Optional Files",
            opt_status,
            f"{len(files_status['optional_files']['existing'])}/{files_status['optional_files']['total']}",
            f"{opt_rate:.1f}%"
        )
        
        console.print(file_table)
        
        # Show missing files if any
        missing_required = files_status["required_files"]["missing"]
        if missing_required:
            missing_content = "\n".join([f"❌ {file}: {desc}" for file, desc in missing_required])
            console.print(Panel(
                missing_content,
                title="⚠️ Missing Required Files",
                border_style="red"
            ))
    
    def _display_services_status(self):
        """Display services status"""
        services_table = Table(title="🔧 Services Status")
        services_table.add_column("Service", style="cyan")
        services_table.add_column("Status", style="bold")
        services_table.add_column("Details", style="dim")
        
        for name, status in self.system_info["services_status"].items():
            if status["status"] == "running":
                status_display = Text("🟢 Running", style="green")
            elif status["status"] == "stopped":
                status_display = Text("🔴 Stopped", style="red")
            elif status["status"] == "available":
                status_display = Text("🟡 Available", style="yellow")
            elif status["status"] == "missing":
                status_display = Text("❌ Missing", style="red")
            else:
                status_display = Text("❓ Unknown", style="dim")
            
            services_table.add_row(name, status_display, status["details"])
        
        console.print(services_table)
    
    def _display_system_stats(self):
        """Display system statistics"""
        stats = self.system_info["system_stats"]
        
        if "error" in stats:
            console.print(Panel(
                f"⚠️ System stats unavailable: {stats['error']}",
                title="📊 System Statistics",
                border_style="yellow"
            ))
            return
        
        # System resources
        system_content = ""
        if "cpu_usage" in stats:
            cpu_color = "red" if stats["cpu_usage"] > 80 else "yellow" if stats["cpu_usage"] > 50 else "green"
            system_content += f"🖥️  CPU Usage: [{cpu_color}]{stats['cpu_usage']:.1f}%[/{cpu_color}]\n"
        
        if "memory" in stats:
            mem = stats["memory"]
            mem_color = "red" if mem["used_percent"] > 85 else "yellow" if mem["used_percent"] > 70 else "green"
            system_content += f"🧠 Memory: [{mem_color}]{mem['used_percent']:.1f}%[/{mem_color}] ({mem['available_gb']:.1f}GB available)\n"
        
        if "disk" in stats:
            disk = stats["disk"]
            disk_color = "red" if disk["used_percent"] > 90 else "yellow" if disk["used_percent"] > 80 else "green"
            system_content += f"💾 Disk: [{disk_color}]{disk['used_percent']:.1f}%[/{disk_color}] ({disk['free_gb']:.1f}GB free)\n"
        
        if "processes" in stats:
            system_content += f"🔢 Processes: [cyan]{stats['processes']}[/cyan]\n"
        
        # Project stats
        if "project" in stats:
            proj = stats["project"]
            system_content += f"\n[bold yellow]📊 Project Statistics:[/bold yellow]\n"
            system_content += f"🐍 Python Files: [cyan]{proj['python_files']}[/cyan]\n"
            system_content += f"📝 Log Files: [cyan]{proj['log_files']}[/cyan]\n"
            system_content += f"⚙️ Config Files: [cyan]{proj['config_files']}[/cyan]"
        
        console.print(Panel(system_content, title="📊 System Statistics", border_style="green"))
    
    def _display_recommendations(self):
        """Display recommendations"""
        if not self.system_info["recommendations"]:
            return
        
        rec_content = "\n".join([f"• {rec}" for rec in self.system_info["recommendations"]])
        
        console.print(Panel(
            rec_content,
            title="💡 Recommendations",
            border_style="blue"
        ))
    
    def _display_quick_start(self):
        """Display quick start guide"""
        quick_start_content = """
[bold green]🚀 Quick Start Commands:[/bold green]

[yellow]📦 First Time Setup:[/yellow]
[cyan]python one_click_deploy.py[/cyan]          # Complete system deployment

[yellow]🔧 System Management:[/yellow]
[cyan]./start_services.sh[/cyan]                 # Start all services
[cyan]./stop_services.sh[/cyan]                  # Stop all services
[cyan]python system_maintenance.py monitor[/cyan]      # Monitor system
[cyan]python system_maintenance.py health[/cyan]       # Health check

[yellow]🤖 AI Team:[/yellow]
[cyan]python ai_orchestrator.py[/cyan]           # AI orchestrator (recommended)
[cyan]python ai_team_manager.py[/cyan]           # AI team manager
[cyan]python ai_assistant_brain.py[/cyan]        # AI assistant

[yellow]🔐 Authentication:[/yellow]
[cyan]python src/single_user_auth.py[/cyan]      # Manage users

[yellow]🧪 Testing:[/yellow]
[cyan]python system_demo.py[/cyan]               # System demonstration
[cyan]python final_integration_live_test.py[/cyan]    # Integration test

[yellow]📚 Documentation:[/yellow]
[cyan]cat ADMIN_GUIDE.md[/cyan]                  # Administrator guide
[cyan]http://localhost:8000/docs[/cyan]          # API documentation (when running)
        """
        
        console.print(Panel(
            quick_start_content,
            title="📚 Quick Start Guide",
            border_style="bright_blue"
        ))

def main():
    """Main entry point"""
    print("🔍 NICEGOLD Enterprise System Status Check")
    print("=" * 50)
    
    if not DEPENDENCIES_AVAILABLE:
        print("⚠️  Note: Some dependencies not available. Install psutil and pyyaml for full system stats.")
    
    status_checker = NICEGOLDSystemStatus()
    report = status_checker.generate_status_report()
    
    # Save report to file
    report_file = Path("system_status_report.json")
    try:
        # Convert datetime to string for JSON serialization
        json_report = {
            **report,
            "timestamp": report["timestamp"].isoformat()
        }
        
        with open(report_file, 'w') as f:
            json.dump(json_report, f, indent=2)
        
        print(f"\n📄 Report saved to: {report_file}")
    except Exception as e:
        print(f"⚠️  Could not save report: {e}")
    
    return True

if __name__ == "__main__":
    main()
