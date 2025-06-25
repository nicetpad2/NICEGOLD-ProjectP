#!/usr/bin/env python3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
    from rich.prompt import Confirm, IntPrompt, Prompt
    from rich.table import Table
    from rich.text import Text
    from rich.tree import Tree
            from src.single_user_auth import auth_manager
from typing import Any, Dict, List, Optional, Union
                    import getpass
import hashlib
import json
import logging
import os
            import psutil
import secrets
import shutil
        import socket
import sqlite3
import subprocess
import sys
import time
import yaml
"""
🚀 NICEGOLD Enterprise Production Setup & Management System 🚀
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

ระบบติดตั้งและจัดการ production สำหรับ NICEGOLD Enterprise Trading Platform
พร้อมระบบ single - user authentication และการจัดการระบบแบบ enterprise - grade

Features:
- Single Admin User Control System
- Production Environment Setup
- Security Configuration
- Database Management
- Service Health Monitoring
- Automated Backup System
- SSL/TLS Configuration
- Log Management
- Performance Monitoring
"""


# Rich imports for beautiful output
try:
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️ Rich library not available. Install with: pip install rich")

console = Console() if RICH_AVAILABLE else None

class ProductionSetupManager:
    """
    Production setup and management system for NICEGOLD Enterprise
    """

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.config_dir = self.project_root / "config"
        self.logs_dir = self.project_root / "logs"
        self.data_dir = self.project_root / "data"
        self.database_dir = self.project_root / "database"
        self.backup_dir = self.project_root / "backups"
        self.run_dir = self.project_root / "run"
        self.ssl_dir = self.project_root / "ssl"

        # Create essential directories
        self._create_directories()

        # Setup logging
        self._setup_logging()

        # Load configuration
        self.config = self._load_configuration()

        # System info
        self.system_info = self._gather_system_info()

    def _create_directories(self):
        """Create essential production directories"""
        directories = [
            self.config_dir, 
            self.logs_dir, 
            self.data_dir, 
            self.database_dir, 
            self.backup_dir, 
            self.run_dir, 
            self.ssl_dir, 
            self.config_dir / "auth", 
            self.config_dir / "production", 
            self.logs_dir / "application", 
            self.logs_dir / "security", 
            self.logs_dir / "system", 
            self.logs_dir / "deployment"
        ]

        for directory in directories:
            directory.mkdir(parents = True, exist_ok = True)

    def _setup_logging(self):
        """Setup comprehensive logging system"""
        log_file = self.logs_dir / "deployment" / f"production_setup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level = logging.INFO, 
            format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
            handlers = [
                logging.FileHandler(log_file), 
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _load_configuration(self) -> Dict[str, Any]:
        """Load or create production configuration"""
        prod_config_file = self.config_dir / "production.yaml"

        if prod_config_file.exists():
            try:
                with open(prod_config_file, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                self.logger.error(f"Failed to load production config: {e}")

        # Create default production configuration
        default_config = {
            "application": {
                "name": "NICEGOLD Enterprise", 
                "version": "1.0.0", 
                "environment": "production", 
                "debug": False, 
                "single_user_mode": True
            }, 
            "security": {
                "auth_enabled": True, 
                "jwt_expiry_hours": 24, 
                "max_login_attempts": 5, 
                "lockout_duration_minutes": 30, 
                "session_timeout_hours": 8, 
                "password_min_length": 8, 
                "require_https": True, 
                "security_headers": True
            }, 
            "database": {
                "type": "sqlite", 
                "file": "database/production.db", 
                "backup_enabled": True, 
                "backup_interval_hours": 6
            }, 
            "api": {
                "host": "127.0.0.1", 
                "port": 8000, 
                "workers": 4, 
                "reload": False, 
                "access_log": True
            }, 
            "dashboard": {
                "enabled": True, 
                "host": "127.0.0.1", 
                "port": 8501, 
                "auto_refresh_seconds": 30
            }, 
            "monitoring": {
                "enabled": True, 
                "health_check_interval": 60, 
                "metrics_enabled": True, 
                "alerting_enabled": True
            }, 
            "logging": {
                "level": "INFO", 
                "max_size_mb": 100, 
                "backup_count": 5, 
                "rotation": "daily"
            }
        }

        # Save default configuration
        with open(prod_config_file, 'w') as f:
            yaml.dump(default_config, f, default_flow_style = False, sort_keys = False)

        return default_config

    def _gather_system_info(self) -> Dict[str, Any]:
        """Gather system information for production readiness"""
        try:

            return {
                "cpu_count": psutil.cpu_count(), 
                "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2), 
                "disk_free_gb": round(psutil.disk_usage('/').free / (1024**3), 2), 
                "python_version": sys.version.split()[0], 
                "platform": sys.platform, 
                "timestamp": datetime.now().isoformat()
            }
        except ImportError:
            return {
                "python_version": sys.version.split()[0], 
                "platform": sys.platform, 
                "timestamp": datetime.now().isoformat()
            }

    def run_complete_setup(self) -> bool:
        """Run complete production setup"""
        if RICH_AVAILABLE:
            console.print(Panel.fit(
                "[bold blue]🚀 NICEGOLD Enterprise Production Setup[/bold blue]\n"
                "[yellow]Automated Production Environment Configuration[/yellow]\n\n"
                "[green]This will setup:[/green]\n"
                "• Single User Authentication System\n"
                "• Production Database\n"
                "• Security Configuration\n"
                "• Monitoring & Logging\n"
                "• API & Dashboard Services\n"
                "• Backup System\n"
                "• SSL Configuration", 
                title = "Production Setup"
            ))
        else:
            print("🚀 NICEGOLD Enterprise Production Setup")
            print(" = " * 50)

        try:
            # Step 1: System Requirements Check
            if not self._check_system_requirements():
                return False

            # Step 2: Setup Authentication System
            if not self._setup_authentication_system():
                return False

            # Step 3: Initialize Production Database
            if not self._setup_production_database():
                return False

            # Step 4: Configure Security
            if not self._configure_security():
                return False

            # Step 5: Setup Services
            if not self._setup_services():
                return False

            # Step 6: Initialize Monitoring
            if not self._setup_monitoring():
                return False

            # Step 7: Create Management Scripts
            if not self._create_management_scripts():
                return False

            # Step 8: Final Validation
            if not self._validate_installation():
                return False

            self._show_setup_completion()
            return True

        except Exception as e:
            self.logger.error(f"Production setup failed: {e}")
            if RICH_AVAILABLE:
                console.print(f"[bold red]❌ Setup failed: {e}[/bold red]")
            return False

    def _check_system_requirements(self) -> bool:
        """Check system requirements for production deployment"""
        if RICH_AVAILABLE:
            console.print("\n[bold blue]📋 Checking System Requirements[/bold blue]")

        checks = [
            ("Python Version", self._check_python_version), 
            ("System Resources", self._check_system_resources), 
            ("Required Packages", self._check_required_packages), 
            ("Permissions", self._check_permissions), 
            ("Network", self._check_network)
        ]

        all_passed = True

        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(), 
                TextColumn("[progress.description]{task.description}"), 
                console = console
            ) as progress:
                for check_name, check_func in checks:
                    task = progress.add_task(f"Checking {check_name}...", total = None)
                    result = check_func()
                    progress.update(task, completed = True)

                    if result:
                        console.print(f"[green]✅ {check_name}: OK[/green]")
                    else:
                        console.print(f"[red]❌ {check_name}: Failed[/red]")
                        all_passed = False
        else:
            for check_name, check_func in checks:
                print(f"Checking {check_name}...")
                result = check_func()
                if result:
                    print(f"✅ {check_name}: OK")
                else:
                    print(f"❌ {check_name}: Failed")
                    all_passed = False

        return all_passed

    def _check_python_version(self) -> bool:
        """Check Python version compatibility"""
        try:
            major, minor = sys.version_info[:2]
            return major == 3 and minor >= 8
        except:
            return False

    def _check_system_resources(self) -> bool:
        """Check system resources (RAM, disk space)"""
        try:

            # Check RAM (minimum 4GB)
            memory_gb = psutil.virtual_memory().total / (1024**3)
            if memory_gb < 4:
                self.logger.warning(f"Low memory: {memory_gb:.1f}GB (recommended: 4GB + )")

            # Check disk space (minimum 10GB)
            disk_free_gb = psutil.disk_usage('/').free / (1024**3)
            if disk_free_gb < 10:
                self.logger.error(f"Insufficient disk space: {disk_free_gb:.1f}GB (required: 10GB + )")
                return False

            return True
        except ImportError:
            self.logger.warning("psutil not available - skipping resource check")
            return True

    def _check_required_packages(self) -> bool:
        """Check if required packages are installed"""
        required_packages = [
            ('fastapi', 'fastapi'), 
            ('uvicorn', 'uvicorn'), 
            ('streamlit', 'streamlit'), 
            ('pandas', 'pandas'), 
            ('numpy', 'numpy'), 
            ('sklearn', 'scikit - learn'), 
            ('jwt', 'pyjwt'), 
            ('yaml', 'pyyaml'), 
            ('rich', 'rich')
        ]

        missing_packages = []

        for import_name, package_name in required_packages:
            try:
                __import__(import_name)
            except ImportError:
                missing_packages.append(package_name)

        if missing_packages:
            self.logger.error(f"Missing packages: {', '.join(missing_packages)}")
            if RICH_AVAILABLE:
                console.print(f"[red]Install missing packages: pip install {' '.join(missing_packages)}[/red]")
            return False

        return True

    def _check_permissions(self) -> bool:
        """Check file system permissions"""
        try:
            # Test write permission in project directory
            test_file = self.project_root / ".permission_test"
            test_file.write_text("test")
            test_file.unlink()
            return True
        except Exception as e:
            self.logger.error(f"Permission check failed: {e}")
            return False

    def _check_network(self) -> bool:
        """Check network connectivity and ports"""

        # Check if required ports are available
        required_ports = [8000, 8501]  # API and Dashboard

        for port in required_ports:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('127.0.0.1', port))
            except OSError:
                self.logger.warning(f"Port {port} may be in use")

        return True

    def _setup_authentication_system(self) -> bool:
        """Setup single user authentication system"""
        if RICH_AVAILABLE:
            console.print("\n[bold blue]🔐 Setting up Single User Authentication[/bold blue]")

        try:
            # Import authentication system

            # Check if user already exists
            status = auth_manager.get_system_status()
            if not status["user_configured"]:
                # Setup admin user
                if RICH_AVAILABLE:
                    console.print("[yellow]Setting up admin user...[/yellow]")
                    username = Prompt.ask("Enter admin username", default = "admin")
                    password = Prompt.ask("Enter admin password", password = True)
                    password_confirm = Prompt.ask("Confirm admin password", password = True)
                else:
                    username = input("Enter admin username [admin]: ").strip() or "admin"
                    password = getpass.getpass("Enter admin password: ")
                    password_confirm = getpass.getpass("Confirm admin password: ")

                if password != password_confirm:
                    if RICH_AVAILABLE:
                        console.print("[red]❌ Passwords do not match[/red]")
                    return False

                if len(password) < 8:
                    if RICH_AVAILABLE:
                        console.print("[red]❌ Password must be at least 8 characters[/red]")
                    return False

                # Create admin user
                if not auth_manager.setup_user(username, password):
                    if RICH_AVAILABLE:
                        console.print("[red]❌ Failed to create admin user[/red]")
                    return False

                if RICH_AVAILABLE:
                    console.print(f"[green]✅ Admin user '{username}' created successfully[/green]")
            else:
                if RICH_AVAILABLE:
                    console.print(f"[green]✅ Admin user already configured: {status['username']}[/green]")

            return True

        except Exception as e:
            self.logger.error(f"Authentication setup failed: {e}")
            return False

    def _setup_production_database(self) -> bool:
        """Setup production database"""
        if RICH_AVAILABLE:
            console.print("\n[bold blue]🗄️ Setting up Production Database[/bold blue]")

        try:
            db_path = self.database_dir / "production.db"

            # Create production database
            with sqlite3.connect(db_path) as conn:
                # Create essential tables
                cursor = conn.cursor()

                # Users table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, 
                        username TEXT UNIQUE NOT NULL, 
                        password_hash TEXT NOT NULL, 
                        salt TEXT NOT NULL, 
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
                        last_login TIMESTAMP, 
                        login_count INTEGER DEFAULT 0, 
                        is_active BOOLEAN DEFAULT 1
                    )
                """)

                # Sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, 
                        token TEXT UNIQUE NOT NULL, 
                        username TEXT NOT NULL, 
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
                        expires_at TIMESTAMP NOT NULL, 
                        ip_address TEXT, 
                        user_agent TEXT, 
                        is_active BOOLEAN DEFAULT 1
                    )
                """)

                # Trades table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, 
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
                        symbol TEXT NOT NULL, 
                        side TEXT NOT NULL, 
                        quantity REAL NOT NULL, 
                        price REAL NOT NULL, 
                        profit_loss REAL, 
                        status TEXT DEFAULT 'pending', 
                        strategy TEXT, 
                        model_version TEXT
                    )
                """)

                # Models table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS models (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, 
                        name TEXT UNIQUE NOT NULL, 
                        version TEXT NOT NULL, 
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
                        accuracy REAL, 
                        auc_score REAL, 
                        file_path TEXT, 
                        is_active BOOLEAN DEFAULT 0, 
                        performance_metrics TEXT
                    )
                """)

                # System logs table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, 
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
                        level TEXT NOT NULL, 
                        component TEXT NOT NULL, 
                        message TEXT NOT NULL, 
                        details TEXT
                    )
                """)

                conn.commit()

            if RICH_AVAILABLE:
                console.print(f"[green]✅ Production database created: {db_path}[/green]")

            return True

        except Exception as e:
            self.logger.error(f"Database setup failed: {e}")
            return False

    def _configure_security(self) -> bool:
        """Configure security settings"""
        if RICH_AVAILABLE:
            console.print("\n[bold blue]🔒 Configuring Security[/bold blue]")

        try:
            # Create security configuration
            security_config = {
                "authentication": {
                    "enabled": True, 
                    "method": "jwt", 
                    "token_expiry_hours": 24, 
                    "max_login_attempts": 5, 
                    "lockout_duration_minutes": 30
                }, 
                "session": {
                    "timeout_hours": 8, 
                    "secure_cookies": True, 
                    "httponly_cookies": True
                }, 
                "password_policy": {
                    "min_length": 8, 
                    "require_uppercase": True, 
                    "require_lowercase": True, 
                    "require_numbers": True, 
                    "require_special_chars": False
                }, 
                "api_security": {
                    "rate_limiting": True, 
                    "requests_per_minute": 60, 
                    "cors_enabled": True, 
                    "allowed_origins": ["http://localhost:8501"]
                }, 
                "ssl": {
                    "enabled": False, 
                    "cert_file": "ssl/cert.pem", 
                    "key_file": "ssl/key.pem"
                }
            }

            # Save security configuration
            security_config_file = self.config_dir / "security.yaml"
            with open(security_config_file, 'w') as f:
                yaml.dump(security_config, f, default_flow_style = False)

            # Create environment variables file
            env_file = self.project_root / ".env.production"
            env_content = f"""# NICEGOLD Enterprise Production Environment
ENVIRONMENT = production
DEBUG = false
SECRET_KEY = {secrets.token_urlsafe(32)}
JWT_SECRET = {secrets.token_urlsafe(32)}
DATABASE_URL = sqlite:///database/production.db
LOG_LEVEL = INFO
API_HOST = 127.0.0.1
API_PORT = 8000
DASHBOARD_HOST = 127.0.0.1
DASHBOARD_PORT = 8501
"""

            with open(env_file, 'w') as f:
                f.write(env_content)

            if RICH_AVAILABLE:
                console.print("[green]✅ Security configuration completed[/green]")

            return True

        except Exception as e:
            self.logger.error(f"Security configuration failed: {e}")
            return False

    def _setup_services(self) -> bool:
        """Setup application services"""
        if RICH_AVAILABLE:
            console.print("\n[bold blue]⚙️ Setting up Services[/bold blue]")

        try:
            # Create service configurations
            services_config = {
                "api": {
                    "enabled": True, 
                    "host": "127.0.0.1", 
                    "port": 8000, 
                    "workers": 4, 
                    "reload": False, 
                    "log_level": "info"
                }, 
                "dashboard": {
                    "enabled": True, 
                    "host": "127.0.0.1", 
                    "port": 8501, 
                    "auto_refresh": True, 
                    "refresh_interval": 30
                }, 
                "scheduler": {
                    "enabled": True, 
                    "job_interval_minutes": 5, 
                    "max_concurrent_jobs": 2
                }
            }

            # Save services configuration
            services_config_file = self.config_dir / "services.yaml"
            with open(services_config_file, 'w') as f:
                yaml.dump(services_config, f, default_flow_style = False)

            if RICH_AVAILABLE:
                console.print("[green]✅ Services configuration completed[/green]")

            return True

        except Exception as e:
            self.logger.error(f"Services setup failed: {e}")
            return False

    def _setup_monitoring(self) -> bool:
        """Setup monitoring and logging"""
        if RICH_AVAILABLE:
            console.print("\n[bold blue]📊 Setting up Monitoring[/bold blue]")

        try:
            # Create monitoring configuration
            monitoring_config = {
                "health_checks": {
                    "enabled": True, 
                    "interval_seconds": 60, 
                    "endpoints": [
                        "http://127.0.0.1:8000/health", 
                        "http://127.0.0.1:8501"
                    ]
                }, 
                "metrics": {
                    "enabled": True, 
                    "collection_interval": 30, 
                    "retention_days": 30
                }, 
                "alerts": {
                    "enabled": True, 
                    "email_notifications": False, 
                    "thresholds": {
                        "cpu_usage": 80, 
                        "memory_usage": 80, 
                        "disk_usage": 90, 
                        "error_rate": 5
                    }
                }, 
                "logging": {
                    "level": "INFO", 
                    "max_file_size_mb": 100, 
                    "backup_count": 5, 
                    "log_rotation": "daily"
                }
            }

            # Save monitoring configuration
            monitoring_config_file = self.config_dir / "monitoring.yaml"
            with open(monitoring_config_file, 'w') as f:
                yaml.dump(monitoring_config, f, default_flow_style = False)

            # Create log configuration
            log_config = {
                "version": 1, 
                "disable_existing_loggers": False, 
                "formatters": {
                    "standard": {
                        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    }, 
                    "detailed": {
                        "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
                    }
                }, 
                "handlers": {
                    "file": {
                        "class": "logging.handlers.RotatingFileHandler", 
                        "filename": str(self.logs_dir / "application" / "nicegold.log"), 
                        "maxBytes": 100 * 1024 * 1024,  # 100MB
                        "backupCount": 5, 
                        "formatter": "detailed"
                    }, 
                    "console": {
                        "class": "logging.StreamHandler", 
                        "formatter": "standard"
                    }
                }, 
                "root": {
                    "level": "INFO", 
                    "handlers": ["file", "console"]
                }
            }

            # Save log configuration
            log_config_file = self.config_dir / "logging.yaml"
            with open(log_config_file, 'w') as f:
                yaml.dump(log_config, f, default_flow_style = False)

            if RICH_AVAILABLE:
                console.print("[green]✅ Monitoring and logging configured[/green]")

            return True

        except Exception as e:
            self.logger.error(f"Monitoring setup failed: {e}")
            return False

    def _create_management_scripts(self) -> bool:
        """Create management and utility scripts"""
        if RICH_AVAILABLE:
            console.print("\n[bold blue]📝 Creating Management Scripts[/bold blue]")

        try:
            scripts_dir = self.project_root / "scripts"
            scripts_dir.mkdir(exist_ok = True)

            # Create start script
            start_script = scripts_dir / "start.sh"
            start_script_content = """#!/bin/bash
# NICEGOLD Enterprise Start Script

echo "🚀 Starting NICEGOLD Enterprise..."

# Load environment
if [ -f .env.production ]; then
    export $(cat .env.production | grep -v '^#' | xargs)
fi

# Start API server
echo "Starting API server..."
nohup python -m uvicorn src.api:app - - host $API_HOST - - port $API_PORT - - workers 4 > logs/api.log 2>&1 &
echo $! > run/api.pid

# Start Dashboard
echo "Starting Dashboard..."
nohup streamlit run single_user_dashboard.py - - server.address $DASHBOARD_HOST - - server.port $DASHBOARD_PORT > logs/dashboard.log 2>&1 &
echo $! > run/dashboard.pid

echo "✅ NICEGOLD Enterprise started successfully!"
echo "API: http://$API_HOST:$API_PORT"
echo "Dashboard: http://$DASHBOARD_HOST:$DASHBOARD_PORT"
"""

            with open(start_script, 'w') as f:
                f.write(start_script_content)
            start_script.chmod(0o755)

            # Create stop script
            stop_script = scripts_dir / "stop.sh"
            stop_script_content = """#!/bin/bash
# NICEGOLD Enterprise Stop Script

echo "🛑 Stopping NICEGOLD Enterprise..."

# Stop API server
if [ -f run/api.pid ]; then
    PID = $(cat run/api.pid)
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        echo "API server stopped"
    fi
    rm -f run/api.pid
fi

# Stop Dashboard
if [ -f run/dashboard.pid ]; then
    PID = $(cat run/dashboard.pid)
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        echo "Dashboard stopped"
    fi
    rm -f run/dashboard.pid
fi

echo "✅ NICEGOLD Enterprise stopped successfully!"
"""

            with open(stop_script, 'w') as f:
                f.write(stop_script_content)
            stop_script.chmod(0o755)

            # Create status script
            status_script = scripts_dir / "status.sh"
            status_script_content = """#!/bin/bash
# NICEGOLD Enterprise Status Script

echo "📊 NICEGOLD Enterprise Status"
echo " =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = "

# Check API server
if [ -f run/api.pid ]; then
    PID = $(cat run/api.pid)
    if kill -0 $PID 2>/dev/null; then
        echo "✅ API Server: Running (PID: $PID)"
    else
        echo "❌ API Server: Not running"
        rm -f run/api.pid
    fi
else
    echo "❌ API Server: Not running"
fi

# Check Dashboard
if [ -f run/dashboard.pid ]; then
    PID = $(cat run/dashboard.pid)
    if kill -0 $PID 2>/dev/null; then
        echo "✅ Dashboard: Running (PID: $PID)"
    else
        echo "❌ Dashboard: Not running"
        rm -f run/dashboard.pid
    fi
else
    echo "❌ Dashboard: Not running"
fi

# Check database
if [ -f database/production.db ]; then
    echo "✅ Database: Available"
else
    echo "❌ Database: Not found"
fi

echo ""
echo "System Information:"
echo " =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = "
df -h . | tail -1 | awk '{print "Disk Usage: " $5 " (" $3 " used, " $4 " available)"}'
free -h | grep Mem | awk '{print "Memory Usage: " $3 "/" $2}'
"""

            with open(status_script, 'w') as f:
                f.write(status_script_content)
            status_script.chmod(0o755)

            # Create backup script
            backup_script = scripts_dir / "backup.sh"
            backup_script_content = """#!/bin/bash
# NICEGOLD Enterprise Backup Script

BACKUP_DIR = "backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "🗄️ Creating NICEGOLD Enterprise backup..."

# Backup database
if [ -f database/production.db ]; then
    cp database/production.db "$BACKUP_DIR/"
    echo "✅ Database backed up"
fi

# Backup configuration
cp -r config "$BACKUP_DIR/"
echo "✅ Configuration backed up"

# Backup logs (last 7 days)
find logs -name "*.log" -mtime -7 -exec cp {} "$BACKUP_DIR/" \;
echo "✅ Recent logs backed up"

# Create archive
tar -czf "$BACKUP_DIR.tar.gz" -C backups $(basename "$BACKUP_DIR")
rm -rf "$BACKUP_DIR"

echo "✅ Backup completed: $BACKUP_DIR.tar.gz"

# Cleanup old backups (keep last 10)
cd backups
ls -t *.tar.gz | tail -n +11 | xargs -r rm
"""

            with open(backup_script, 'w') as f:
                f.write(backup_script_content)
            backup_script.chmod(0o755)

            if RICH_AVAILABLE:
                console.print("[green]✅ Management scripts created[/green]")

            return True

        except Exception as e:
            self.logger.error(f"Script creation failed: {e}")
            return False

    def _validate_installation(self) -> bool:
        """Validate production installation"""
        if RICH_AVAILABLE:
            console.print("\n[bold blue]✅ Validating Installation[/bold blue]")

        try:
            # Check essential files
            essential_files = [
                self.config_dir / "production.yaml", 
                self.config_dir / "security.yaml", 
                self.config_dir / "services.yaml", 
                self.config_dir / "monitoring.yaml", 
                self.database_dir / "production.db", 
                self.project_root / ".env.production"
            ]

            missing_files = []
            for file_path in essential_files:
                if not file_path.exists():
                    missing_files.append(str(file_path))

            if missing_files:
                if RICH_AVAILABLE:
                    console.print(f"[red]❌ Missing files: {', '.join(missing_files)}[/red]")
                return False

            # Test authentication system
            status = auth_manager.get_system_status()
            if not status["user_configured"]:
                if RICH_AVAILABLE:
                    console.print("[red]❌ Authentication system not configured[/red]")
                return False

            # Test database connection
            db_path = self.database_dir / "production.db"
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'")
                table_count = cursor.fetchone()[0]
                if table_count < 5:  # Should have at least 5 tables
                    if RICH_AVAILABLE:
                        console.print("[red]❌ Database schema incomplete[/red]")
                    return False

            if RICH_AVAILABLE:
                console.print("[green]✅ Installation validation passed[/green]")

            return True

        except Exception as e:
            self.logger.error(f"Installation validation failed: {e}")
            return False

    def _show_setup_completion(self):
        """Show setup completion summary"""
        if RICH_AVAILABLE:
            console.print("\n" + " = "*60)
            console.print(Panel.fit(
                "[bold green]🎉 NICEGOLD Enterprise Production Setup Complete! 🎉[/bold green]\n\n"
                "[cyan]Your single - user trading platform is ready for production use.[/cyan]\n\n"
                "[yellow]Quick Start Commands:[/yellow]\n"
                "• Start services: [bold cyan]./scripts/start.sh[/bold cyan]\n"
                "• Check status: [bold cyan]./scripts/status.sh[/bold cyan]\n"
                "• Stop services: [bold cyan]./scripts/stop.sh[/bold cyan]\n"
                "• Create backup: [bold cyan]./scripts/backup.sh[/bold cyan]\n\n"
                "[yellow]Access Points:[/yellow]\n"
                "• API Server: [bold cyan]http://127.0.0.1:8000[/bold cyan]\n"
                "• Dashboard: [bold cyan]http://127.0.0.1:8501[/bold cyan]\n"
                "• Health Check: [bold cyan]http://127.0.0.1:8000/health[/bold cyan]\n\n"
                "[yellow]Important Files:[/yellow]\n"
                "• Main Config: [bold cyan]config/production.yaml[/bold cyan]\n"
                "• Environment: [bold cyan].env.production[/bold cyan]\n"
                "• Database: [bold cyan]database/production.db[/bold cyan]\n"
                "• Logs: [bold cyan]logs/[/bold cyan]", 
                title = "🚀 Setup Complete"
            ))

            # Show system status
            console.print("\n[bold blue]📊 System Information[/bold blue]")

            table = Table(show_header = True, header_style = "bold blue")
            table.add_column("Component", style = "cyan")
            table.add_column("Status", style = "green")
            table.add_column("Details")

            auth_status = auth_manager.get_system_status()

            table.add_row("Authentication", "✅ Configured", f"User: {auth_status['username']}")
            table.add_row("Database", "✅ Ready", "SQLite Production DB")
            table.add_row("Security", "✅ Configured", "JWT + Session Management")
            table.add_row("Monitoring", "✅ Enabled", "Health checks & Logging")
            table.add_row("Backup System", "✅ Ready", "Automated backups available")

            console.print(table)

            console.print("\n[bold red]🔒 Security Notes:[/bold red]")
            console.print("• Change default passwords before production use")
            console.print("• Review security configuration in config/security.yaml")
            console.print("• Setup firewall rules for your environment")
            console.print("• Enable HTTPS for production deployment")
            console.print("• Regularly backup your system")

        else:
            print("\n" + " = "*60)
            print("🎉 NICEGOLD Enterprise Production Setup Complete! 🎉")
            print("\nQuick Start Commands:")
            print("• Start services: ./scripts/start.sh")
            print("• Check status: ./scripts/status.sh")
            print("• Stop services: ./scripts/stop.sh")
            print("• Create backup: ./scripts/backup.sh")
            print("\nAccess Points:")
            print("• API Server: http://127.0.0.1:8000")
            print("• Dashboard: http://127.0.0.1:8501")

    def show_system_status(self):
        """Show current system status"""
        if RICH_AVAILABLE:
            console.print(Panel.fit(
                "[bold blue]📊 NICEGOLD Enterprise System Status[/bold blue]", 
                title = "System Status"
            ))

        try:
            # Check authentication system
            auth_status = auth_manager.get_system_status()

            # Check database
            db_path = self.database_dir / "production.db"
            db_status = "Available" if db_path.exists() else "Not Found"

            # Check services (placeholder - would need actual service checking)
            api_status = "Unknown"
            dashboard_status = "Unknown"

            if RICH_AVAILABLE:
                table = Table(show_header = True, header_style = "bold blue")
                table.add_column("Component", style = "cyan")
                table.add_column("Status")
                table.add_column("Details")

                # Authentication
                auth_icon = "✅" if auth_status["user_configured"] else "❌"
                table.add_row("Authentication", f"{auth_icon} {('Configured' if auth_status['user_configured'] else 'Not Configured')}", 
                             f"User: {auth_status['username']}")

                # Database
                db_icon = "✅" if db_status == "Available" else "❌"
                table.add_row("Database", f"{db_icon} {db_status}", str(db_path))

                # Services
                table.add_row("API Server", f"❓ {api_status}", "Port 8000")
                table.add_row("Dashboard", f"❓ {dashboard_status}", "Port 8501")

                console.print(table)

                # Show authentication details
                console.print(f"\n[bold blue]🔐 Authentication Details[/bold blue]")
                auth_table = Table(show_header = True, header_style = "bold blue")
                auth_table.add_column("Property", style = "cyan")
                auth_table.add_column("Value")

                for key, value in auth_status.items():
                    auth_table.add_row(key.replace('_', ' ').title(), str(value))

                console.print(auth_table)
            else:
                print("📊 NICEGOLD Enterprise System Status")
                print(" = "*50)
                print(f"Authentication: {'✅ Configured' if auth_status['user_configured'] else '❌ Not Configured'}")
                print(f"Database: {'✅ Available' if db_status == 'Available' else '❌ Not Found'}")
                print(f"API Server: ❓ {api_status}")
                print(f"Dashboard: ❓ {dashboard_status}")

        except Exception as e:
            if RICH_AVAILABLE:
                console.print(f"[red]❌ Error checking system status: {e}[/red]")
            else:
                print(f"❌ Error checking system status: {e}")

def main():
    """Main setup function"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        setup_manager = ProductionSetupManager()

        if command == "setup":
            # Run complete production setup
            success = setup_manager.run_complete_setup()
            sys.exit(0 if success else 1)

        elif command == "status":
            # Show system status
            setup_manager.show_system_status()

        else:
            print(f"Unknown command: {command}")
            print("Available commands: setup, status")
            sys.exit(1)
    else:
        print("NICEGOLD Enterprise Production Setup")
        print("Usage:")
        print("  python production_setup.py setup   - Run complete production setup")
        print("  python production_setup.py status  - Show system status")

if __name__ == "__main__":
    main()