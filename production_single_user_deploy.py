#!/usr/bin/env python3
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
            from src.single_user_auth import auth_manager
from typing import Any, Dict, List, Optional
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
import yaml
"""
Production Deployment System for Single User NICEGOLD Trading Platform
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

ระบบ deployment ระดับ production พร้อมระบบ authentication คนเดียว
และการจัดการระบบแบบ enterprise - grade
"""


console = Console()

class ProductionDeploymentManager:
    """
    Production deployment manager for single user trading system
    """

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.config_dir = self.project_root / "config"
        self.logs_dir = self.project_root / "logs"
        self.data_dir = self.project_root / "data"

        # Create directories
        for directory in [self.config_dir, self.logs_dir, self.data_dir]:
            directory.mkdir(parents = True, exist_ok = True)

        # Setup logging
        self._setup_logging()

        # Deployment configuration
        self.deployment_config = {
            "project_name": "nicegold - enterprise", 
            "version": "1.0.0", 
            "environment": "production", 
            "single_user_mode": True, 
            "auto_ssl": True, 
            "backup_enabled": True, 
            "monitoring_enabled": True
        }

    def _setup_logging(self):
        """Setup deployment logging"""
        log_file = self.logs_dir / f"deployment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level = logging.INFO, 
            format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
            handlers = [
                logging.FileHandler(log_file), 
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def deploy_single_user_production(self) -> bool:
        """
        Deploy complete single user production system
        """
        console.print(Panel.fit(
            "[bold green]🚀 NICEGOLD Enterprise Production Deployment[/bold green]\n"
            "[yellow]Single User Mode - Production Ready[/yellow]", 
            title = "Production Deployment"
        ))

        try:
            # Step 1: Pre - deployment checks
            if not self._pre_deployment_checks():
                return False

            # Step 2: Setup authentication system
            if not self._setup_single_user_auth():
                return False

            # Step 3: Configure production environment
            if not self._configure_production_environment():
                return False

            # Step 4: Setup database and storage
            if not self._setup_database_storage():
                return False

            # Step 5: Deploy services
            if not self._deploy_core_services():
                return False

            # Step 6: Setup monitoring and logging
            if not self._setup_monitoring():
                return False

            # Step 7: Configure security and backup
            if not self._setup_security_backup():
                return False

            # Step 8: Final validation
            if not self._validate_deployment():
                return False

            self._show_deployment_summary()
            return True

        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            console.print(f"[bold red]❌ Deployment failed: {e}[/bold red]")
            return False

    def _pre_deployment_checks(self) -> bool:
        """Pre - deployment system checks"""
        console.print("\n[bold blue]📋 Pre - deployment Checks[/bold blue]")

        checks = [
            ("Python Version", self._check_python_version), 
            ("Required Packages", self._check_required_packages), 
            ("System Resources", self._check_system_resources), 
            ("Permissions", self._check_permissions), 
            ("Network Connectivity", self._check_network)
        ]

        with Progress(
            SpinnerColumn(), 
            TextColumn("[progress.description]{task.description}"), 
            console = console
        ) as progress:

            for check_name, check_func in checks:
                task = progress.add_task(f"Checking {check_name}...", total = None)

                try:
                    result = check_func()
                    if result:
                        console.print(f"✅ {check_name}: [green]PASSED[/green]")
                    else:
                        console.print(f"❌ {check_name}: [red]FAILED[/red]")
                        return False
                except Exception as e:
                    console.print(f"❌ {check_name}: [red]ERROR - {e}[/red]")
                    return False

                progress.remove_task(task)

        return True

    def _check_python_version(self) -> bool:
        """Check Python version compatibility"""
        version = sys.version_info
        return version.major == 3 and version.minor >= 8

    def _check_required_packages(self) -> bool:
        """Check if required packages are installed"""
        required_packages = [
            "fastapi", "uvicorn", "sqlalchemy", "alembic", 
            "redis", "celery", "prometheus_client", "streamlit", 
            "pandas", "numpy", "scikit - learn", "mlflow"
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            console.print(f"[red]Missing packages: {', '.join(missing_packages)}[/red]")
            return False

        return True

    def _check_system_resources(self) -> bool:
        """Check system resources (RAM, disk space)"""

        # Check RAM (minimum 4GB)
        memory = psutil.virtual_memory()
        if memory.total < 4 * 1024 * 1024 * 1024:  # 4GB
            console.print("[red]Insufficient RAM: minimum 4GB required[/red]")
            return False

        # Check disk space (minimum 10GB)
        disk = psutil.disk_usage(str(self.project_root))
        if disk.free < 10 * 1024 * 1024 * 1024:  # 10GB
            console.print("[red]Insufficient disk space: minimum 10GB required[/red]")
            return False

        return True

    def _check_permissions(self) -> bool:
        """Check file system permissions"""
        try:
            # Test write permissions
            test_file = self.project_root / ".test_write"
            test_file.write_text("test")
            test_file.unlink()
            return True
        except Exception:
            return False

    def _check_network(self) -> bool:
        """Check network connectivity"""
        try:
            socket.create_connection(("8.8.8.8", 53), timeout = 5)
            return True
        except Exception:
            return False

    def _setup_single_user_auth(self) -> bool:
        """Setup single user authentication system"""
        console.print("\n[bold blue]🔐 Setting up Single User Authentication[/bold blue]")

        try:
            # Copy authentication module
            auth_source = self.project_root / "src" / "single_user_auth.py"
            if not auth_source.exists():
                console.print("[red]Authentication module not found[/red]")
                return False

            # Initialize authentication system

            # Check if user already exists
            status = auth_manager.get_system_status()
            if not status["user_configured"]:
                console.print("[yellow]No admin user configured. Setting up admin user...[/yellow]")

                # Get admin credentials
                username = Prompt.ask("Enter admin username", default = "admin")
                password = Prompt.ask("Enter admin password", password = True)

                if len(password) < 8:
                    console.print("[red]Password must be at least 8 characters[/red]")
                    return False

                # Create admin user
                if not auth_manager.setup_user(username, password):
                    console.print("[red]Failed to create admin user[/red]")
                    return False

                console.print(f"[green]✅ Admin user '{username}' created successfully[/green]")
            else:
                console.print(f"[green]✅ Admin user already configured: {status['username']}[/green]")

            return True

        except Exception as e:
            console.print(f"[red]Authentication setup failed: {e}[/red]")
            return False

    def _configure_production_environment(self) -> bool:
        """Configure production environment variables and settings"""
        console.print("\n[bold blue]⚙️ Configuring Production Environment[/bold blue]")

        try:
            # Create production configuration
            production_config = {
                "environment": "production", 
                "debug": False, 
                "single_user_mode": True, 

                # API Configuration
                "api": {
                    "host": "0.0.0.0", 
                    "port": 8000, 
                    "workers": 4, 
                    "enable_docs": False,  # Disable docs in production
                    "cors_origins": ["https://localhost", "https://127.0.0.1"], 
                    "rate_limit": "100/minute"
                }, 

                # Database Configuration
                "database": {
                    "url": "sqlite:///production.db", 
                    "echo": False, 
                    "pool_size": 10, 
                    "max_overflow": 20
                }, 

                # Security Configuration
                "security": {
                    "jwt_secret_key": self._generate_secret_key(), 
                    "jwt_expire_hours": 24, 
                    "session_timeout_hours": 8, 
                    "max_login_attempts": 5, 
                    "lockout_duration_minutes": 30, 
                    "enable_https": True, 
                    "ssl_cert_path": "/etc/ssl/certs/nicegold.crt", 
                    "ssl_key_path": "/etc/ssl/private/nicegold.key"
                }, 

                # Trading Configuration
                "trading": {
                    "max_position_size": 1000000,  # $1M
                    "risk_limit": 0.02,  # 2% max risk per trade
                    "emergency_stop_enabled": True, 
                    "trading_hours": {
                        "start": "09:00", 
                        "end": "17:00", 
                        "timezone": "UTC"
                    }
                }, 

                # ML Configuration
                "ml": {
                    "model_update_interval": 3600,  # 1 hour
                    "prediction_cache_ttl": 300,  # 5 minutes
                    "feature_store_enabled": True, 
                    "auto_retrain": True
                }, 

                # Monitoring Configuration
                "monitoring": {
                    "enabled": True, 
                    "prometheus_port": 9090, 
                    "health_check_interval": 30, 
                    "alert_thresholds": {
                        "cpu_percent": 80, 
                        "memory_percent": 85, 
                        "disk_percent": 90, 
                        "error_rate": 0.05
                    }
                }, 

                # Logging Configuration
                "logging": {
                    "level": "INFO", 
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
                    "max_file_size": "100MB", 
                    "backup_count": 10, 
                    "audit_enabled": True
                }, 

                # Backup Configuration
                "backup": {
                    "enabled": True, 
                    "schedule": "0 2 * * *",  # Daily at 2 AM
                    "retention_days": 30, 
                    "encrypt_backups": True
                }
            }

            # Save production configuration
            config_file = self.config_dir / "production.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(production_config, f, default_flow_style = False, indent = 2)

            # Create environment file
            env_file = self.project_root / ".env.production"
            env_content = f"""# NICEGOLD Enterprise Production Environment
ENVIRONMENT = production
DEBUG = false

# API Configuration
API_HOST = 0.0.0.0
API_PORT = 8000
API_WORKERS = 4

# Database
DATABASE_URL = sqlite:///production.db

# Security
JWT_SECRET_KEY = {production_config['security']['jwt_secret_key']}
JWT_EXPIRE_HOURS = 24

# Trading
MAX_POSITION_SIZE = 1000000
RISK_LIMIT = 0.02

# Monitoring
PROMETHEUS_PORT = 9090
HEALTH_CHECK_INTERVAL = 30

# Logging
LOG_LEVEL = INFO
AUDIT_ENABLED = true

# Generated on {datetime.now().isoformat()}
"""

            with open(env_file, 'w') as f:
                f.write(env_content)

            console.print("[green]✅ Production environment configured[/green]")
            return True

        except Exception as e:
            console.print(f"[red]Environment configuration failed: {e}[/red]")
            return False

    def _generate_secret_key(self) -> str:
        """Generate cryptographically secure secret key"""
        return secrets.token_urlsafe(64)

    def _setup_database_storage(self) -> bool:
        """Setup database and storage systems"""
        console.print("\n[bold blue]🗄️ Setting up Database and Storage[/bold blue]")

        try:
            # Create database directory
            db_dir = self.project_root / "database"
            db_dir.mkdir(exist_ok = True)

            # Create storage directories
            storage_dirs = [
                "data/market_data", 
                "data/processed", 
                "models/trained", 
                "models/artifacts", 
                "logs/application", 
                "logs/trading", 
                "logs/security", 
                "backups/database", 
                "backups/models", 
                "cache/predictions", 
                "cache/features"
            ]

            for storage_dir in storage_dirs:
                (self.project_root / storage_dir).mkdir(parents = True, exist_ok = True)

            # Initialize database schema
            self._initialize_database_schema()

            console.print("[green]✅ Database and storage setup complete[/green]")
            return True

        except Exception as e:
            console.print(f"[red]Database setup failed: {e}[/red]")
            return False

    def _initialize_database_schema(self):
        """Initialize database schema for production"""
        schema_sql = """
        -- Users table (single user system)
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY, 
            username VARCHAR(50) UNIQUE NOT NULL, 
            password_hash VARCHAR(255) NOT NULL, 
            salt VARCHAR(64) NOT NULL, 
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
            last_login TIMESTAMP, 
            login_count INTEGER DEFAULT 0, 
            is_active BOOLEAN DEFAULT TRUE
        );

        -- Sessions table
        CREATE TABLE IF NOT EXISTS user_sessions (
            token VARCHAR(255) PRIMARY KEY, 
            username VARCHAR(50) NOT NULL, 
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
            expires_at TIMESTAMP NOT NULL, 
            ip_address VARCHAR(45), 
            user_agent TEXT, 
            is_active BOOLEAN DEFAULT TRUE
        );

        -- Trading positions
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY, 
            symbol VARCHAR(20) NOT NULL, 
            side VARCHAR(10) NOT NULL, 
            size DECIMAL(15, 8) NOT NULL, 
            entry_price DECIMAL(15, 8) NOT NULL, 
            current_price DECIMAL(15, 8), 
            pnl DECIMAL(15, 8), 
            opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
            closed_at TIMESTAMP, 
            status VARCHAR(20) DEFAULT 'open'
        );

        -- Risk events
        CREATE TABLE IF NOT EXISTS risk_events (
            id INTEGER PRIMARY KEY, 
            event_type VARCHAR(50) NOT NULL, 
            severity VARCHAR(20) NOT NULL, 
            message TEXT NOT NULL, 
            data JSON, 
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
            resolved_at TIMESTAMP, 
            status VARCHAR(20) DEFAULT 'active'
        );

        -- Model registry
        CREATE TABLE IF NOT EXISTS model_registry (
            id INTEGER PRIMARY KEY, 
            name VARCHAR(100) NOT NULL, 
            version VARCHAR(20) NOT NULL, 
            algorithm VARCHAR(50) NOT NULL, 
            performance_metrics JSON, 
            trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
            deployed_at TIMESTAMP, 
            status VARCHAR(20) DEFAULT 'trained', 
            file_path VARCHAR(500), 
            metadata JSON
        );

        -- Predictions log
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY, 
            model_id INTEGER, 
            input_features JSON, 
            prediction DECIMAL(15, 8), 
            confidence DECIMAL(5, 4), 
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
            actual_outcome DECIMAL(15, 8), 
            FOREIGN KEY (model_id) REFERENCES model_registry (id)
        );

        -- Audit log
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY, 
            username VARCHAR(50), 
            action VARCHAR(100) NOT NULL, 
            resource VARCHAR(100), 
            ip_address VARCHAR(45), 
            user_agent TEXT, 
            request_data JSON, 
            response_data JSON, 
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
            status_code INTEGER
        );

        -- System configuration
        CREATE TABLE IF NOT EXISTS system_config (
            key VARCHAR(100) PRIMARY KEY, 
            value TEXT NOT NULL, 
            description TEXT, 
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
            updated_by VARCHAR(50)
        );
        """

        # Execute schema creation
        db_path = self.project_root / "database" / "production.db"

        with sqlite3.connect(db_path) as conn:
            conn.executescript(schema_sql)
            conn.commit()

    def _deploy_core_services(self) -> bool:
        """Deploy core application services"""
        console.print("\n[bold blue]🚀 Deploying Core Services[/bold blue]")

        try:
            # Create service configuration files
            self._create_systemd_services()
            self._create_nginx_config()
            self._create_docker_compose()

            console.print("[green]✅ Core services deployed[/green]")
            return True

        except Exception as e:
            console.print(f"[red]Service deployment failed: {e}[/red]")
            return False

    def _create_systemd_services(self):
        """Create systemd service files"""
        service_dir = self.project_root / "deploy" / "systemd"
        service_dir.mkdir(parents = True, exist_ok = True)

        # Main API service
        api_service = f"""[Unit]
Description = NICEGOLD Enterprise API
After = network.target

[Service]
Type = exec
User = nicegold
Group = nicegold
WorkingDirectory = {self.project_root}
Environment = PATH = {self.project_root}/.venv/bin
ExecStart = {self.project_root}/.venv/bin/uvicorn src.api:app - - host 0.0.0.0 - - port 8000 - - workers 4
Restart = always
RestartSec = 3

[Install]
WantedBy = multi - user.target
"""

        # Monitoring service
        monitoring_service = f"""[Unit]
Description = NICEGOLD Monitoring Service
After = network.target

[Service]
Type = exec
User = nicegold
Group = nicegold
WorkingDirectory = {self.project_root}
Environment = PATH = {self.project_root}/.venv/bin
ExecStart = {self.project_root}/.venv/bin/python src/monitoring_service.py
Restart = always
RestartSec = 3

[Install]
WantedBy = multi - user.target
"""

        with open(service_dir / "nicegold - api.service", 'w') as f:
            f.write(api_service)

        with open(service_dir / "nicegold - monitoring.service", 'w') as f:
            f.write(monitoring_service)

    def _create_nginx_config(self):
        """Create nginx configuration"""
        nginx_dir = self.project_root / "deploy" / "nginx"
        nginx_dir.mkdir(parents = True, exist_ok = True)

        nginx_config = """
server {
    listen 80;
    server_name localhost;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name localhost;

    # SSL Configuration
    ssl_certificate /etc/ssl/certs/nicegold.crt;
    ssl_certificate_key /etc/ssl/private/nicegold.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE - RSA - AES256 - GCM - SHA512:DHE - RSA - AES256 - GCM - SHA512:ECDHE - RSA - AES256 - GCM - SHA384:DHE - RSA - AES256 - GCM - SHA384;
    ssl_prefer_server_ciphers off;

    # Security Headers
    add_header X - Frame - Options DENY;
    add_header X - Content - Type - Options nosniff;
    add_header X - XSS - Protection "1; mode = block";
    add_header Strict - Transport - Security "max - age = 63072000; includeSubDomains; preload";

    # Rate Limiting
    limit_req_zone $binary_remote_addr zone = api:10m rate = 10r/s;

    # API Proxy
    location /api/ {
        limit_req zone = api burst = 20 nodelay;
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X - Real - IP $remote_addr;
        proxy_set_header X - Forwarded - For $proxy_add_x_forwarded_for;
        proxy_set_header X - Forwarded - Proto $scheme;
    }

    # Dashboard
    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_set_header Host $host;
        proxy_set_header X - Real - IP $remote_addr;
        proxy_set_header X - Forwarded - For $proxy_add_x_forwarded_for;
        proxy_set_header X - Forwarded - Proto $scheme;

        # WebSocket support for Streamlit
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Monitoring (restricted access)
    location /metrics {
        allow 127.0.0.1;
        deny all;
        proxy_pass http://127.0.0.1:9090;
    }
}
"""

        with open(nginx_dir / "nicegold.conf", 'w') as f:
            f.write(nginx_config)

    def _create_docker_compose(self):
        """Create Docker Compose configuration"""
        docker_dir = self.project_root / "deploy" / "docker"
        docker_dir.mkdir(parents = True, exist_ok = True)

        docker_compose = """version: '3.8'

services:
  nicegold - api:
    build:
      context: ../..
      dockerfile: Dockerfile
    container_name: nicegold - api
    restart: unless - stopped
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT = production
      - DATABASE_URL = sqlite:///data/production.db
    volumes:
      - ../../data:/app/data
      - ../../logs:/app/logs
      - ../../models:/app/models
    depends_on:
      - redis
    networks:
      - nicegold - network

  redis:
    image: redis:7 - alpine
    container_name: nicegold - redis
    restart: unless - stopped
    ports:
      - "6379:6379"
    volumes:
      - redis - data:/data
    networks:
      - nicegold - network

  nginx:
    image: nginx:alpine
    container_name: nicegold - nginx
    restart: unless - stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nicegold.conf:/etc/nginx/conf.d/default.conf
      - ../../ssl:/etc/ssl
    depends_on:
      - nicegold - api
    networks:
      - nicegold - network

  prometheus:
    image: prom/prometheus:latest
    container_name: nicegold - prometheus
    restart: unless - stopped
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus - data:/prometheus
    networks:
      - nicegold - network

networks:
  nicegold - network:
    driver: bridge

volumes:
  redis - data:
  prometheus - data:
"""

        with open(docker_dir / "docker - compose.yml", 'w') as f:
            f.write(docker_compose)

    def _setup_monitoring(self) -> bool:
        """Setup monitoring and alerting"""
        console.print("\n[bold blue]📊 Setting up Monitoring[/bold blue]")

        try:
            # Create monitoring configuration
            monitoring_dir = self.project_root / "deploy" / "monitoring"
            monitoring_dir.mkdir(parents = True, exist_ok = True)

            # Prometheus configuration
            prometheus_config = """
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'nicegold - api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'node - exporter'
    static_configs:
      - targets: ['localhost:9100']
"""

            # Alert rules
            alert_rules = """
groups:
  - name: nicegold_alerts
    rules:
      - alert: HighCPUUsage
        expr: cpu_usage_percent > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage is above 80% for more than 5 minutes"

      - alert: HighMemoryUsage
        expr: memory_usage_percent > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is above 85% for more than 5 minutes"

      - alert: APIErrorRate
        expr: error_rate > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High API error rate"
          description: "API error rate is above 5% for more than 2 minutes"
"""

            with open(monitoring_dir / "prometheus.yml", 'w') as f:
                f.write(prometheus_config)

            with open(monitoring_dir / "alert_rules.yml", 'w') as f:
                f.write(alert_rules)

            console.print("[green]✅ Monitoring setup complete[/green]")
            return True

        except Exception as e:
            console.print(f"[red]Monitoring setup failed: {e}[/red]")
            return False

    def _setup_security_backup(self) -> bool:
        """Setup security measures and backup system"""
        console.print("\n[bold blue]🔒 Setting up Security and Backup[/bold blue]")

        try:
            # Create SSL certificates (self - signed for development)
            ssl_dir = self.project_root / "ssl"
            ssl_dir.mkdir(exist_ok = True)

            # Generate self - signed certificate
            self._generate_ssl_certificate(ssl_dir)

            # Create backup script
            self._create_backup_script()

            # Setup security hardening
            self._setup_security_hardening()

            console.print("[green]✅ Security and backup setup complete[/green]")
            return True

        except Exception as e:
            console.print(f"[red]Security setup failed: {e}[/red]")
            return False

    def _generate_ssl_certificate(self, ssl_dir: Path):
        """Generate self - signed SSL certificate"""
        cert_file = ssl_dir / "nicegold.crt"
        key_file = ssl_dir / "nicegold.key"

        if not cert_file.exists() or not key_file.exists():
            cmd = [
                "openssl", "req", " - x509", " - newkey", "rsa:4096", 
                " - keyout", str(key_file), " - out", str(cert_file), 
                " - days", "365", " - nodes", 
                " - subj", "/C = TH/ST = Bangkok/L = Bangkok/O = NICEGOLD/CN = localhost"
            ]

            try:
                subprocess.run(cmd, check = True, capture_output = True)
                # Set secure permissions
                os.chmod(key_file, 0o600)
                os.chmod(cert_file, 0o644)
            except subprocess.CalledProcessError:
                console.print("[yellow]⚠️ OpenSSL not found. SSL certificate not generated.[/yellow]")

    def _create_backup_script(self):
        """Create automated backup script"""
        backup_dir = self.project_root / "scripts"
        backup_dir.mkdir(exist_ok = True)

        backup_script = f"""#!/bin/bash
# NICEGOLD Enterprise Backup Script

BACKUP_DIR = "{self.project_root}/backups"
DATE = $(date +%Y%m%d_%H%M%S)
BACKUP_NAME = "nicegold_backup_$DATE"

echo "Starting backup: $BACKUP_NAME"

# Create backup directory
mkdir -p "$BACKUP_DIR/$BACKUP_NAME"

# Backup database
cp "{self.project_root}/database/production.db" "$BACKUP_DIR/$BACKUP_NAME/"

# Backup configuration
cp -r "{self.project_root}/config" "$BACKUP_DIR/$BACKUP_NAME/"

# Backup models
cp -r "{self.project_root}/models" "$BACKUP_DIR/$BACKUP_NAME/"

# Backup logs (last 7 days)
find "{self.project_root}/logs" -name "*.log" -mtime -7 -exec cp {{}} "$BACKUP_DIR/$BACKUP_NAME/" \\;

# Create compressed archive
cd "$BACKUP_DIR"
tar -czf "$BACKUP_NAME.tar.gz" "$BACKUP_NAME"
rm -rf "$BACKUP_NAME"

# Remove old backups (keep 30 days)
find "$BACKUP_DIR" -name "nicegold_backup_*.tar.gz" -mtime +30 -delete

echo "Backup completed: $BACKUP_NAME.tar.gz"
"""

        backup_script_file = backup_dir / "backup.sh"
        with open(backup_script_file, 'w') as f:
            f.write(backup_script)

        # Make executable
        os.chmod(backup_script_file, 0o755)

    def _setup_security_hardening(self):
        """Setup security hardening measures"""
        security_dir = self.project_root / "config" / "security"
        security_dir.mkdir(parents = True, exist_ok = True)

        # Create security policy
        security_policy = {
            "password_policy": {
                "min_length": 8, 
                "require_uppercase": True, 
                "require_lowercase": True, 
                "require_numbers": True, 
                "require_special_chars": True
            }, 
            "session_policy": {
                "timeout_hours": 8, 
                "max_concurrent_sessions": 1, 
                "require_ip_validation": True
            }, 
            "api_security": {
                "rate_limit": "100/minute", 
                "require_https": True, 
                "enable_cors": False, 
                "allowed_origins": ["https://localhost"]
            }, 
            "audit_policy": {
                "log_all_requests": True, 
                "log_sensitive_data": False, 
                "retention_days": 90
            }
        }

        with open(security_dir / "security_policy.yaml", 'w') as f:
            yaml.dump(security_policy, f, default_flow_style = False, indent = 2)

    def _validate_deployment(self) -> bool:
        """Validate the deployment"""
        console.print("\n[bold blue]✅ Validating Deployment[/bold blue]")

        validations = [
            ("Authentication System", self._validate_auth_system), 
            ("Database Connection", self._validate_database), 
            ("Configuration Files", self._validate_configuration), 
            ("Service Files", self._validate_services), 
            ("Security Setup", self._validate_security)
        ]

        all_passed = True

        for validation_name, validation_func in validations:
            try:
                result = validation_func()
                if result:
                    console.print(f"✅ {validation_name}: [green]PASSED[/green]")
                else:
                    console.print(f"❌ {validation_name}: [red]FAILED[/red]")
                    all_passed = False
            except Exception as e:
                console.print(f"❌ {validation_name}: [red]ERROR - {e}[/red]")
                all_passed = False

        return all_passed

    def _validate_auth_system(self) -> bool:
        """Validate authentication system"""
        try:
            status = auth_manager.get_system_status()
            return status["user_configured"]
        except Exception:
            return False

    def _validate_database(self) -> bool:
        """Validate database connection"""
        try:
            db_path = self.project_root / "database" / "production.db"
            if not db_path.exists():
                return False

            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table';")
                tables = [row[0] for row in cursor.fetchall()]
                required_tables = ['users', 'user_sessions', 'positions', 'model_registry']
                return all(table in tables for table in required_tables)
        except Exception:
            return False

    def _validate_configuration(self) -> bool:
        """Validate configuration files"""
        required_configs = [
            "config/production.yaml", 
            ".env.production", 
            "config/security/security_policy.yaml"
        ]

        return all((self.project_root / config).exists() for config in required_configs)

    def _validate_services(self) -> bool:
        """Validate service files"""
        service_files = [
            "deploy/systemd/nicegold - api.service", 
            "deploy/nginx/nicegold.conf", 
            "deploy/docker/docker - compose.yml"
        ]

        return all((self.project_root / service).exists() for service in service_files)

    def _validate_security(self) -> bool:
        """Validate security setup"""
        security_files = [
            "config/auth/jwt_secret.key", 
            "ssl/nicegold.crt", 
            "ssl/nicegold.key", 
            "scripts/backup.sh"
        ]

        return all((self.project_root / security).exists() for security in security_files)

    def _show_deployment_summary(self):
        """Show deployment summary and next steps"""
        console.print("\n" + " = "*60)
        console.print(Panel.fit(
            "[bold green]🎉 DEPLOYMENT SUCCESSFUL![/bold green]\n"
            "[yellow]NICEGOLD Enterprise Production System Ready[/yellow]", 
            title = "🚀 Success"
        ))

        # Show access information
        table = Table(title = "🔗 Access Information")
        table.add_column("Service", style = "cyan")
        table.add_column("URL", style = "green")
        table.add_column("Description", style = "yellow")

        table.add_row("API", "https://localhost:8000", "Main API endpoint")
        table.add_row("Dashboard", "https://localhost:8501", "Streamlit dashboard")
        table.add_row("Monitoring", "http://localhost:9090", "Prometheus metrics")
        table.add_row("Documentation", "https://localhost:8000/docs", "API documentation (if enabled)")

        console.print(table)

        # Show next steps
        console.print("\n[bold blue]📋 Next Steps:[/bold blue]")
        console.print("1. 🔑 Login with your admin credentials")
        console.print("2. 📊 Configure trading parameters in the dashboard")
        console.print("3. 🔄 Start the services:")
        console.print("   • [cyan]./scripts/start_production.sh[/cyan]")
        console.print("   • [cyan]docker - compose -f deploy/docker/docker - compose.yml up -d[/cyan]")
        console.print("4. 📈 Monitor system performance via dashboard")
        console.print("5. 🔒 Setup SSL certificates for production use")

        # Show management commands
        console.print("\n[bold blue]🛠️ Management Commands:[/bold blue]")
        console.print("• Start services: [cyan]systemctl start nicegold - api[/cyan]")
        console.print("• Check status: [cyan]systemctl status nicegold - api[/cyan]")
        console.print("• View logs: [cyan]journalctl -u nicegold - api -f[/cyan]")
        console.print("• Backup system: [cyan]./scripts/backup.sh[/cyan]")

        # Show security notes
        console.print("\n[bold red]🔒 Security Notes:[/bold red]")
        console.print("• Change default SSL certificates for production")
        console.print("• Review and update security policy")
        console.print("• Setup firewall rules")
        console.print("• Enable automatic security updates")
        console.print("• Regularly backup the system")

        console.print("\n[bold green]✨ NICEGOLD Enterprise is ready for production use![/bold green]")

def main():
    """Main deployment function"""
    console.print(Panel.fit(
        "[bold blue]🚀 NICEGOLD Enterprise Production Deployment[/bold blue]\n"
        "[yellow]Single User Trading Platform[/yellow]", 
        title = "Production Deployment"
    ))

    # Confirm deployment
    if not Confirm.ask("\n[yellow]⚠️ This will setup NICEGOLD for production use. Continue?[/yellow]"):
        console.print("[red]Deployment cancelled[/red]")
        return False

    # Create deployment manager
    deployment_manager = ProductionDeploymentManager()

    # Run deployment
    success = deployment_manager.deploy_single_user_production()

    if success:
        console.print("\n[bold green]🎉 Production deployment completed successfully![/bold green]")
        return True
    else:
        console.print("\n[bold red]❌ Production deployment failed![/bold red]")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)