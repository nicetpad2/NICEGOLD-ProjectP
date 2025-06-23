# 📦 Installation Guide

## 🎯 Overview
This guide covers complete installation of the Phiradon168 Enterprise Trading System with experiment tracking capabilities.

## 📋 System Requirements

### Minimum Requirements
- **OS**: Windows 10+, Ubuntu 18.04+, macOS 10.15+
- **Python**: 3.8-3.10 (recommended: 3.9)
- **RAM**: 8GB
- **Storage**: 10GB free space
- **Network**: Broadband internet connection

### Recommended for Production
- **OS**: Ubuntu 20.04 LTS or Windows Server 2019+
- **Python**: 3.9
- **CPU**: 8+ cores
- **RAM**: 32GB+
- **Storage**: 500GB SSD
- **Network**: 100+ Mbps

## 🚀 Quick Installation

### Option 1: Automated Setup (Recommended)
```bash
# Download and run automated installer
curl -sSL https://raw.githubusercontent.com/Phiradon168/setup.sh | bash

# Or for Windows PowerShell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/Phiradon168/setup.ps1" -OutFile "setup.ps1"
PowerShell -ExecutionPolicy Bypass -File setup.ps1
```

### Option 2: One-Command Setup
```bash
# Clone and setup in one command
git clone <repo-url> && cd Phiradon168 && python setup_new_environment.py
```

## 📋 Step-by-Step Installation

### Step 1: Prerequisites

#### Install Python
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.9 python3.9-venv python3.9-dev

# CentOS/RHEL
sudo yum install python39 python39-devel

# macOS (using Homebrew)
brew install python@3.9

# Windows (using Chocolatey)
choco install python --version=3.9.0
```

#### Install Git
```bash
# Ubuntu/Debian
sudo apt install git

# CentOS/RHEL
sudo yum install git

# macOS
brew install git

# Windows
choco install git
```

#### Install Build Tools
```bash
# Ubuntu/Debian
sudo apt install build-essential

# CentOS/RHEL
sudo yum groupinstall "Development Tools"

# macOS
xcode-select --install

# Windows
# Install Visual Studio Build Tools from Microsoft
```

### Step 2: Project Setup

#### Clone Repository
```bash
git clone <repo-url>
cd Phiradon168
```

#### Create Virtual Environment
```bash
# Create virtual environment
python3.9 -m venv .venv

# Activate virtual environment
# Linux/macOS
source .venv/bin/activate

# Windows
.\.venv\Scripts\activate

# Verify activation
which python  # Should show .venv path
python --version  # Should show Python 3.9.x
```

### Step 3: Dependencies Installation

#### Core Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Install core requirements
pip install -r requirements.txt

# Install tracking requirements
pip install -r tracking_requirements.txt

# Install development dependencies (optional)
pip install -r dev-requirements.txt
```

#### Verify Installation
```bash
# Check installed packages
pip list

# Verify key packages
python -c "import mlflow; print(f'MLflow: {mlflow.__version__}')"
python -c "import wandb; print(f'WandB: {wandb.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
```

### Step 4: System Configuration

#### Environment Variables
```bash
# Copy template
cp .env.example .env

# Edit configuration
nano .env  # or use your preferred editor
```

**Required Environment Variables:**
```bash
# Basic Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO
DATA_PATH=./data
MODEL_PATH=./models

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=nicegold_trading

# Security (Generate secure keys)
SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
ENCRYPTION_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")

# Optional: WandB Configuration
WANDB_API_KEY=your_wandb_api_key
WANDB_PROJECT=nicegold-enterprise
```

#### Directory Structure
```bash
# Create required directories
python -c "
import os
dirs = [
    'data', 'models', 'logs', 'artifacts', 'reports',
    'enterprise_tracking', 'enterprise_mlruns',
    'configs', 'scripts', 'notebooks', 'backups'
]
for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f'Created: {d}')
"
```

### Step 5: Initialize Tracking System

#### Basic Initialization
```bash
# Initialize tracking system
python init_tracking_system.py

# Verify installation
python tracking_cli.py --status
```

#### Advanced Initialization
```bash
# Initialize with specific providers
python init_tracking_system.py --providers mlflow,wandb,local

# Initialize for production
python init_tracking_system.py --production

# Initialize with custom config
python init_tracking_system.py --config custom_tracking_config.yaml
```

### Step 6: Verification

#### System Health Check
```bash
# Complete system check
python tracking_cli.py --health-check

# Test basic functionality
python tracking_examples.py

# Run integration test
python -c "
from tracking import EnterpriseTracker
tracker = EnterpriseTracker('test_install')
with tracker.start_run('verification'):
    tracker.log_params({'test': True})
    tracker.log_metrics({'status': 1.0})
print('✅ Installation verified successfully!')
"
```

#### Start Services
```bash
# Start MLflow UI
python tracking_cli.py --start-ui &

# Start monitoring (optional)
python tracking_integration.py --start-monitoring &

# Test web interfaces
curl http://localhost:5000  # MLflow UI
```

## 🐳 Docker Installation

### Quick Docker Setup
```bash
# Build image
docker build -t nicegold-enterprise .

# Run container
docker run -d \
  --name nicegold-trading \
  -p 8000:8000 \
  -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  nicegold-enterprise

# Check status
docker logs nicegold-trading
```

### Docker Compose Setup
```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Custom Docker Configuration
```dockerfile
# Dockerfile.custom
FROM nicegold-enterprise:latest

# Add custom configurations
COPY custom_config.yaml /app/config/
COPY custom_scripts/ /app/scripts/

# Set custom environment
ENV ENVIRONMENT=production
ENV TRACKING_PROVIDERS=mlflow,wandb

# Custom startup command
CMD ["python", "ProjectP.py", "--mode", "production"]
```

## ☁️ Cloud Installation

### Azure Deployment
```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login to Azure
az login

# Deploy using script
python cloud_deployment.py --provider azure \
  --resource-group nicegold-rg \
  --location eastus \
  --sku Standard_D4s_v3
```

### AWS Deployment
```bash
# Install AWS CLI
pip install awscli

# Configure AWS
aws configure

# Deploy to EC2
python cloud_deployment.py --provider aws \
  --region us-east-1 \
  --instance-type m5.xlarge \
  --key-name your-key-pair
```

### Google Cloud Deployment
```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash

# Authenticate
gcloud auth login

# Deploy to Compute Engine
python cloud_deployment.py --provider gcp \
  --project your-project-id \
  --zone us-central1-a \
  --machine-type e2-standard-4
```

## 🔧 Advanced Configuration

### Database Setup

#### PostgreSQL (Recommended for Production)
```bash
# Install PostgreSQL
sudo apt install postgresql postgresql-contrib

# Create database
sudo -u postgres psql
CREATE DATABASE nicegold_tracking;
CREATE USER nicegold WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE nicegold_tracking TO nicegold;
\q

# Update configuration
export MLFLOW_BACKEND_STORE_URI="postgresql://nicegold:secure_password@localhost/nicegold_tracking"
```

#### MySQL Alternative
```bash
# Install MySQL
sudo apt install mysql-server

# Create database
mysql -u root -p
CREATE DATABASE nicegold_tracking;
CREATE USER 'nicegold'@'localhost' IDENTIFIED BY 'secure_password';
GRANT ALL PRIVILEGES ON nicegold_tracking.* TO 'nicegold'@'localhost';
FLUSH PRIVILEGES;
EXIT;

# Update configuration
export MLFLOW_BACKEND_STORE_URI="mysql://nicegold:secure_password@localhost/nicegold_tracking"
```

### Storage Configuration

#### S3 Artifact Store
```bash
# Install boto3
pip install boto3

# Configure S3
export MLFLOW_S3_ENDPOINT_URL=https://s3.amazonaws.com
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export MLFLOW_ARTIFACT_ROOT=s3://your-bucket/artifacts
```

#### Azure Blob Storage
```bash
# Install Azure storage SDK
pip install azure-storage-blob

# Configure Azure
export AZURE_STORAGE_CONNECTION_STRING="your_connection_string"
export MLFLOW_ARTIFACT_ROOT=azure://your-container/artifacts
```

### Monitoring Setup

#### Prometheus & Grafana
```bash
# Install with Docker Compose
cat > monitoring-docker-compose.yml << EOF
version: '3.8'
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
EOF

docker-compose -f monitoring-docker-compose.yml up -d
```

## 🔐 Security Setup

### SSL/TLS Configuration
```bash
# Generate self-signed certificate for development
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# For production, use Let's Encrypt
sudo apt install certbot
sudo certbot certonly --standalone -d your-domain.com
```

### Authentication Setup
```bash
# Create admin user
python -c "
from tracking.auth import create_user
create_user('admin', 'secure_password', role='admin')
print('Admin user created')
"

# Configure OAuth (optional)
export OAUTH_CLIENT_ID=your_client_id
export OAUTH_CLIENT_SECRET=your_client_secret
export OAUTH_PROVIDER=google  # or github, azure, etc.
```

## 🧪 Testing Installation

### Run Test Suite
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/unit/ -v            # Unit tests
python -m pytest tests/integration/ -v     # Integration tests
python -m pytest tests/tracking/ -v        # Tracking system tests

# Run performance tests
python -m pytest tests/performance/ -v
```

### Load Testing
```bash
# Install locust for load testing
pip install locust

# Run load test
locust -f tests/load/test_api.py --host=http://localhost:8000
```

## 🔄 Migration from Existing System

### Export Existing Data
```bash
# Export from legacy MLflow
python migration/export_mlflow.py --source-uri sqlite:///old_mlflow.db

# Export from legacy system
python migration/export_legacy.py --source-path /path/to/old/system
```

### Import to New System
```bash
# Import experiments
python migration/import_experiments.py --data-file exported_data.json

# Validate migration
python migration/validate_migration.py
```

## 🛠️ Troubleshooting

### Common Issues

#### Permission Denied
```bash
# Fix permissions
sudo chown -R $USER:$USER .
chmod +x scripts/*.py
```

#### Port Already in Use
```bash
# Find process using port
lsof -i :5000
# or
netstat -tulpn | grep :5000

# Kill process
kill -9 <process_id>

# Use different port
export MLFLOW_PORT=5001
python tracking_cli.py --start-ui --port 5001
```

#### Memory Issues
```bash
# Enable memory optimization
export MEMORY_OPTIMIZATION=true
export OMP_NUM_THREADS=1

# Monitor memory usage
python -c "
import psutil
print(f'Available memory: {psutil.virtual_memory().available / 1024**3:.1f} GB')
"
```

#### Database Connection Issues
```bash
# Test database connection
python -c "
import sqlalchemy
engine = sqlalchemy.create_engine('$MLFLOW_BACKEND_STORE_URI')
try:
    engine.connect()
    print('✅ Database connection successful')
except Exception as e:
    print(f'❌ Database connection failed: {e}')
"
```

### Getting Help

#### Logs Location
```bash
# System logs
tail -f logs/system.log

# MLflow logs
tail -f logs/mlflow.log

# Application logs
tail -f logs/application.log
```

#### Diagnostic Commands
```bash
# System information
python tracking_cli.py --system-info

# Health check
python tracking_cli.py --health-check --verbose

# Configuration check
python tracking_cli.py --check-config
```

### Support Resources
- 📧 **Email**: support@nicegold.enterprise
- 💬 **Discord**: https://discord.gg/nicegold
- 📚 **Documentation**: https://docs.nicegold.enterprise
- 🐛 **Issues**: https://github.com/Phiradon168/issues

## ✅ Post-Installation Checklist

### Immediate Tasks
- [ ] Environment variables configured
- [ ] Virtual environment activated
- [ ] All dependencies installed
- [ ] Tracking system initialized
- [ ] Health check passed
- [ ] Basic functionality tested

### Security Tasks
- [ ] Strong passwords set
- [ ] SSL certificates configured (production)
- [ ] Authentication enabled
- [ ] Encryption keys generated
- [ ] Firewall rules configured

### Production Tasks
- [ ] Database configured
- [ ] Storage backend configured
- [ ] Monitoring enabled
- [ ] Backup strategy implemented
- [ ] Disaster recovery plan
- [ ] Performance monitoring

### Documentation Tasks
- [ ] Configuration documented
- [ ] User accounts created
- [ ] Training materials prepared
- [ ] Support procedures established

---

## 🎉 Congratulations!

Your Phiradon168 Enterprise Trading System is now installed and ready to use!

### Next Steps
1. **Read the [USAGE_QUICK_REFERENCE.md](USAGE_QUICK_REFERENCE.md)** for common commands
2. **Run your first experiment** with `python tracking_examples.py`
3. **Explore the MLflow UI** at http://localhost:5000
4. **Join our community** for support and updates

### Quick Start Command
```bash
# Run your first trading pipeline with tracking
python ProjectP.py --mode full_pipeline --track_experiments
```

**Happy Trading! 🚀📈**
