# 🚀 Quick Usage Reference

## One-Line Commands

### Setup & Installation
```bash
# Complete setup in one command
python setup_new_environment.py

# Quick test
python tracking_examples.py
```

### Trading Operations
```bash
# Basic trading pipeline
python ProjectP.py --mode full_pipeline

# Backtest with tracking
python ProjectP.py --mode backtest --track_experiments

# Walk-forward validation
python ProjectP.py --mode wfv --all

# Real-time trading
python ProjectP.py --mode realtime --enable_monitoring
```

### Experiment Tracking
```bash
# System status
python tracking_cli.py --status

# Start MLflow UI
python tracking_cli.py --start-ui

# List experiments
python tracking_cli.py --list-experiments

# Compare experiments  
python tracking_cli.py --compare exp1,exp2,exp3

# Deploy model
python tracking_cli.py --deploy model_name 1
```

### Monitoring & Health
```bash
# Health check
python tracking_cli.py --health-check

# Start monitoring
python tracking_integration.py --start-monitoring

# View dashboard
python dashboard_app.py

# Production status
python production_monitor.py --status
```

### Data Management
```bash
# Export experiment
python tracking_cli.py --export --experiment exp1

# Import data
python tracking_cli.py --import --file data.json

# Backup system
python tracking_cli.py --backup

# Migration
python quick_migration.py --export --output backup.zip
```

## Common Workflows

### 1. Development Workflow
```bash
# Setup
python init_tracking_system.py --dev

# Develop & test
python ProjectP.py --mode full_pipeline --debug

# Track experiments
python tracking_examples.py

# Review results
python tracking_cli.py --list-experiments
```

### 2. Production Deployment
```bash
# Setup production
python init_tracking_system.py --production

# Deploy
python cloud_deployment.py --provider azure

# Monitor
python tracking_integration.py --start-monitoring

# Health check
python tracking_cli.py --health-check
```

### 3. Migration Workflow
```bash
# Export from old system
python quick_migration.py --export --output old_system.zip

# Setup new environment
python setup_new_environment.py

# Import to new system
python quick_migration.py --import --file old_system.zip

# Validate
python tracking_cli.py --status
```

## Troubleshooting

### Quick Fixes
```bash
# Reset system
python tracking_cli.py --reset

# Fix permissions
python setup_tracking.py --fix-permissions

# Clear cache
python tracking_cli.py --clear-cache

# Repair database
python tracking_cli.py --repair-db
```

### Common Issues
```bash
# MLflow not starting
python tracking_cli.py --start-ui --port 5001

# WandB auth issues
wandb logout && wandb login

# Memory issues
export MEMORY_OPTIMIZATION=true

# Data drift
python tracking_cli.py --data-quality-report
```

## Configuration Templates

### Minimal Config
```yaml
# tracking_config.yaml
mlflow:
  tracking_uri: "http://localhost:5000"
local:
  base_path: "./tracking"
monitoring:
  enable_alerts: false
```

### Production Config
```yaml
# tracking_config.yaml
mlflow:
  tracking_uri: "https://your-mlflow-server.com"
wandb:
  project: "production-trading"
security:
  encrypt_artifacts: true
monitoring:
  enable_alerts: true
  check_interval: 60
```

## Environment Variables

### Development
```bash
export ENVIRONMENT=development
export LOG_LEVEL=DEBUG
export TRACKING_PROVIDERS=mlflow,local
```

### Production
```bash
export ENVIRONMENT=production
export LOG_LEVEL=INFO
export TRACKING_PROVIDERS=mlflow,wandb,local
export ENABLE_MONITORING=true
export ENCRYPT_ARTIFACTS=true
```

## API Quick Reference

### Python API
```python
from tracking import EnterpriseTracker

# Basic usage
tracker = EnterpriseTracker("my_experiment")
with tracker.start_run("test_run"):
    tracker.log_params({"lr": 0.01})
    tracker.log_metrics({"accuracy": 0.85})
    tracker.log_model(model, "my_model")
```

### CLI API
```bash
# Most common commands
tracking_cli.py --status                    # System status
tracking_cli.py --start-ui                  # Start UI
tracking_cli.py --list-experiments          # List experiments
tracking_cli.py --deploy model_name 1       # Deploy model
tracking_cli.py --health-check              # Health check
```

## Performance Tips

### Speed Optimization
```bash
# Use memory optimization
export MEMORY_OPTIMIZATION=true

# Enable parallel processing
export PARALLEL_PROCESSING=true

# Cache features
export CACHE_FEATURES=true

# Limit log verbosity
export LOG_LEVEL=WARNING
```

### Resource Management
```bash
# Monitor resources
python tracking_cli.py --system-stats

# Clean up old data
python tracking_cli.py --cleanup --older-than 30d

# Optimize database
python tracking_cli.py --optimize-db
```

---

**For detailed documentation, see [README.md](README.md)**
