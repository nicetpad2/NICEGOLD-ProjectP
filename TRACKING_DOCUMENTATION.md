# Enterprise ML Tracking System

## 🚀 Overview

Enterprise-grade experiment tracking system for machine learning projects with support for:
- **MLflow** - Industry standard experiment tracking
- **Weights & Biases** - Advanced experiment visualization  
- **Local tracking** - Offline experiment storage
- **Production monitoring** - Live model performance tracking
- **Data pipeline tracking** - ETL/data processing monitoring
- **Model deployment tracking** - Deployment lifecycle management

## 📋 Features

### 🧪 Experiment Tracking
- **Multiple backends**: MLflow, WandB, local storage
- **Automatic fallbacks**: System continues working even if backends fail
- **Rich metadata**: Parameters, metrics, artifacts, code snapshots
- **Context managers**: Clean experiment lifecycle management
- **Batch operations**: Efficient logging for large experiments

### 📊 Production Monitoring  
- **Real-time metrics**: Live model performance tracking
- **System monitoring**: CPU, memory, disk usage
- **Alert system**: Automated alerts for anomalies
- **Prediction logging**: Individual prediction tracking
- **Performance profiling**: Latency and throughput monitoring

### 🔄 Data Pipeline Integration
- **Pipeline tracking**: ETL/data processing monitoring
- **Data quality metrics**: Completeness, accuracy tracking
- **Stage-by-stage logging**: Detailed pipeline breakdown
- **Error tracking**: Failed operations monitoring
- **Resource usage**: Processing time and resource consumption

### 🎛️ Management Tools
- **CLI interface**: Command-line management tools
- **Web dashboard**: Browser-based experiment browser
- **Report generation**: Automated HTML/PDF reports
- **Configuration management**: Flexible YAML-based config
- **Backup/restore**: Experiment data backup utilities

## 🛠️ Installation

### Quick Setup
```bash
# Clone the tracking files to your project
# Run the setup script
python setup_tracking.py
```

### Manual Installation
```bash
# Install requirements
pip install -r tracking_requirements.txt

# Configure tracking
cp tracking_config.yaml.example tracking_config.yaml
# Edit tracking_config.yaml with your settings

# Test installation
python -c "from tracking import tracker; print('✅ Tracking system ready!')"
```

## 📖 Usage Guide

### Basic Experiment Tracking

```python
from tracking import start_experiment

# Start an experiment
with start_experiment("my_experiment", "run_001") as exp:
    # Log parameters
    exp.log_params({
        "learning_rate": 0.01,
        "batch_size": 32,
        "model_type": "RandomForest"
    })
    
    # Train your model
    model = train_model(params)
    
    # Log metrics
    exp.log_metrics({
        "accuracy": 0.95,
        "precision": 0.93,
        "recall": 0.94,
        "f1_score": 0.935
    })
    
    # Log model artifact
    exp.log_model(model, "best_model")
    
    # Log plots
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(training_history)
    plt.title("Training History")
    exp.log_figure(plt.gcf(), "training_history")
```

### Advanced Features

```python
from tracking import ExperimentTracker

# Custom tracker with configuration
tracker = ExperimentTracker("custom_config.yaml")

# Experiment with tags and description
with tracker.start_run(
    experiment_name="hyperparameter_tuning",
    run_name="grid_search_001",
    tags={
        "optimizer": "grid_search",
        "dataset": "financial_data_v2",
        "environment": "production"
    },
    description="Grid search for optimal hyperparameters on financial dataset"
) as exp:
    
    # Log system information automatically
    exp.log_params({
        "search_space": str(param_grid),
        "cv_folds": 5,
        "scoring": "f1_weighted"
    })
    
    # Step-by-step metric logging
    for step, (params, score) in enumerate(grid_search_results):
        exp.log_metric("cv_score", score, step=step)
        exp.log_params(params, prefix=f"step_{step}_")
    
    # Log best results
    exp.log_metrics({
        "best_score": best_score,
        "best_params_hash": hash(str(best_params))
    })
    
    # Set final tags
    exp.set_tags({
        "status": "completed",
        "best_score": str(best_score)
    })
```

### Production Monitoring

```python
from tracking_integration import start_production_monitoring, log_prediction

# Start monitoring deployed model
deployment_id = "trading_model_v1_prod"
start_production_monitoring("trading_model_v1", deployment_id)

# Log individual predictions
for market_data in live_data_stream:
    start_time = time.time()
    
    # Make prediction
    prediction = model.predict(market_data)
    confidence = model.predict_proba(market_data).max()
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Log prediction
    log_prediction(
        deployment_id=deployment_id,
        input_data=market_data,
        prediction=prediction,
        confidence=confidence,
        latency_ms=latency_ms
    )
    
    # Log trading results (if available)
    if trade_executed:
        log_trade_result(deployment_id, {
            "pnl": trade_pnl,
            "return_pct": return_percentage,
            "position_size": position_size
        })
```

### Data Pipeline Tracking

```python
from tracking_integration import start_data_pipeline

# Track data pipeline
with start_data_pipeline("daily_market_data", "yahoo_finance", 10000) as pipeline:
    
    # Stage 1: Data extraction
    raw_data = extract_market_data()
    pipeline.log_stage("extraction", len(raw_data), errors=0, duration_seconds=45.2)
    
    # Stage 2: Data cleaning
    clean_data = clean_market_data(raw_data)
    missing_count = len(raw_data) - len(clean_data)
    pipeline.log_stage("cleaning", len(clean_data), errors=missing_count, duration_seconds=12.5)
    
    # Stage 3: Feature engineering
    features = engineer_features(clean_data)
    pipeline.log_stage("feature_engineering", len(features), errors=0, duration_seconds=67.3)
    
    # Log data quality metrics
    pipeline.log_data_quality({
        "completeness": len(clean_data) / len(raw_data),
        "accuracy": calculate_accuracy_score(features),
        "consistency": check_data_consistency(features)
    })
    
    # Complete pipeline
    pipeline.complete_pipeline(len(features), success=True)
```

## 🖥️ Command Line Interface

### Basic Commands

```bash
# List recent experiments
tracking_cli list-experiments --limit 20

# Show specific experiment details
tracking_cli show-run run_20241223_143052_a1b2c3d4

# Find best runs by metric
tracking_cli best-runs --metric accuracy --mode max --top-k 5

# Generate HTML report
tracking_cli generate-report --days 30 --output monthly_report.html
```

### Production Monitoring

```bash
# Start production monitoring
tracking_cli production start-monitoring "trading_model_v1" "prod_deployment_001"

# Check monitoring status
tracking_cli production status "prod_deployment_001"

# View production alerts
tracking_cli production alerts --deployment "prod_deployment_001" --last-hours 24
```

### Configuration Management

```bash
# Validate configuration
tracking_cli config validate

# Show current configuration
tracking_cli config show

# Update configuration
tracking_cli config set mlflow.enabled true
tracking_cli config set wandb.project "new_project_name"
```

## ⚙️ Configuration

### Main Configuration File (`tracking_config.yaml`)

```yaml
# MLflow settings
mlflow:
  enabled: true
  tracking_uri: "./mlruns"  # Local or remote URI
  experiment_name: "trading_ml_production"

# Weights & Biases settings  
wandb:
  enabled: false
  project: "trading_ml"
  entity: "your_username"

# Local tracking
local:
  enabled: true
  save_models: true
  save_plots: true
  save_data: true

# Directories
tracking_dir: "./experiment_tracking"
models_dir: "./models"
artifacts_dir: "./artifacts"

# Auto-logging
auto_log:
  enabled: true
  log_system_info: true
  log_git_info: true
  log_environment: true

# Production monitoring
monitoring:
  enabled: true
  alert_thresholds:
    cpu_percent: 90
    memory_percent: 85
    latency_ms: 1000
    error_rate: 0.05

# Notifications
notifications:
  email:
    enabled: false
    smtp_host: "smtp.gmail.com"
    recipients: ["admin@company.com"]
  slack:
    enabled: false
    webhook_url: "https://hooks.slack.com/..."
```

### Environment Variables

```bash
# MLflow
export MLFLOW_TRACKING_URI="http://your-mlflow-server:5000"
export MLFLOW_EXPERIMENT_NAME="production_experiments"

# Weights & Biases
export WANDB_API_KEY="your_wandb_api_key"
export WANDB_PROJECT="trading_ml_production"

# Database (for enterprise deployments)
export TRACKING_DB_URL="postgresql://user:pass@localhost/tracking"

# Storage (for cloud deployments)
export AWS_ACCESS_KEY_ID="your_aws_key"
export AWS_SECRET_ACCESS_KEY="your_aws_secret"
export S3_BUCKET="ml-tracking-artifacts"
```

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Enterprise ML Tracking                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │    CLI      │  │  Web UI     │  │  Python API         │  │
│  │ Interface   │  │ Dashboard   │  │  (tracking.py)      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   MLflow    │  │   WandB     │  │  Local Storage      │  │
│  │  Backend    │  │  Backend    │  │  Backend            │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Production  │  │ Data Pipeline│  │ Model Deployment   │  │
│  │ Monitoring  │  │ Tracking     │  │ Tracking           │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
[Experiment Code] 
       ↓
[Tracking API] → [Parameter/Metric Validation]
       ↓
[Backend Router] → [MLflow] → [MLflow Server/Storage]
       ↓         → [WandB]   → [WandB Cloud]
       ↓         → [Local]   → [Local Files/Database]
       ↓
[Aggregation Layer] → [Reports] → [Web Dashboard]
                   → [Alerts]  → [Notifications]
```

## 🔧 Advanced Configuration

### Custom Backend Integration

```python
from tracking import ExperimentTracker

class CustomBackend:
    def log_param(self, key, value):
        # Custom parameter logging logic
        pass
    
    def log_metric(self, key, value, step=None):
        # Custom metric logging logic
        pass

# Register custom backend
tracker = ExperimentTracker()
tracker.backends['custom'] = CustomBackend()
```

### Database Integration

```python
# For PostgreSQL backend
import sqlalchemy as sa
from tracking import ExperimentTracker

# Configure database
config = {
    'database': {
        'enabled': True,
        'url': 'postgresql://user:pass@localhost/tracking'
    }
}

tracker = ExperimentTracker(config)
```

### Cloud Storage Integration

```python
# For AWS S3 artifact storage
config = {
    'storage': {
        'type': 's3',
        'bucket': 'ml-tracking-artifacts',
        'prefix': 'experiments/'
    }
}
```

## 📊 Monitoring & Alerts

### Production Alerts

The system automatically monitors:
- **System Resources**: CPU, memory, disk usage
- **Model Performance**: Prediction latency, confidence scores
- **Business Metrics**: Trading P&L, accuracy, error rates
- **Data Quality**: Missing values, schema changes

### Alert Configuration

```yaml
monitoring:
  alerts:
    high_cpu:
      threshold: 90
      window_minutes: 5
      action: "email,slack"
    
    low_accuracy:
      threshold: 0.7
      window_hours: 1
      action: "email"
    
    high_latency:
      threshold: 1000  # milliseconds
      window_minutes: 10
      action: "slack"
```

## 🚀 Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY tracking_requirements.txt .
RUN pip install -r tracking_requirements.txt

COPY tracking.py tracking_integration.py tracking_config.yaml ./
COPY your_ml_code.py ./

CMD ["python", "your_ml_code.py"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-tracking-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-tracking
  template:
    metadata:
      labels:
        app: ml-tracking
    spec:
      containers:
      - name: tracking
        image: your-registry/ml-tracking:latest
        env:
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow-server:5000"
        - name: TRACKING_CONFIG_PATH
          value: "/config/tracking_config.yaml"
        volumeMounts:
        - name: config
          mountPath: /config
      volumes:
      - name: config
        configMap:
          name: tracking-config
```

## 🔍 Troubleshooting

### Common Issues

1. **MLflow Connection Failed**
   ```bash
   # Check MLflow server status
   curl http://your-mlflow-server:5000/health
   
   # Verify environment variables
   echo $MLFLOW_TRACKING_URI
   ```

2. **WandB Authentication**
   ```bash
   # Login to WandB
   wandb login
   
   # Verify API key
   echo $WANDB_API_KEY
   ```

3. **Permission Errors**
   ```bash
   # Check directory permissions
   ls -la ./experiment_tracking/
   
   # Fix permissions
   chmod -R 755 ./experiment_tracking/
   ```

4. **Memory Issues**
   ```python
   # Enable batch logging for large experiments
   config = {
       'performance': {
           'batch_logging': True,
           'cache_size_mb': 1024
       }
   }
   ```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from tracking import ExperimentTracker
tracker = ExperimentTracker(config)  # Will show debug info
```

## 📈 Performance Optimization

### Best Practices

1. **Batch Operations**: Log multiple metrics at once
2. **Async Logging**: Use background threads for I/O
3. **Compression**: Enable artifact compression
4. **Cleanup**: Regular cleanup of old experiments

```python
# Efficient logging
with start_experiment("training") as exp:
    # Batch metric logging
    all_metrics = {}
    for epoch in range(100):
        all_metrics[f"epoch_{epoch}_loss"] = losses[epoch]
        all_metrics[f"epoch_{epoch}_acc"] = accuracies[epoch]
    
    exp.log_metrics(all_metrics)  # Single batch operation
```

## 📚 API Reference

### Main Classes

- `ExperimentTracker`: Main tracking class
- `ProductionTracker`: Production monitoring
- `DataPipelineTracker`: Data pipeline tracking
- `ModelDeploymentTracker`: Deployment tracking

### Key Methods

- `start_run()`: Start experiment context
- `log_params()`: Log parameters
- `log_metrics()`: Log metrics
- `log_model()`: Log model artifacts
- `log_figure()`: Log plots/visualizations

See source code for complete API documentation.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Submit pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

- **Documentation**: This README
- **Issues**: GitHub Issues
- **Community**: Discussion forums
- **Enterprise**: Contact for enterprise support

---

**Built for production ML systems that need reliable, scalable experiment tracking.** 🚀
