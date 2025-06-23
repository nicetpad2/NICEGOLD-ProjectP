# 🚀 Enterprise ML Tracking System - คู่มือการติดตั้งและใช้งาน

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-2.9.0+-green.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ระบบ Enterprise-grade ML Tracking สำหรับการติดตาม experiments และ production monitoring แบบมืออาชีพ

## 📋 สารบัญ

- [🎯 ภาพรวมระบบ](#-ภาพรวมระบบ)
- [⚡ Quick Start](#-quick-start)
- [🔧 การติดตั้งแบบละเอียด](#-การติดตั้งแบบละเอียด)
- [🏗️ การย้ายโปรเจ็กต์](#️-การย้ายโปรเจ็กต์)
- [📖 การใช้งาน](#-การใช้งาน)
- [🏭 Production Deployment](#-production-deployment)
- [🔍 Troubleshooting](#-troubleshooting)
- [📚 Examples](#-examples)

## 🎯 ภาพรวมระบบ

ระบบ tracking นี้ประกอบด้วย:

### 📊 Tracking Backends
- **MLflow** - Primary tracking backend
- **Weights & Biases** - Cloud-based tracking (optional)
- **Local Storage** - File-based tracking

### 🏗️ Architecture Components
```
enterprise_tracking/
├── 📁 experiments/         # Experiment data
├── 📁 models/             # Model artifacts
├── 📁 artifacts/          # Training artifacts
├── 📁 logs/               # System logs
├── 📁 monitoring/         # Production monitoring
├── 📁 reports/            # Analysis reports
└── 📁 backups/            # Data backups
```

### ✨ Key Features
- 🔄 Multi-backend tracking (MLflow, WandB, Local)
- 📈 Production monitoring & alerting
- 🔧 Automated experiment management
- 🌐 Web-based dashboard
- 📊 Advanced analytics & reporting
- 🔒 Enterprise security features
- 📦 Easy migration & deployment

---

## ⚡ Quick Start

### 1️⃣ การติดตั้งด่วน (5 นาที)

```bash
# Clone หรือ copy โปรเจ็กต์
cd your-project-directory

# ติดตั้ง dependencies
pip install -r tracking_requirements.txt

# รันสคริปต์ติดตั้งอัตโนมัติ
python enterprise_setup_tracking.py

# เริ่มใช้งาน
python tracking_examples.py
```

### 2️⃣ การใช้งานพื้นฐาน

```python
from tracking import ExperimentTracker

# เริ่ม tracking
tracker = ExperimentTracker("tracking_config.yaml")

with tracker.start_run("my_experiment") as run:
    # ฝึก model
    model = train_model()
    
    # Log metrics
    run.log_metric("accuracy", 0.95)
    run.log_metric("f1_score", 0.92)
    
    # Log model
    run.log_model(model, "trading_model")
    
    # Log artifacts
    run.log_artifact("model_report.html")
```

---

## 🔧 การติดตั้งแบบละเอียด

### 📋 System Requirements

```yaml
Operating System:
  - Windows 10/11
  - macOS 10.15+
  - Linux (Ubuntu 18.04+)

Python:
  - Version: 3.8 - 3.11
  - Recommended: 3.10

Hardware:
  - RAM: 8GB+ (16GB+ recommended)
  - Storage: 10GB+ free space
  - CPU: 4+ cores recommended
```

### 🔧 Step-by-Step Installation

#### 1️⃣ การเตรียม Python Environment

```bash
# สร้าง virtual environment
python -m venv tracking_env

# Activate environment
# Windows:
tracking_env\Scripts\activate
# macOS/Linux:
source tracking_env/bin/activate

# อัปเดต pip
python -m pip install --upgrade pip
```

#### 2️⃣ การติดตั้ง Dependencies

```bash
# ติดตั้ง core packages
pip install mlflow>=2.9.0
pip install wandb>=0.16.0
pip install rich>=13.0.0
pip install typer>=0.9.0
pip install pyyaml>=6.0

# ติดตั้ง ML packages
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install scikit-learn>=1.3.0
pip install matplotlib>=3.6.0
pip install seaborn>=0.12.0

# ติดตั้ง monitoring packages
pip install psutil>=5.9.0
pip install evidently>=0.4.0

# หรือติดตั้งทั้งหมดจากไฟล์
pip install -r tracking_requirements.txt
```

#### 3️⃣ การตั้งค่าไฟล์ Environment

```bash
# Copy environment template
cp .env.example .env

# แก้ไขไฟล์ .env
nano .env  # หรือใช้ text editor ที่ชอบ
```

**ตัวอย่าง .env file:**
```env
# MLflow Settings
MLFLOW_TRACKING_URI=./enterprise_mlruns
MLFLOW_DEFAULT_ARTIFACT_ROOT=./enterprise_mlruns/artifacts

# Weights & Biases (Optional)
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=your_project_name
WANDB_ENTITY=your_username

# Database (Optional)
DATABASE_URL=sqlite:///tracking.db

# Cloud Storage (Optional)
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_S3_BUCKET=your_bucket_name

# Monitoring
ENABLE_MONITORING=true
ALERT_EMAIL=your_email@domain.com
```

#### 4️⃣ การรันสคริปต์ติดตั้ง

```bash
# รันสคริปต์ติดตั้งอัตโนมัติ
python enterprise_setup_tracking.py

# ตรวจสอบการติดตั้ง
python init_tracking_system.py

# ทดสอบระบบ
python tracking_examples.py
```

#### 5️⃣ การตรวจสอบการติดตั้ง

```bash
# ตรวจสอบ MLflow
mlflow ui --backend-store-uri ./enterprise_mlruns --port 5000

# ตรวจสอบ CLI
python tracking_cli.py status

# ตรวจสอบ dashboard
streamlit run dashboard_app.py
```

---

## 🏗️ การย้ายโปรเจ็กต์

### 📦 Export โปรเจ็กต์ปัจจุบัน

#### 1️⃣ การ Export แบบง่าย

```bash
# Export ทุกอย่าง
python tracking_cli.py export --output ./project_backup.zip

# Export เฉพาะ experiments
python tracking_cli.py export --experiments-only --output ./experiments.zip

# Export เฉพาะ models
python tracking_cli.py export --models-only --output ./models.zip
```

#### 2️⃣ การ Export แบบละเอียด

```python
# สร้างสคริปต์ export กำหนดเอง
from tracking_migration import ProjectMigrator

migrator = ProjectMigrator()

# Export configuration
export_config = {
    'include_data': True,
    'include_models': True,
    'include_artifacts': True,
    'include_logs': False,
    'compress': True,
    'encrypt': False
}

# รัน export
migrator.export_project(
    output_path="./full_backup", 
    config=export_config
)
```

### 📥 Import ไปยังสภาพแวดล้อมใหม่

#### 1️⃣ การติดตั้งในสภาพแวดล้อมใหม่

```bash
# 1. ติดตั้งระบบ tracking ใหม่
python enterprise_setup_tracking.py

# 2. Import โปรเจ็กต์
python tracking_cli.py import --input ./project_backup.zip

# 3. ตรวจสอบการ import
python tracking_cli.py validate
```

#### 2️⃣ การ Import แบบละเอียด

```python
from tracking_migration import ProjectMigrator

migrator = ProjectMigrator()

# Import configuration
import_config = {
    'overwrite_existing': False,
    'validate_data': True,
    'create_backups': True,
    'skip_errors': False
}

# รัน import
result = migrator.import_project(
    input_path="./project_backup.zip",
    config=import_config
)

print(f"Import result: {result}")
```

### 🌐 Migration ระหว่าง Environments

#### 1️⃣ Development → Staging

```bash
# Export จาก dev
python tracking_cli.py export \
    --environment dev \
    --output ./dev_to_staging.zip \
    --include-data \
    --include-models

# Import ไปยัง staging
python tracking_cli.py import \
    --environment staging \
    --input ./dev_to_staging.zip \
    --validate
```

#### 2️⃣ Staging → Production

```bash
# Export โมเดลที่ผ่านการทดสอบแล้ว
python tracking_cli.py export \
    --environment staging \
    --models-only \
    --filter "status=approved" \
    --output ./staging_to_prod.zip

# Import ไปยัง production
python tracking_cli.py import \
    --environment production \
    --input ./staging_to_prod.zip \
    --production-mode
```

### 🔄 Cross-Platform Migration

#### Windows → Linux/Mac

```bash
# ใน Windows
python tracking_cli.py export \
    --cross-platform \
    --fix-paths \
    --output ./windows_export.zip

# ใน Linux/Mac
python tracking_cli.py import \
    --input ./windows_export.zip \
    --fix-paths \
    --platform linux
```

#### Local → Cloud

```bash
# Export สำหรับ cloud
python tracking_cli.py export \
    --cloud-ready \
    --compress \
    --output ./cloud_export.zip

# Upload ไปยัง cloud storage
aws s3 cp ./cloud_export.zip s3://your-bucket/backups/

# Download และ import ใน cloud instance
aws s3 cp s3://your-bucket/backups/cloud_export.zip ./
python tracking_cli.py import --input ./cloud_export.zip
```

---

## 📖 การใช้งาน

### 🧪 Experiment Tracking

#### 1️⃣ Basic Tracking

```python
from tracking import ExperimentTracker
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Initialize tracker
tracker = ExperimentTracker("tracking_config.yaml")

# Start experiment
with tracker.start_run("trading_model_v1") as run:
    # Load data
    data = pd.read_csv("trading_data.csv")
    X, y = prepare_features(data)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    
    # Evaluate
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    f1 = f1_score(y, predictions, average='weighted')
    
    # Log parameters
    run.log_param("n_estimators", 100)
    run.log_param("data_size", len(data))
    
    # Log metrics
    run.log_metric("accuracy", accuracy)
    run.log_metric("f1_score", f1)
    
    # Log model
    run.log_model(model, "rf_trading_model")
    
    # Log data
    run.log_artifact(data, "training_data.csv")
```

#### 2️⃣ Advanced Tracking

```python
# Hyperparameter tuning with tracking
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

with tracker.start_run("hyperparameter_tuning") as run:
    # Grid search
    grid_search = GridSearchCV(
        RandomForestClassifier(),
        param_grid,
        cv=5,
        scoring='f1_weighted'
    )
    
    grid_search.fit(X, y)
    
    # Log best parameters
    for param, value in grid_search.best_params_.items():
        run.log_param(f"best_{param}", value)
    
    # Log all trial results
    for i, params in enumerate(grid_search.cv_results_['params']):
        with run.start_nested_run(f"trial_{i}") as trial:
            for param, value in params.items():
                trial.log_param(param, value)
            
            score = grid_search.cv_results_['mean_test_score'][i]
            trial.log_metric("cv_score", score)
    
    # Log best model
    run.log_model(grid_search.best_estimator_, "best_model")
```

### 📊 Production Monitoring

#### 1️⃣ Model Performance Monitoring

```python
from tracking_integration import ProductionMonitor

monitor = ProductionMonitor()

# Monitor model predictions
@monitor.track_predictions
def predict_trade_signal(features):
    model = load_model("production_model")
    prediction = model.predict(features)
    
    # Monitor prediction quality
    monitor.log_prediction_metrics({
        'prediction_confidence': prediction.max(),
        'feature_drift': calculate_drift(features),
        'model_version': get_model_version()
    })
    
    return prediction

# Monitor system health
@monitor.track_system_health
def trading_pipeline():
    # Your trading logic here
    pass
```

#### 2️⃣ Data Quality Monitoring

```python
from tracking_integration import DataQualityTracker

dq_tracker = DataQualityTracker()

# Monitor incoming data
@dq_tracker.monitor_data_quality
def process_market_data(raw_data):
    # Check data quality
    quality_report = dq_tracker.analyze_data(raw_data)
    
    if quality_report['quality_score'] < 0.8:
        dq_tracker.trigger_alert(
            f"Data quality issue: {quality_report['issues']}"
        )
    
    return clean_data(raw_data)
```

### 🖥️ Dashboard Usage

#### 1️⃣ การเริ่ม Dashboard

```bash
# เริ่ม MLflow UI
mlflow ui --backend-store-uri ./enterprise_mlruns --port 5000

# เริ่ม Custom Dashboard
streamlit run dashboard_app.py --server.port 8501

# เริ่ม Monitoring Dashboard
python tracking_cli.py dashboard --port 8502
```

#### 2️⃣ การเข้าถึง Dashboard

- **MLflow UI**: http://localhost:5000
- **Custom Dashboard**: http://localhost:8501
- **Monitoring Dashboard**: http://localhost:8502

### 🔧 CLI Usage

#### 1️⃣ Experiment Management

```bash
# ดู experiments ทั้งหมด
python tracking_cli.py list-experiments

# ดู runs ในexperi ment
python tracking_cli.py list-runs --experiment "trading_model_v1"

# เปรียบเทียบ runs
python tracking_cli.py compare --runs run1,run2,run3

# Archive experiment
python tracking_cli.py archive --experiment "old_experiment"
```

#### 2️⃣ Model Management

```bash
# ดู models ทั้งหมด
python tracking_cli.py list-models

# Register model
python tracking_cli.py register-model \
    --name "trading_model" \
    --version "1.0" \
    --stage "Production"

# Deploy model
python tracking_cli.py deploy-model \
    --name "trading_model" \
    --version "1.0" \
    --endpoint "api"
```

#### 3️⃣ Monitoring & Alerts

```bash
# ดู system status
python tracking_cli.py status

# ตั้งค่า alerts
python tracking_cli.py set-alert \
    --metric "accuracy" \
    --threshold 0.8 \
    --action "email"

# ดู alert history
python tracking_cli.py list-alerts
```

---

## 🏭 Production Deployment

### 🐳 Docker Deployment

#### 1️⃣ การสร้าง Docker Image

```dockerfile
# Dockerfile.tracking
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY tracking_requirements.txt .
RUN pip install -r tracking_requirements.txt

# Copy tracking system
COPY tracking.py .
COPY tracking_config.yaml .
COPY tracking_cli.py .
COPY tracking_integration.py .

# Copy your project files
COPY . .

# Expose ports
EXPOSE 5000 8501 8502

# Start tracking services
CMD ["python", "tracking_cli.py", "serve", "--all"]
```

#### 2️⃣ การ Build และ Run

```bash
# Build image
docker build -f Dockerfile.tracking -t ml-tracking:latest .

# Run container
docker run -d \
    --name ml-tracking \
    -p 5000:5000 \
    -p 8501:8501 \
    -p 8502:8502 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    ml-tracking:latest
```

### ☸️ Kubernetes Deployment

#### 1️⃣ Configuration Files

```yaml
# k8s/tracking-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-tracking
spec:
  replicas: 2
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
        image: ml-tracking:latest
        ports:
        - containerPort: 5000
        - containerPort: 8501
        - containerPort: 8502
        env:
        - name: MLFLOW_TRACKING_URI
          value: "postgresql://user:pass@postgres:5432/mlflow"
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: models-volume
          mountPath: /app/models
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: tracking-data-pvc
      - name: models-volume
        persistentVolumeClaim:
          claimName: tracking-models-pvc
```

#### 2️⃣ Deploy to Kubernetes

```bash
# Apply configurations
kubectl apply -f k8s/

# Check deployment
kubectl get pods -l app=ml-tracking

# Port forward for access
kubectl port-forward service/ml-tracking 5000:5000
```

### ☁️ Cloud Deployment

#### 1️⃣ AWS Deployment

```bash
# ECR upload
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-west-2.amazonaws.com

docker tag ml-tracking:latest your-account.dkr.ecr.us-west-2.amazonaws.com/ml-tracking:latest
docker push your-account.dkr.ecr.us-west-2.amazonaws.com/ml-tracking:latest

# ECS deployment
aws ecs create-service \
    --cluster ml-cluster \
    --service-name ml-tracking \
    --task-definition ml-tracking:1 \
    --desired-count 2
```

#### 2️⃣ Azure Deployment

```bash
# Container Registry
az acr login --name yourregistry
docker tag ml-tracking:latest yourregistry.azurecr.io/ml-tracking:latest
docker push yourregistry.azurecr.io/ml-tracking:latest

# Container Instances
az container create \
    --resource-group myResourceGroup \
    --name ml-tracking \
    --image yourregistry.azurecr.io/ml-tracking:latest \
    --ports 5000 8501 8502
```

---

## 🔍 Troubleshooting

### ❌ Common Issues

#### 1️⃣ Import Errors

**Problem**: `ModuleNotFoundError: No module named 'mlflow'`

**Solution**:
```bash
# ตรวจสอบ environment
which python
pip list | grep mlflow

# ติดตั้งใหม่
pip install --upgrade mlflow

# ตรวจสอบ virtual environment
source your_venv/bin/activate  # Linux/Mac
your_venv\Scripts\activate     # Windows
```

#### 2️⃣ MLflow Connection Issues

**Problem**: `Connection refused to MLflow tracking server`

**Solution**:
```bash
# ตรวจสอบ MLflow server
mlflow server --backend-store-uri ./enterprise_mlruns --port 5000

# ตรวจสอบ tracking URI
python -c "import mlflow; print(mlflow.get_tracking_uri())"

# Reset tracking URI
export MLFLOW_TRACKING_URI=./enterprise_mlruns
```

#### 3️⃣ Permission Issues

**Problem**: `Permission denied: cannot create directory`

**Solution**:
```bash
# ตรวจสอบ permissions
ls -la

# แก้ไข permissions
chmod 755 enterprise_tracking/
chmod 755 enterprise_mlruns/

# เปลี่ยน owner (Linux/Mac)
sudo chown -R $USER:$USER enterprise_tracking/
```

#### 4️⃣ Memory Issues

**Problem**: `MemoryError: Unable to allocate array`

**Solution**:
```python
# ใน tracking_config.yaml
performance:
  batch_logging: true
  cache_size_mb: 256  # ลดลง
  cleanup_old_runs: true
  max_concurrent_runs: 5  # ลดลง
```

### 🔧 Debug Mode

```bash
# เปิด debug logging
export TRACKING_DEBUG=true
python tracking_examples.py

# ตรวจสอบ system info
python tracking_cli.py system-info

# รัน health check
python tracking_cli.py health-check
```

### 📋 Log Analysis

```bash
# ดู tracking logs
tail -f logs/tracking.log

# ดู error logs
tail -f logs/errors.log

# ดู MLflow logs
tail -f enterprise_mlruns/mlflow.log
```

---

## 📚 Examples

### 🧪 Complete Training Example

```python
# complete_training_example.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from tracking import ExperimentTracker

def main():
    # Initialize tracker
    tracker = ExperimentTracker("tracking_config.yaml")
    
    # Load and prepare data
    data = pd.read_csv("trading_data.csv")
    X = data.drop(['target', 'timestamp'], axis=1)
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Start experiment
    with tracker.start_run("complete_trading_model") as run:
        # Log data info
        run.log_param("train_size", len(X_train))
        run.log_param("test_size", len(X_test))
        run.log_param("features", list(X.columns))
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        # Log metrics
        run.log_metric("train_accuracy", train_score)
        run.log_metric("test_accuracy", test_score)
        
        # Generate and log plots
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
        plt.title('Top 10 Feature Importance')
        plt.tight_layout()
        run.log_figure(plt.gcf(), "feature_importance.png")
        plt.close()
        
        # Log model
        run.log_model(model, "trading_model")
        
        # Log classification report
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        run.log_artifact(report, "classification_report.json")
        
        print(f"✅ Experiment completed successfully!")
        print(f"🎯 Test Accuracy: {test_score:.4f}")

if __name__ == "__main__":
    main()
```

### 🏭 Production Monitoring Example

```python
# production_monitoring_example.py
import time
import random
from tracking_integration import ProductionMonitor

def simulate_trading_system():
    monitor = ProductionMonitor()
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        for i in range(100):
            # Simulate trading prediction
            features = np.random.randn(10)
            
            with monitor.track_prediction() as prediction_tracker:
                # Your actual prediction logic here
                prediction = random.choice([0, 1])
                confidence = random.uniform(0.6, 0.95)
                
                # Log prediction metrics
                prediction_tracker.log_metrics({
                    'prediction': prediction,
                    'confidence': confidence,
                    'processing_time': random.uniform(0.1, 0.5)
                })
                
                # Simulate model drift
                if random.random() < 0.1:  # 10% chance
                    prediction_tracker.log_alert(
                        'Model drift detected',
                        severity='warning'
                    )
            
            # Monitor system health
            monitor.log_system_metrics()
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\\n⚠️ Monitoring stopped by user")
    finally:
        monitor.stop_monitoring()
        print("📊 Monitoring session completed")

if __name__ == "__main__":
    simulate_trading_system()
```

### 📊 Advanced Analytics Example

```python
# advanced_analytics_example.py
from tracking_cli import TrackingCLI
import pandas as pd
import matplotlib.pyplot as plt

def analyze_experiments():
    cli = TrackingCLI()
    
    # Get all experiments
    experiments = cli.list_experiments()
    
    # Analyze performance trends
    performance_data = []
    
    for exp in experiments:
        runs = cli.get_experiment_runs(exp['experiment_id'])
        
        for run in runs:
            performance_data.append({
                'experiment': exp['name'],
                'run_id': run['run_id'],
                'accuracy': run.get('metrics', {}).get('accuracy', 0),
                'f1_score': run.get('metrics', {}).get('f1_score', 0),
                'timestamp': run.get('start_time', '')
            })
    
    df = pd.DataFrame(performance_data)
    
    # Create performance dashboard
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy over time
    df_sorted = df.sort_values('timestamp')
    axes[0, 0].plot(df_sorted['timestamp'], df_sorted['accuracy'])
    axes[0, 0].set_title('Accuracy Over Time')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # F1 Score distribution
    axes[0, 1].hist(df['f1_score'], bins=20)
    axes[0, 1].set_title('F1 Score Distribution')
    
    # Performance by experiment
    exp_performance = df.groupby('experiment').agg({
        'accuracy': 'mean',
        'f1_score': 'mean'
    }).reset_index()
    
    axes[1, 0].bar(exp_performance['experiment'], exp_performance['accuracy'])
    axes[1, 0].set_title('Average Accuracy by Experiment')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Model comparison
    axes[1, 1].scatter(df['accuracy'], df['f1_score'])
    axes[1, 1].set_xlabel('Accuracy')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('Accuracy vs F1 Score')
    
    plt.tight_layout()
    plt.savefig('experiment_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Generate report
    report = f"""
    # Experiment Analysis Report
    
    ## Summary
    - Total Experiments: {len(experiments)}
    - Total Runs: {len(df)}
    - Best Accuracy: {df['accuracy'].max():.4f}
    - Best F1 Score: {df['f1_score'].max():.4f}
    
    ## Top Performing Experiments
    {exp_performance.sort_values('accuracy', ascending=False).head().to_string()}
    
    ## Recommendations
    - Focus on experiments with accuracy > {df['accuracy'].quantile(0.8):.4f}
    - Consider ensembling top performing models
    - Investigate experiments with high variance
    """
    
    with open('experiment_analysis_report.md', 'w') as f:
        f.write(report)
    
    print("📊 Analysis completed! Check experiment_analysis_report.md")

if __name__ == "__main__":
    analyze_experiments()
```

---

## 🔗 Additional Resources

### 📖 Documentation Links
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Weights & Biases Docs](https://docs.wandb.ai/)
- [Evidently Documentation](https://docs.evidentlyai.com/)

### 🎓 Tutorials
- [MLflow Tutorial](https://mlflow.org/docs/latest/tutorials-and-examples/index.html)
- [Experiment Tracking Best Practices](https://neptune.ai/blog/ml-experiment-tracking)
- [Production ML Monitoring](https://christophergs.com/machine%20learning/2020/03/14/how-to-monitor-machine-learning-models/)

### 🛠️ Tools Integration
- [VS Code MLflow Extension](https://marketplace.visualstudio.com/items?itemName=mlflow.mlflow)
- [Jupyter MLflow Integration](https://mlflow.org/docs/latest/python_api/mlflow.jupyter.html)
- [DVC + MLflow Integration](https://dvc.org/doc/use-cases/versioning-data-and-model-files/tutorial)

---

## 🤝 Support & Contributing

### 🆘 Getting Help
1. Check this README and troubleshooting section
2. Search existing issues in the repository
3. Create a new issue with detailed description
4. Join our community discussions

### 🔧 Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📈 Roadmap

### 🚀 Upcoming Features
- [ ] Advanced model versioning
- [ ] Automated model deployment
- [ ] Real-time monitoring dashboard
- [ ] Integration with more ML frameworks
- [ ] Enhanced security features
- [ ] Multi-tenant support

### 🏆 Version History
- **v1.0.0** - Initial release with basic tracking
- **v1.1.0** - Added production monitoring
- **v1.2.0** - Enhanced CLI and dashboard
- **v1.3.0** - Cloud deployment support
- **v2.0.0** - Enterprise features (current)

---

**🎉 Happy Tracking! สำหรับคำถามเพิ่มเติม กรุณาติดต่อทีมพัฒนา**
