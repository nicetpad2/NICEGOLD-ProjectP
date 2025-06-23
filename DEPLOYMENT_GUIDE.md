# 🏗️ การย้ายโปรเจ็กต์และ Deployment Guide

## 📋 สารบัญ
- [🚚 การย้ายโปรเจ็กต์](#-การย้ายโปรเจ็กต์)
- [🐳 Docker Deployment](#-docker-deployment)
- [☸️ Kubernetes Deployment](#️-kubernetes-deployment)
- [☁️ Cloud Deployment](#️-cloud-deployment)
- [🔄 CI/CD Pipeline](#-cicd-pipeline)

---

## 🚚 การย้ายโปรเจ็กต์

### 📦 วิธีที่ 1: การใช้ Quick Migration Tool

#### สร้าง Export Package
```bash
# ในเครื่องต้นทาง
python quick_migration.py export --output my_ml_project.zip
```

#### Import ในเครื่องใหม่
```bash
# ในเครื่องปลายทาง
python quick_migration.py import --input my_ml_project.zip --target ./new_project

# ติดตั้งระบบ
cd new_project
python setup_new_environment.py
```

### 📁 วิธีที่ 2: การคัดลอกไฟล์ด้วยตนเอง

#### ไฟล์และโฟลเดอร์ที่จำเป็น:
```
📁 Essential Files:
├── tracking.py                    # Core tracking system
├── tracking_config.yaml          # Configuration
├── tracking_cli.py               # Command line interface
├── tracking_integration.py       # Production integration
├── tracking_examples.py          # Usage examples
├── tracking_requirements.txt     # Dependencies
├── .env.example                  # Environment template

📁 Essential Directories:
├── enterprise_tracking/          # Local tracking data
├── enterprise_mlruns/           # MLflow experiments
├── models/                      # Model artifacts
├── artifacts/                   # Training artifacts
├── data/                        # Dataset storage
├── logs/                        # System logs
└── configs/                     # Configuration files
```

#### ขั้นตอนการคัดลอก:
```bash
# 1. สร้างโครงสร้างโฟลเดอร์ในเครื่องใหม่
mkdir -p new_project/{enterprise_tracking,enterprise_mlruns,models,artifacts,data,logs,configs}

# 2. คัดลอกไฟล์หลัก
cp tracking*.py new_project/
cp *.yaml new_project/
cp requirements*.txt new_project/
cp .env.example new_project/

# 3. คัดลอกข้อมูล (ถ้าต้องการ)
cp -r enterprise_tracking/ new_project/
cp -r enterprise_mlruns/ new_project/
cp -r models/ new_project/

# 4. ติดตั้งในเครื่องใหม่
cd new_project
pip install -r tracking_requirements.txt
python enterprise_setup_tracking.py
```

### 🌐 วิธีที่ 3: การใช้ Git Repository

```bash
# 1. สร้าง Git repository ในโปรเจ็กต์ต้นทาง
git init
git add tracking*.py *.yaml requirements*.txt .env.example
git add enterprise_tracking/ models/ configs/
git commit -m "Initial ML tracking system setup"

# 2. Push ไปยัง remote repository
git remote add origin https://github.com/yourusername/ml-tracking.git
git push -u origin main

# 3. Clone ในเครื่องใหม่
git clone https://github.com/yourusername/ml-tracking.git
cd ml-tracking
pip install -r tracking_requirements.txt
python enterprise_setup_tracking.py
```

---

## 🐳 Docker Deployment

### 📋 การเตรียม Docker Files

#### 1️⃣ สร้าง Dockerfile
```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY tracking_requirements.txt .
RUN pip install --no-cache-dir -r tracking_requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p enterprise_tracking enterprise_mlruns models artifacts logs data

# Set environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=./enterprise_mlruns

# Expose ports
EXPOSE 5000 8501 8502

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start MLflow server
CMD ["mlflow", "server", "--backend-store-uri", "./enterprise_mlruns", "--host", "0.0.0.0", "--port", "5000"]
```

#### 2️⃣ สร้าง docker-compose.yml
```yaml
# docker-compose.yml
version: '3.8'

services:
  ml-tracking:
    build: .
    ports:
      - "5000:5000"   # MLflow UI
      - "8501:8501"   # Streamlit Dashboard
      - "8502:8502"   # Monitoring
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./enterprise_mlruns:/app/enterprise_mlruns
      - ./logs:/app/logs
    environment:
      - MLFLOW_TRACKING_URI=./enterprise_mlruns
      - PYTHONPATH=/app
    restart: unless-stopped
    networks:
      - ml-network

  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: mlflow
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    networks:
      - ml-network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    networks:
      - ml-network

volumes:
  postgres_data:

networks:
  ml-network:
    driver: bridge
```

#### 3️⃣ การ Deploy
```bash
# Build และ start services
docker-compose up --build -d

# ตรวจสอบ status
docker-compose ps

# ดู logs
docker-compose logs -f ml-tracking

# Stop services
docker-compose down
```

### 🔧 การจัดการ Docker Container

```bash
# เข้าไปใน container
docker-compose exec ml-tracking bash

# Backup data
docker-compose exec ml-tracking tar -czf /tmp/backup.tar.gz enterprise_mlruns models
docker cp container_name:/tmp/backup.tar.gz ./backup.tar.gz

# Restore data
docker cp ./backup.tar.gz container_name:/tmp/
docker-compose exec ml-tracking tar -xzf /tmp/backup.tar.gz
```

---

## ☸️ Kubernetes Deployment

### 📋 K8s Configuration Files

#### 1️⃣ Namespace
```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ml-tracking
```

#### 2️⃣ ConfigMap
```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-tracking-config
  namespace: ml-tracking
data:
  tracking_config.yaml: |
    mlflow:
      enabled: true
      tracking_uri: "postgresql://mlflow:password@postgres:5432/mlflow"
      experiment_name: "production_experiments"
    
    local:
      enabled: true
      save_models: true
      save_plots: true
    
    monitoring:
      enabled: true
      alert_on_failure: true
```

#### 3️⃣ Persistent Volume Claims
```yaml
# pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ml-tracking-data-pvc
  namespace: ml-tracking
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ml-tracking-models-pvc
  namespace: ml-tracking
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
```

#### 4️⃣ Deployment
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-tracking
  namespace: ml-tracking
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
        image: your-registry/ml-tracking:latest
        ports:
        - containerPort: 5000
        - containerPort: 8501
        - containerPort: 8502
        env:
        - name: MLFLOW_TRACKING_URI
          value: "postgresql://mlflow:password@postgres:5432/mlflow"
        - name: PYTHONPATH
          value: "/app"
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: models-volume
          mountPath: /app/models
        - name: config-volume
          mountPath: /app/config
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: ml-tracking-data-pvc
      - name: models-volume
        persistentVolumeClaim:
          claimName: ml-tracking-models-pvc
      - name: config-volume
        configMap:
          name: ml-tracking-config
```

#### 5️⃣ Service & Ingress
```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-tracking-service
  namespace: ml-tracking
spec:
  selector:
    app: ml-tracking
  ports:
  - name: mlflow
    port: 5000
    targetPort: 5000
  - name: dashboard
    port: 8501
    targetPort: 8501
  - name: monitoring
    port: 8502
    targetPort: 8502
  type: LoadBalancer

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-tracking-ingress
  namespace: ml-tracking
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - ml-tracking.yourdomain.com
    secretName: ml-tracking-tls
  rules:
  - host: ml-tracking.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ml-tracking-service
            port:
              number: 5000
```

#### 6️⃣ การ Deploy
```bash
# Apply configurations
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f pvc.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml

# ตรวจสอบ deployment
kubectl get pods -n ml-tracking
kubectl get services -n ml-tracking
kubectl get ingress -n ml-tracking

# ดู logs
kubectl logs -f deployment/ml-tracking -n ml-tracking

# Port forward สำหรับทดสอบ
kubectl port-forward service/ml-tracking-service 5000:5000 -n ml-tracking
```

---

## ☁️ Cloud Deployment

### 🔥 AWS Deployment

#### 1️⃣ ECS with Fargate
```bash
# สร้าง ECR repository
aws ecr create-repository --repository-name ml-tracking

# Build และ push image
docker build -t ml-tracking .
docker tag ml-tracking:latest your-account.dkr.ecr.region.amazonaws.com/ml-tracking:latest
docker push your-account.dkr.ecr.region.amazonaws.com/ml-tracking:latest

# สร้าง ECS cluster
aws ecs create-cluster --cluster-name ml-tracking-cluster

# สร้าง task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# สร้าง service
aws ecs create-service \
    --cluster ml-tracking-cluster \
    --service-name ml-tracking-service \
    --task-definition ml-tracking:1 \
    --desired-count 2 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

#### 2️⃣ EC2 Auto Scaling Group
```bash
# สร้าง Launch Template
aws ec2 create-launch-template \
    --launch-template-name ml-tracking-template \
    --launch-template-data file://launch-template.json

# สร้าง Auto Scaling Group
aws autoscaling create-auto-scaling-group \
    --auto-scaling-group-name ml-tracking-asg \
    --launch-template "LaunchTemplateName=ml-tracking-template,Version=1" \
    --min-size 1 \
    --max-size 5 \
    --desired-capacity 2 \
    --target-group-arns arn:aws:elasticloadbalancing:region:account:targetgroup/ml-tracking-tg/xxx
```

### 🔷 Azure Deployment

#### 1️⃣ Container Instances
```bash
# สร้าง resource group
az group create --name ml-tracking-rg --location eastus

# สร้าง container registry
az acr create --resource-group ml-tracking-rg --name mltrackingregistry --sku Basic

# Push image
az acr login --name mltrackingregistry
docker tag ml-tracking:latest mltrackingregistry.azurecr.io/ml-tracking:latest
docker push mltrackingregistry.azurecr.io/ml-tracking:latest

# สร้าง container instance
az container create \
    --resource-group ml-tracking-rg \
    --name ml-tracking-container \
    --image mltrackingregistry.azurecr.io/ml-tracking:latest \
    --ports 5000 8501 8502 \
    --dns-name-label ml-tracking-unique \
    --location eastus
```

#### 2️⃣ App Service
```bash
# สร้าง App Service Plan
az appservice plan create \
    --name ml-tracking-plan \
    --resource-group ml-tracking-rg \
    --sku P1V2 \
    --is-linux

# สร้าง Web App
az webapp create \
    --resource-group ml-tracking-rg \
    --plan ml-tracking-plan \
    --name ml-tracking-app \
    --deployment-container-image-name mltrackingregistry.azurecr.io/ml-tracking:latest
```

### 🔶 Google Cloud Deployment

#### 1️⃣ Cloud Run
```bash
# Build และ push ไปยัง Container Registry
gcloud builds submit --tag gcr.io/your-project/ml-tracking

# Deploy ไปยัง Cloud Run
gcloud run deploy ml-tracking \
    --image gcr.io/your-project/ml-tracking \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 5000 \
    --memory 4Gi \
    --cpu 2
```

#### 2️⃣ GKE (Google Kubernetes Engine)
```bash
# สร้าง GKE cluster
gcloud container clusters create ml-tracking-cluster \
    --zone us-central1-a \
    --num-nodes 3

# Get credentials
gcloud container clusters get-credentials ml-tracking-cluster --zone us-central1-a

# Deploy applications
kubectl apply -f k8s/
```

---

## 🔄 CI/CD Pipeline

### 🐙 GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy ML Tracking System

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r tracking_requirements.txt
    
    - name: Run tests
      run: |
        python -m pytest tests/
    
    - name: Test tracking system
      run: |
        python tracking_examples.py

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t ml-tracking:${{ github.sha }} .
    
    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push ml-tracking:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - name: Deploy to production
      run: |
        # Add your deployment commands here
        kubectl set image deployment/ml-tracking tracking=ml-tracking:${{ github.sha }}
```

### 🦊 GitLab CI/CD

```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

variables:
  DOCKER_IMAGE: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA

test:
  stage: test
  image: python:3.10
  script:
    - pip install -r tracking_requirements.txt
    - python -m pytest tests/
    - python tracking_examples.py

build:
  stage: build
  image: docker:20.10.16
  services:
    - docker:20.10.16-dind
  script:
    - docker build -t $DOCKER_IMAGE .
    - docker push $DOCKER_IMAGE

deploy:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl set image deployment/ml-tracking tracking=$DOCKER_IMAGE
  only:
    - main
```

---

## 📋 Checklist การ Deployment

### ✅ Pre-Deployment

- [ ] ทดสอบระบบในสภาพแวดล้อม development
- [ ] เตรียม configuration files สำหรับ production
- [ ] ตั้งค่า environment variables
- [ ] เตรียม SSL certificates (ถ้าจำเป็น)
- [ ] ตั้งค่า monitoring และ logging
- [ ] เตรียม backup strategy

### ✅ During Deployment

- [ ] Deploy infrastructure components
- [ ] Deploy application
- [ ] ตรวจสอบ health checks
- [ ] ทดสอบ connectivity
- [ ] Verify data persistence
- [ ] ตรวจสอบ monitoring dashboards

### ✅ Post-Deployment

- [ ] ทดสอบ end-to-end functionality
- [ ] ตั้งค่า alerts และ notifications
- [ ] สร้าง documentation สำหรับ operations
- [ ] ฝึกอบรม team
- [ ] กำหนด maintenance schedule

---

## 🆘 Troubleshooting

### 🔧 Docker Issues

```bash
# ตรวจสอบ container logs
docker-compose logs ml-tracking

# เข้าไปใน container
docker-compose exec ml-tracking bash

# ตรวจสอบ resource usage
docker stats

# Cleanup unused resources
docker system prune -a
```

### ☸️ Kubernetes Issues

```bash
# ตรวจสอบ pod status
kubectl get pods -n ml-tracking

# ดู pod logs
kubectl logs -f pod-name -n ml-tracking

# Describe pod สำหรับ troubleshooting
kubectl describe pod pod-name -n ml-tracking

# ตรวจสอบ events
kubectl get events -n ml-tracking --sort-by='.metadata.creationTimestamp'
```

### ☁️ Cloud Issues

```bash
# AWS: ตรวจสอบ ECS service
aws ecs describe-services --cluster cluster-name --services service-name

# Azure: ตรวจสอบ container logs
az container logs --resource-group rg-name --name container-name

# GCP: ตรวจสอบ Cloud Run logs
gcloud logging read "resource.type=cloud_run_revision"
```

---

**🎯 ขอให้ deployment สำเร็จลุล่วง!**
