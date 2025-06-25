from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from sagemaker.pytorch import PyTorchModel
from typing import Dict, List, Optional
import json
import os
import sagemaker
import subprocess
import yaml
"""
Cloud Deployment and Environment Setup Scripts
สำหรับ deploy ไปยัง Cloud platforms ต่างๆ
"""


console = Console()

class CloudDeployer:
    """ระบบสำหรับ deploy ไป Cloud platforms"""

    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path)
        self.cloud_configs = {}

    def create_aws_deployment(self):
        """สร้าง AWS deployment files"""

        # EC2 User Data Script
        user_data = """#!/bin/bash
# AWS EC2 User Data Script for ML Project

# Update system
yum update -y
yum install -y docker git python3 python3 - pip

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker - compose - $(uname -s) - $(uname -m)" -o /usr/local/bin/docker - compose
chmod +x /usr/local/bin/docker - compose

# Start Docker
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2 - user

# Clone project (replace with your repo)
cd /home/ec2 - user
# git clone YOUR_REPO_URL project
# cd project

# Install Python dependencies
pip3 install mlflow rich pandas numpy scikit - learn

# Create directories
mkdir -p enterprise_tracking enterprise_mlruns models artifacts logs data

# Set permissions
chown -R ec2 - user:ec2 - user /home/ec2 - user/

# Start services
# docker - compose up -d
"""

        with open(self.project_path / "aws_user_data.sh", "w") as f:
            f.write(user_data)

        # CloudFormation Template
        cloudformation = {
            "AWSTemplateFormatVersion": "2010 - 09 - 09", 
            "Description": "ML Project Infrastructure", 
            "Parameters": {
                "InstanceType": {
                    "Type": "String", 
                    "Default": "t3.medium", 
                    "AllowedValues": ["t3.micro", "t3.small", "t3.medium", "t3.large"]
                }, 
                "KeyName": {
                    "Type": "AWS::EC2::KeyPair::KeyName", 
                    "Description": "EC2 Key Pair for SSH access"
                }
            }, 
            "Resources": {
                "MLProjectInstance": {
                    "Type": "AWS::EC2::Instance", 
                    "Properties": {
                        "InstanceType": {"Ref": "InstanceType"}, 
                        "KeyName": {"Ref": "KeyName"}, 
                        "ImageId": "ami - 0abcdef1234567890",  # Update with latest Amazon Linux 2 AMI
                        "SecurityGroupIds": [{"Ref": "MLProjectSecurityGroup"}], 
                        "UserData": {
                            "Fn::Base64": {
                                "Fn::Sub": """#!/bin/bash
yum update -y
yum install -y docker git python3 python3 - pip
pip3 install mlflow rich pandas numpy scikit - learn
"""
                            }
                        }, 
                        "Tags": [
                            {"Key": "Name", "Value": "ML - Project - Instance"}
                        ]
                    }
                }, 
                "MLProjectSecurityGroup": {
                    "Type": "AWS::EC2::SecurityGroup", 
                    "Properties": {
                        "GroupDescription": "Security group for ML Project", 
                        "SecurityGroupIngress": [
                            {
                                "IpProtocol": "tcp", 
                                "FromPort": 22, 
                                "ToPort": 22, 
                                "CidrIp": "0.0.0.0/0"
                            }, 
                            {
                                "IpProtocol": "tcp", 
                                "FromPort": 5000, 
                                "ToPort": 5000, 
                                "CidrIp": "0.0.0.0/0"
                            }, 
                            {
                                "IpProtocol": "tcp", 
                                "FromPort": 8080, 
                                "ToPort": 8080, 
                                "CidrIp": "0.0.0.0/0"
                            }
                        ]
                    }
                }
            }, 
            "Outputs": {
                "InstancePublicIP": {
                    "Description": "Public IP of the EC2 instance", 
                    "Value": {"Fn::GetAtt": ["MLProjectInstance", "PublicIp"]}
                }, 
                "MLflowURL": {
                    "Description": "MLflow UI URL", 
                    "Value": {"Fn::Sub": "http://${MLProjectInstance.PublicIp}:5000"}
                }
            }
        }

        with open(self.project_path / "aws_cloudformation.yaml", "w") as f:
            yaml.dump(cloudformation, f, default_flow_style = False)

        # AWS CLI deployment script
        aws_deploy = """#!/bin/bash
# AWS Deployment Script

echo "🚀 Deploying to AWS..."

# Variables
STACK_NAME = "ml - project - stack"
KEY_NAME = "your - ec2 - key"  # Replace with your key pair name
INSTANCE_TYPE = "t3.medium"

# Deploy CloudFormation stack
aws cloudformation deploy \\
    - - template - file aws_cloudformation.yaml \\
    - - stack - name $STACK_NAME \\
    - - parameter - overrides \\
        KeyName = $KEY_NAME \\
        InstanceType = $INSTANCE_TYPE \\
    - - capabilities CAPABILITY_IAM

# Get instance IP
INSTANCE_IP = $(aws cloudformation describe - stacks \\
    - - stack - name $STACK_NAME \\
    - - query 'Stacks[0].Outputs[?OutputKey =  = `InstancePublicIP`].OutputValue' \\
    - - output text)

echo "✅ Deployment complete!"
echo "📍 Instance IP: $INSTANCE_IP"
echo "🌐 MLflow UI: http://$INSTANCE_IP:5000"
echo "🔗 SSH: ssh -i your - key.pem ec2 - user@$INSTANCE_IP"
"""

        with open(self.project_path / "deploy_aws.sh", "w") as f:
            f.write(aws_deploy)

        # Make script executable
        os.chmod(self.project_path / "deploy_aws.sh", 0o755)

    def create_gcp_deployment(self):
        """สร้าง GCP deployment files"""

        # Compute Engine startup script
        startup_script = """#!/bin/bash
# GCP Compute Engine Startup Script

# Update system
apt - get update
apt - get install -y docker.io docker - compose git python3 python3 - pip

# Start Docker
systemctl start docker
systemctl enable docker
usermod -a -G docker $USER

# Install Python dependencies
pip3 install mlflow rich pandas numpy scikit - learn

# Create directories
mkdir -p /opt/ml - project
cd /opt/ml - project

# Clone project (replace with your repo)
# git clone YOUR_REPO_URL .

# Create necessary directories
mkdir -p enterprise_tracking enterprise_mlruns models artifacts logs data

# Start services
# docker - compose up -d
"""

        with open(self.project_path / "gcp_startup.sh", "w") as f:
            f.write(startup_script)

        # Deployment Manager template
        gcp_template = {
            "resources": [
                {
                    "name": "ml - project - vm", 
                    "type": "compute.v1.instance", 
                    "properties": {
                        "zone": "us - central1 - a", 
                        "machineType": "zones/us - central1 - a/machineTypes/n1 - standard - 2", 
                        "disks": [
                            {
                                "boot": True, 
                                "autoDelete": True, 
                                "initializeParams": {
                                    "sourceImage": "projects/ubuntu - os - cloud/global/images/family/ubuntu - 2004 - lts", 
                                    "diskType": "zones/us - central1 - a/diskTypes/pd - standard", 
                                    "diskSizeGb": 50
                                }
                            }
                        ], 
                        "networkInterfaces": [
                            {
                                "network": "global/networks/default", 
                                "accessConfigs": [
                                    {
                                        "type": "ONE_TO_ONE_NAT", 
                                        "name": "External NAT"
                                    }
                                ]
                            }
                        ], 
                        "metadata": {
                            "items": [
                                {
                                    "key": "startup - script", 
                                    "value": startup_script
                                }
                            ]
                        }, 
                        "tags": {
                            "items": ["ml - project", "http - server"]
                        }
                    }
                }, 
                {
                    "name": "ml - project - firewall", 
                    "type": "compute.v1.firewall", 
                    "properties": {
                        "allowed": [
                            {
                                "IPProtocol": "tcp", 
                                "ports": ["22", "5000", "8080", "8888"]
                            }
                        ], 
                        "sourceRanges": ["0.0.0.0/0"], 
                        "targetTags": ["ml - project"]
                    }
                }
            ]
        }

        with open(self.project_path / "gcp_deployment.yaml", "w") as f:
            yaml.dump(gcp_template, f, default_flow_style = False)

        # GCP deployment script
        gcp_deploy = """#!/bin/bash
# GCP Deployment Script

echo "🚀 Deploying to Google Cloud..."

# Variables
PROJECT_ID = "your - gcp - project"  # Replace with your project ID
DEPLOYMENT_NAME = "ml - project - deployment"
ZONE = "us - central1 - a"

# Set project
gcloud config set project $PROJECT_ID

# Deploy
gcloud deployment - manager deployments create $DEPLOYMENT_NAME \\
    - - config gcp_deployment.yaml

# Get instance IP
INSTANCE_IP = $(gcloud compute instances describe ml - project - vm \\
    - - zone = $ZONE \\
    - - format = 'get(networkInterfaces[0].accessConfigs[0].natIP)')

echo "✅ Deployment complete!"
echo "📍 Instance IP: $INSTANCE_IP"
echo "🌐 MLflow UI: http://$INSTANCE_IP:5000"
echo "🔗 SSH: gcloud compute ssh ml - project - vm - - zone = $ZONE"
"""

        with open(self.project_path / "deploy_gcp.sh", "w") as f:
            f.write(gcp_deploy)

        os.chmod(self.project_path / "deploy_gcp.sh", 0o755)

    def create_azure_deployment(self):
        """สร้าง Azure deployment files"""

        # Azure Resource Manager template
        arm_template = {
            "$schema": "https://schema.management.azure.com/schemas/2019 - 04 - 01/deploymentTemplate.json#", 
            "contentVersion": "1.0.0.0", 
            "parameters": {
                "vmName": {
                    "type": "string", 
                    "defaultValue": "ml - project - vm"
                }, 
                "adminUsername": {
                    "type": "string", 
                    "defaultValue": "azureuser"
                }, 
                "adminPassword": {
                    "type": "securestring"
                }
            }, 
            "variables": {
                "vnetName": "ml - project - vnet", 
                "subnetName": "default", 
                "nsgName": "ml - project - nsg", 
                "publicIPName": "ml - project - ip"
            }, 
            "resources": [
                {
                    "type": "Microsoft.Network/virtualNetworks", 
                    "apiVersion": "2020 - 06 - 01", 
                    "name": "[variables('vnetName')]", 
                    "location": "[resourceGroup().location]", 
                    "properties": {
                        "addressSpace": {
                            "addressPrefixes": ["10.0.0.0/16"]
                        }, 
                        "subnets": [
                            {
                                "name": "[variables('subnetName')]", 
                                "properties": {
                                    "addressPrefix": "10.0.0.0/24"
                                }
                            }
                        ]
                    }
                }, 
                {
                    "type": "Microsoft.Network/networkSecurityGroups", 
                    "apiVersion": "2020 - 06 - 01", 
                    "name": "[variables('nsgName')]", 
                    "location": "[resourceGroup().location]", 
                    "properties": {
                        "securityRules": [
                            {
                                "name": "SSH", 
                                "properties": {
                                    "priority": 1001, 
                                    "protocol": "TCP", 
                                    "access": "Allow", 
                                    "direction": "Inbound", 
                                    "sourceAddressPrefix": "*", 
                                    "sourcePortRange": "*", 
                                    "destinationAddressPrefix": "*", 
                                    "destinationPortRange": "22"
                                }
                            }, 
                            {
                                "name": "MLflow", 
                                "properties": {
                                    "priority": 1002, 
                                    "protocol": "TCP", 
                                    "access": "Allow", 
                                    "direction": "Inbound", 
                                    "sourceAddressPrefix": "*", 
                                    "sourcePortRange": "*", 
                                    "destinationAddressPrefix": "*", 
                                    "destinationPortRange": "5000"
                                }
                            }, 
                            {
                                "name": "HTTP", 
                                "properties": {
                                    "priority": 1003, 
                                    "protocol": "TCP", 
                                    "access": "Allow", 
                                    "direction": "Inbound", 
                                    "sourceAddressPrefix": "*", 
                                    "sourcePortRange": "*", 
                                    "destinationAddressPrefix": "*", 
                                    "destinationPortRange": "8080"
                                }
                            }
                        ]
                    }
                }
            ]
        }

        with open(self.project_path / "azure_template.json", "w") as f:
            json.dump(arm_template, f, indent = 2)

        # Azure CLI deployment script
        azure_deploy = """#!/bin/bash
# Azure Deployment Script

echo "🚀 Deploying to Azure..."

# Variables
RESOURCE_GROUP = "ml - project - rg"
LOCATION = "eastus"
VM_NAME = "ml - project - vm"
ADMIN_USERNAME = "azureuser"

# Create resource group
az group create - - name $RESOURCE_GROUP - - location $LOCATION

# Deploy template
az deployment group create \\
    - - resource - group $RESOURCE_GROUP \\
    - - template - file azure_template.json \\
    - - parameters vmName = $VM_NAME adminUsername = $ADMIN_USERNAME

# Get public IP
PUBLIC_IP = $(az vm show \\
    - - resource - group $RESOURCE_GROUP \\
    - - name $VM_NAME \\
    - - show - details \\
    - - query publicIps \\
    - - output tsv)

echo "✅ Deployment complete!"
echo "📍 Public IP: $PUBLIC_IP"
echo "🌐 MLflow UI: http://$PUBLIC_IP:5000"
echo "🔗 SSH: ssh $ADMIN_USERNAME@$PUBLIC_IP"
"""

        with open(self.project_path / "deploy_azure.sh", "w") as f:
            f.write(azure_deploy)

        os.chmod(self.project_path / "deploy_azure.sh", 0o755)

    def create_kubernetes_deployment(self):
        """สร้าง Kubernetes deployment files"""

        # Kubernetes namespace
        namespace = {
            "apiVersion": "v1", 
            "kind": "Namespace", 
            "metadata": {
                "name": "ml - project"
            }
        }

        # Deployment
        deployment = {
            "apiVersion": "apps/v1", 
            "kind": "Deployment", 
            "metadata": {
                "name": "ml - project - app", 
                "namespace": "ml - project"
            }, 
            "spec": {
                "replicas": 2, 
                "selector": {
                    "matchLabels": {
                        "app": "ml - project"
                    }
                }, 
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "ml - project"
                        }
                    }, 
                    "spec": {
                        "containers": [
                            {
                                "name": "ml - project", 
                                "image": "ml - project:latest",  # Build and push your image
                                "ports": [
                                    {"containerPort": 5000}, 
                                    {"containerPort": 8080}
                                ], 
                                "env": [
                                    {
                                        "name": "MLFLOW_TRACKING_URI", 
                                        "value": "./enterprise_mlruns"
                                    }
                                ], 
                                "volumeMounts": [
                                    {
                                        "name": "data - storage", 
                                        "mountPath": "/app/data"
                                    }
                                ]
                            }
                        ], 
                        "volumes": [
                            {
                                "name": "data - storage", 
                                "persistentVolumeClaim": {
                                    "claimName": "ml - project - pvc"
                                }
                            }
                        ]
                    }
                }
            }
        }

        # Service
        service = {
            "apiVersion": "v1", 
            "kind": "Service", 
            "metadata": {
                "name": "ml - project - service", 
                "namespace": "ml - project"
            }, 
            "spec": {
                "selector": {
                    "app": "ml - project"
                }, 
                "ports": [
                    {
                        "name": "mlflow", 
                        "port": 5000, 
                        "targetPort": 5000
                    }, 
                    {
                        "name": "app", 
                        "port": 8080, 
                        "targetPort": 8080
                    }
                ], 
                "type": "LoadBalancer"
            }
        }

        # PersistentVolumeClaim
        pvc = {
            "apiVersion": "v1", 
            "kind": "PersistentVolumeClaim", 
            "metadata": {
                "name": "ml - project - pvc", 
                "namespace": "ml - project"
            }, 
            "spec": {
                "accessModes": ["ReadWriteOnce"], 
                "resources": {
                    "requests": {
                        "storage": "10Gi"
                    }
                }
            }
        }

        # Save Kubernetes manifests
        with open(self.project_path / "k8s_namespace.yaml", "w") as f:
            yaml.dump(namespace, f, default_flow_style = False)

        with open(self.project_path / "k8s_deployment.yaml", "w") as f:
            yaml.dump(deployment, f, default_flow_style = False)

        with open(self.project_path / "k8s_service.yaml", "w") as f:
            yaml.dump(service, f, default_flow_style = False)

        with open(self.project_path / "k8s_pvc.yaml", "w") as f:
            yaml.dump(pvc, f, default_flow_style = False)

        # Kubernetes deployment script
        k8s_deploy = """#!/bin/bash
# Kubernetes Deployment Script

echo "🚀 Deploying to Kubernetes..."

# Build Docker image
docker build -t ml - project:latest .

# If using cloud registry, tag and push
# docker tag ml - project:latest gcr.io/YOUR_PROJECT/ml - project:latest
# docker push gcr.io/YOUR_PROJECT/ml - project:latest

# Apply Kubernetes manifests
kubectl apply -f k8s_namespace.yaml
kubectl apply -f k8s_pvc.yaml
kubectl apply -f k8s_deployment.yaml
kubectl apply -f k8s_service.yaml

# Wait for deployment
kubectl rollout status deployment/ml - project - app -n ml - project

# Get service URL
SERVICE_IP = $(kubectl get service ml - project - service -n ml - project \\
    - - output jsonpath = '{.status.loadBalancer.ingress[0].ip}')

if [ -z "$SERVICE_IP" ]; then
    echo "⏳ Waiting for LoadBalancer IP..."
    kubectl get service ml - project - service -n ml - project - - watch
else
    echo "✅ Deployment complete!"
    echo "🌐 MLflow UI: http://$SERVICE_IP:5000"
    echo "🔗 App: http://$SERVICE_IP:8080"
fi

# Show all resources
kubectl get all -n ml - project
"""

        with open(self.project_path / "deploy_k8s.sh", "w") as f:
            f.write(k8s_deploy)

        os.chmod(self.project_path / "deploy_k8s.sh", 0o755)

    def create_docker_swarm_deployment(self):
        """สร้าง Docker Swarm deployment"""

        swarm_compose = {
            "version": "3.8", 
            "services": {
                "ml - app": {
                    "image": "ml - project:latest", 
                    "deploy": {
                        "replicas": 2, 
                        "restart_policy": {
                            "condition": "on - failure"
                        }
                    }, 
                    "ports": ["5000:5000", "8080:8080"], 
                    "volumes": [
                        "ml - data:/app/data", 
                        "ml - models:/app/models", 
                        "ml - tracking:/app/enterprise_tracking"
                    ], 
                    "environment": [
                        "MLFLOW_TRACKING_URI = ./enterprise_mlruns"
                    ], 
                    "networks": ["ml - network"]
                }, 
                "mlflow": {
                    "image": "python:3.9 - slim", 
                    "command": ["sh", " - c", "pip install mlflow && mlflow server - - host 0.0.0.0 - - port 5000"], 
                    "ports": ["5001:5000"], 
                    "volumes": ["ml - tracking:/mlruns"], 
                    "networks": ["ml - network"]
                }
            }, 
            "volumes": {
                "ml - data": {}, 
                "ml - models": {}, 
                "ml - tracking": {}
            }, 
            "networks": {
                "ml - network": {
                    "driver": "overlay"
                }
            }
        }

        with open(self.project_path / "docker - swarm.yml", "w") as f:
            yaml.dump(swarm_compose, f, default_flow_style = False)

        # Swarm deployment script
        swarm_deploy = """#!/bin/bash
# Docker Swarm Deployment Script

echo "🚀 Deploying to Docker Swarm..."

# Initialize swarm (if not already)
docker swarm init || echo "Swarm already initialized"

# Build image
docker build -t ml - project:latest .

# Deploy stack
docker stack deploy -c docker - swarm.yml ml - project

# Wait for services
echo "⏳ Waiting for services to start..."
sleep 30

# Show services
docker service ls

# Get service URLs
echo "✅ Deployment complete!"
echo "🌐 MLflow UI: http://localhost:5000"
echo "🔗 App: http://localhost:8080"
echo "📊 MLflow Server: http://localhost:5001"

# Monitor services
docker service ps ml - project_ml - app
"""

        with open(self.project_path / "deploy_swarm.sh", "w") as f:
            f.write(swarm_deploy)

        os.chmod(self.project_path / "deploy_swarm.sh", 0o755)


def create_deployment_guide():
    """สร้างคู่มือ deployment ครบถ้วน"""

    guide_content = """# 🚀 Cloud Deployment Guide
# คู่มือการ Deploy โปรเจ็กต์ไปยัง Cloud

## 📋 Overview
คู่มือนี้รวมวิธีการ deploy โปรเจ็กต์ ML ไปยัง platform ต่างๆ:

### 🌩️ Cloud Platforms
1. **AWS** - Amazon Web Services
2. **GCP** - Google Cloud Platform
3. **Azure** - Microsoft Azure
4. **Kubernetes** - Container orchestration
5. **Docker Swarm** - Docker native clustering

 -  -  - 

## 🔧 Prerequisites

### ทั่วไป
- Docker และ Docker Compose
- Git
- Project migration package

### Cloud - specific
- **AWS**: AWS CLI, AWS account
- **GCP**: gcloud CLI, GCP account
- **Azure**: Azure CLI, Azure account
- **Kubernetes**: kubectl, cluster access
- **Docker Swarm**: Docker Swarm cluster

 -  -  - 

## 🚀 Deployment Methods

### 1. AWS Deployment

#### วิธีที่ 1: EC2 with CloudFormation
```bash
# ติดตั้ง AWS CLI
aws configure

# Deploy infrastructure
./deploy_aws.sh

# หรือ manual deploy
aws cloudformation deploy \\
    - - template - file aws_cloudformation.yaml \\
    - - stack - name ml - project - stack \\
    - - parameter - overrides KeyName = your - key
```

#### วิธีที่ 2: ECS/Fargate
```bash
# สร้าง ECR repository
aws ecr create - repository - - repository - name ml - project

# Build และ push image
docker build -t ml - project .
docker tag ml - project:latest YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com/ml - project:latest
docker push YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com/ml - project:latest

# สร้าง ECS cluster และ service
aws ecs create - cluster - - cluster - name ml - project - cluster
```

#### วิธีที่ 3: AWS SageMaker
```python
# sagemaker_deploy.py

# Deploy model
model = PyTorchModel(
    entry_point = 'inference.py', 
    source_dir = 'src', 
    model_data = 's3://bucket/model.tar.gz', 
    role = 'SageMakerRole', 
    framework_version = '1.8.0', 
    py_version = 'py3'
)

predictor = model.deploy(
    initial_instance_count = 1, 
    instance_type = 'ml.t2.medium'
)
```

### 2. GCP Deployment

#### วิธีที่ 1: Compute Engine
```bash
# Authentication
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Deploy VM
./deploy_gcp.sh

# หรือ manual
gcloud deployment - manager deployments create ml - project \\
    - - config gcp_deployment.yaml
```

#### วิธีที่ 2: Google Kubernetes Engine (GKE)
```bash
# สร้าง GKE cluster
gcloud container clusters create ml - project - cluster \\
    - - zone us - central1 - a \\
    - - num - nodes 3

# Get credentials
gcloud container clusters get - credentials ml - project - cluster \\
    - - zone us - central1 - a

# Deploy to K8s
kubectl apply -f k8s/
```

#### วิธีที่ 3: Cloud Run
```bash
# Build และ push
gcloud builds submit - - tag gcr.io/PROJECT_ID/ml - project

# Deploy
gcloud run deploy ml - project \\
    - - image gcr.io/PROJECT_ID/ml - project \\
    - - platform managed \\
    - - port 5000 \\
    - - allow - unauthenticated
```

### 3. Azure Deployment

#### วิธีที่ 1: Virtual Machine
```bash
# Login
az login

# Deploy
./deploy_azure.sh

# หรือ manual
az group create - - name ml - project - rg - - location eastus
az deployment group create \\
    - - resource - group ml - project - rg \\
    - - template - file azure_template.json
```

#### วิธีที่ 2: Azure Container Instances
```bash
# สร้าง container registry
az acr create - - resource - group ml - project - rg \\
    - - name mlprojectacr - - sku Basic

# Build และ push
az acr build - - registry mlprojectacr \\
    - - image ml - project:latest .

# Deploy container
az container create \\
    - - resource - group ml - project - rg \\
    - - name ml - project - container \\
    - - image mlprojectacr.azurecr.io/ml - project:latest \\
    - - ports 5000 8080
```

#### วิธีที่ 3: Azure Kubernetes Service (AKS)
```bash
# สร้าง AKS cluster
az aks create \\
    - - resource - group ml - project - rg \\
    - - name ml - project - aks \\
    - - node - count 3 \\
    - - generate - ssh - keys

# Get credentials
az aks get - credentials \\
    - - resource - group ml - project - rg \\
    - - name ml - project - aks

# Deploy
kubectl apply -f k8s/
```

### 4. Kubernetes Deployment

#### Local Kubernetes (minikube/k3s)
```bash
# Start minikube
minikube start

# Deploy
./deploy_k8s.sh

# Access services
minikube service ml - project - service -n ml - project
```

#### Production Kubernetes
```bash
# Deploy namespace และ resources
kubectl apply -f k8s_namespace.yaml
kubectl apply -f k8s_pvc.yaml
kubectl apply -f k8s_deployment.yaml
kubectl apply -f k8s_service.yaml

# Monitor deployment
kubectl rollout status deployment/ml - project - app -n ml - project

# Get service URL
kubectl get service ml - project - service -n ml - project
```

### 5. Docker Swarm Deployment

```bash
# Initialize swarm
docker swarm init

# Deploy stack
./deploy_swarm.sh

# Scale services
docker service scale ml - project_ml - app = 5

# Monitor
docker service ls
docker service ps ml - project_ml - app
```

 -  -  - 

## 🔧 Configuration

### Environment Variables
```bash
# Required for all deployments
export MLFLOW_TRACKING_URI = ./enterprise_mlruns
export PYTHONPATH = /app
export ML_ENV = production

# Cloud - specific
export AWS_DEFAULT_REGION = us - east - 1
export GOOGLE_CLOUD_PROJECT = your - project
export AZURE_SUBSCRIPTION_ID = your - subscription
```

### Secrets Management
```bash
# Kubernetes secrets
kubectl create secret generic ml - secrets \\
    - - from - literal = api - key = your - api - key \\
    - - from - literal = db - password = your - password

# Docker secrets (Swarm)
echo "your - api - key" | docker secret create api_key -

# Cloud secrets
aws secretsmanager create - secret - - name ml - api - key - - secret - string "your - key"
gcloud secrets create ml - api - key - - data - file = key.txt
az keyvault secret set - - vault - name ml - vault - - name api - key - - value "your - key"
```

 -  -  - 

## 📊 Monitoring & Logging

### Application Monitoring
```yaml
# monitoring.yaml
apiVersion: v1
kind: Service
metadata:
  name: prometheus
spec:
  ports:
  - port: 9090
  selector:
    app: prometheus
```

### Log Aggregation
```bash
# ELK Stack deployment
kubectl apply -f https://raw.githubusercontent.com/elastic/cloud - on - k8s/1.8.0/config/crds/v1/all - crds.yaml
kubectl apply -f https://raw.githubusercontent.com/elastic/cloud - on - k8s/1.8.0/config/operator.yaml
```

 -  -  - 

## 🔒 Security Best Practices

### 1. Network Security
- Use VPC/VNet isolation
- Configure security groups/firewall rules
- Enable HTTPS/TLS

### 2. Access Control
- Use IAM roles and policies
- Implement RBAC for Kubernetes
- Rotate credentials regularly

### 3. Data Protection
- Encrypt data at rest and in transit
- Use managed databases when possible
- Backup data regularly

 -  -  - 

## 📈 Scaling Strategies

### Horizontal Scaling
```bash
# Kubernetes
kubectl scale deployment ml - project - app - - replicas = 10

# Docker Swarm
docker service scale ml - project_ml - app = 10

# Cloud auto - scaling
aws autoscaling create - auto - scaling - group \\
    - - auto - scaling - group - name ml - project - asg \\
    - - min - size 2 - - max - size 20 - - desired - capacity 5
```

### Vertical Scaling
```bash
# Update resource requests/limits
kubectl patch deployment ml - project - app -p '{"spec":{"template":{"spec":{"containers":[{"name":"ml - project", "resources":{"requests":{"memory":"2Gi", "cpu":"1000m"}}}]}}}}'
```

 -  -  - 

## 🚨 Troubleshooting

### Common Issues
1. **Image Pull Errors**: ตรวจสอบ registry credentials
2. **Service Discovery**: ตรวจสอบ DNS/service mesh
3. **Resource Limits**: Monitor CPU/memory usage
4. **Storage Issues**: ตรวจสอบ persistent volumes

### Debug Commands
```bash
# Kubernetes debugging
kubectl describe pod <pod - name>
kubectl logs <pod - name> -f
kubectl exec -it <pod - name> -- /bin/bash

# Docker debugging
docker logs <container - id>
docker exec -it <container - id> /bin/bash

# Cloud debugging
aws logs describe - log - groups
gcloud logging read "resource.type = gce_instance"
az monitor activity - log list
```

 -  -  - 

## 📚 Additional Resources

### Documentation Links
- [AWS ECS Guide](https://docs.aws.amazon.com/ecs/)
- [GKE Documentation](https://cloud.google.com/kubernetes - engine/docs)
- [Azure AKS Guide](https://docs.microsoft.com/en - us/azure/aks/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Swarm Guide](https://docs.docker.com/engine/swarm/)

### Monitoring Tools
- Prometheus + Grafana
- DataDog
- New Relic
- Cloud - native monitoring (CloudWatch, Stackdriver, Azure Monitor)

### CI/CD Integration
- GitHub Actions
- GitLab CI/CD
- Jenkins
- Azure DevOps
- AWS CodePipeline

 -  -  - 

Happy Deploying! 🚀
"""

    with open("CLOUD_DEPLOYMENT_GUIDE.md", "w", encoding = "utf - 8") as f:
        f.write(guide_content)


def main():
    """สร้างไฟล์ deployment ทั้งหมด"""
    console.print(Panel("🚀 Creating Cloud Deployment Files", style = "bold blue"))

    deployer = CloudDeployer()

    with console.status("Creating deployment files..."):
        deployer.create_aws_deployment()
        deployer.create_gcp_deployment()
        deployer.create_azure_deployment()
        deployer.create_kubernetes_deployment()
        deployer.create_docker_swarm_deployment()
        create_deployment_guide()

    console.print("✅ Created cloud deployment files:", style = "bold green")

    table = Table(title = "Deployment Files")
    table.add_column("Platform", style = "cyan")
    table.add_column("Files", style = "green")

    table.add_row("AWS", "aws_cloudformation.yaml, deploy_aws.sh, aws_user_data.sh")
    table.add_row("GCP", "gcp_deployment.yaml, deploy_gcp.sh, gcp_startup.sh")
    table.add_row("Azure", "azure_template.json, deploy_azure.sh")
    table.add_row("Kubernetes", "k8s_*.yaml, deploy_k8s.sh")
    table.add_row("Docker Swarm", "docker - swarm.yml, deploy_swarm.sh")
    table.add_row("Documentation", "CLOUD_DEPLOYMENT_GUIDE.md")

    console.print(table)

    console.print("\n📖 อ่าน CLOUD_DEPLOYMENT_GUIDE.md สำหรับคำแนะนำรายละเอียด", style = "bold yellow")


if __name__ == "__main__":
    main()