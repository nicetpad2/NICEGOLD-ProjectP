"""
Production Model Registry และ MLOps Pipeline
จัดการ model lifecycle, versioning, และ deployment
"""

import os
import json
import pickle
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import hashlib
import boto3
from botocore.exceptions import ClientError
import mlflow
import mlflow.sklearn
from sqlalchemy import create_engine, Column, String, DateTime, Float, Integer, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import redis
from prometheus_client import Counter, Histogram, Gauge

# Metrics
MODEL_DEPLOYMENTS = Counter('model_deployments_total', 'Total model deployments')
MODEL_PREDICTIONS = Counter('model_predictions_total', 'Total predictions made', ['model_id', 'version'])
MODEL_ACCURACY = Gauge('model_accuracy', 'Model accuracy score', ['model_id'])
MODEL_LATENCY = Histogram('model_prediction_latency_seconds', 'Model prediction latency')

Base = declarative_base()

@dataclass
class ModelMetadata:
    """Model metadata structure"""
    id: str
    name: str
    version: str
    algorithm: str
    framework: str
    created_at: datetime
    created_by: str
    description: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    dataset_hash: str
    features: List[str]
    target: str
    status: str  # training, validation, staging, production, deprecated
    file_path: str
    file_size: int
    checksum: str

class ModelRegistry(Base):
    """Database model for model registry"""
    __tablename__ = 'model_registry'
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    algorithm = Column(String, nullable=False)
    framework = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)
    created_by = Column(String, nullable=False)
    description = Column(Text)
    hyperparameters = Column(Text)  # JSON string
    metrics = Column(Text)  # JSON string
    dataset_hash = Column(String, nullable=False)
    features = Column(Text)  # JSON string
    target = Column(String, nullable=False)
    status = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    checksum = Column(String, nullable=False)

class ModelDeployment(Base):
    """Database model for model deployments"""
    __tablename__ = 'model_deployments'
    
    id = Column(String, primary_key=True)
    model_id = Column(String, nullable=False)
    environment = Column(String, nullable=False)  # staging, production
    deployed_at = Column(DateTime, nullable=False)
    deployed_by = Column(String, nullable=False)
    status = Column(String, nullable=False)  # active, inactive, failed
    endpoint_url = Column(String)
    health_check_url = Column(String)
    last_health_check = Column(DateTime)
    performance_metrics = Column(Text)  # JSON string

class MLOpsManager:
    """MLOps Manager for model lifecycle management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Database connection
        self.engine = create_engine(config['database_url'])
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.db_session = Session()
        
        # Redis connection for caching
        self.redis_client = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            password=config.get('redis_password'),
            decode_responses=True
        )
        
        # MLflow setup
        mlflow.set_tracking_uri(config.get('mlflow_uri', 'http://localhost:5000'))
        
        # S3/MinIO setup for model storage
        self.s3_client = boto3.client(
            's3',
            endpoint_url=config.get('s3_endpoint'),
            aws_access_key_id=config.get('s3_access_key'),
            aws_secret_access_key=config.get('s3_secret_key')
        )
        self.model_bucket = config.get('model_bucket', 'nicegold-models')
        
        # Ensure bucket exists
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Ensure S3 bucket exists"""
        try:
            self.s3_client.head_bucket(Bucket=self.model_bucket)
        except ClientError:
            try:
                self.s3_client.create_bucket(Bucket=self.model_bucket)
                self.logger.info(f"Created S3 bucket: {self.model_bucket}")
            except ClientError as e:
                self.logger.error(f"Failed to create S3 bucket: {e}")
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate file checksum"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _upload_model_to_s3(self, local_path: str, s3_key: str) -> bool:
        """Upload model file to S3"""
        try:
            self.s3_client.upload_file(local_path, self.model_bucket, s3_key)
            self.logger.info(f"Uploaded model to S3: {s3_key}")
            return True
        except ClientError as e:
            self.logger.error(f"Failed to upload model to S3: {e}")
            return False
    
    def _download_model_from_s3(self, s3_key: str, local_path: str) -> bool:
        """Download model file from S3"""
        try:
            self.s3_client.download_file(self.model_bucket, s3_key, local_path)
            self.logger.info(f"Downloaded model from S3: {s3_key}")
            return True
        except ClientError as e:
            self.logger.error(f"Failed to download model from S3: {e}")
            return False
    
    def register_model(
        self,
        model_object: Any,
        metadata: ModelMetadata,
        local_file_path: Optional[str] = None
    ) -> str:
        """Register a new model"""
        try:
            # Save model locally if not provided
            if local_file_path is None:
                local_file_path = f"/tmp/{metadata.id}_{metadata.version}.pkl"
                with open(local_file_path, 'wb') as f:
                    pickle.dump(model_object, f)
            
            # Calculate checksum and file size
            checksum = self._calculate_checksum(local_file_path)
            file_size = os.path.getsize(local_file_path)
            
            # Update metadata
            metadata.checksum = checksum
            metadata.file_size = file_size
            metadata.file_path = f"models/{metadata.id}/{metadata.version}/model.pkl"
            
            # Upload to S3
            if not self._upload_model_to_s3(local_file_path, metadata.file_path):
                raise Exception("Failed to upload model to S3")
            
            # Save to database
            db_model = ModelRegistry(
                id=metadata.id,
                name=metadata.name,
                version=metadata.version,
                algorithm=metadata.algorithm,
                framework=metadata.framework,
                created_at=metadata.created_at,
                created_by=metadata.created_by,
                description=metadata.description,
                hyperparameters=json.dumps(metadata.hyperparameters),
                metrics=json.dumps(metadata.metrics),
                dataset_hash=metadata.dataset_hash,
                features=json.dumps(metadata.features),
                target=metadata.target,
                status=metadata.status,
                file_path=metadata.file_path,
                file_size=metadata.file_size,
                checksum=metadata.checksum
            )
            
            self.db_session.add(db_model)
            self.db_session.commit()
            
            # Log to MLflow
            with mlflow.start_run(run_name=f"{metadata.name}_v{metadata.version}"):
                mlflow.log_params(metadata.hyperparameters)
                mlflow.log_metrics(metadata.metrics)
                mlflow.sklearn.log_model(model_object, "model")
                mlflow.set_tag("algorithm", metadata.algorithm)
                mlflow.set_tag("version", metadata.version)
            
            # Cache model metadata
            self.redis_client.setex(
                f"model:{metadata.id}",
                3600,  # 1 hour TTL
                json.dumps(asdict(metadata), default=str)
            )
            
            self.logger.info(f"Registered model: {metadata.id} v{metadata.version}")
            return metadata.id
            
        except Exception as e:
            self.logger.error(f"Failed to register model: {e}")
            self.db_session.rollback()
            raise
    
    def load_model(self, model_id: str, version: str = "latest") -> Tuple[Any, ModelMetadata]:
        """Load model from registry"""
        try:
            # Try to get from cache first
            cache_key = f"model:{model_id}:{version}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                metadata_dict = json.loads(cached_data)
                metadata = ModelMetadata(**metadata_dict)
            else:
                # Query from database
                if version == "latest":
                    db_model = (self.db_session.query(ModelRegistry)
                               .filter(ModelRegistry.id == model_id)
                               .order_by(ModelRegistry.created_at.desc())
                               .first())
                else:
                    db_model = (self.db_session.query(ModelRegistry)
                               .filter(ModelRegistry.id == model_id)
                               .filter(ModelRegistry.version == version)
                               .first())
                
                if not db_model:
                    raise ValueError(f"Model not found: {model_id} v{version}")
                
                # Convert to metadata
                metadata = ModelMetadata(
                    id=db_model.id,
                    name=db_model.name,
                    version=db_model.version,
                    algorithm=db_model.algorithm,
                    framework=db_model.framework,
                    created_at=db_model.created_at,
                    created_by=db_model.created_by,
                    description=db_model.description,
                    hyperparameters=json.loads(db_model.hyperparameters),
                    metrics=json.loads(db_model.metrics),
                    dataset_hash=db_model.dataset_hash,
                    features=json.loads(db_model.features),
                    target=db_model.target,
                    status=db_model.status,
                    file_path=db_model.file_path,
                    file_size=db_model.file_size,
                    checksum=db_model.checksum
                )
                
                # Cache metadata
                self.redis_client.setex(
                    cache_key,
                    3600,
                    json.dumps(asdict(metadata), default=str)
                )
            
            # Download and load model
            local_path = f"/tmp/{model_id}_{metadata.version}.pkl"
            if not self._download_model_from_s3(metadata.file_path, local_path):
                raise Exception("Failed to download model from S3")
            
            # Verify checksum
            if self._calculate_checksum(local_path) != metadata.checksum:
                raise Exception("Model file checksum verification failed")
            
            # Load model object
            with open(local_path, 'rb') as f:
                model_object = pickle.load(f)
            
            self.logger.info(f"Loaded model: {model_id} v{metadata.version}")
            return model_object, metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def list_models(self, status: Optional[str] = None) -> List[ModelMetadata]:
        """List all models"""
        try:
            query = self.db_session.query(ModelRegistry)
            if status:
                query = query.filter(ModelRegistry.status == status)
            
            db_models = query.order_by(ModelRegistry.created_at.desc()).all()
            
            models = []
            for db_model in db_models:
                metadata = ModelMetadata(
                    id=db_model.id,
                    name=db_model.name,
                    version=db_model.version,
                    algorithm=db_model.algorithm,
                    framework=db_model.framework,
                    created_at=db_model.created_at,
                    created_by=db_model.created_by,
                    description=db_model.description,
                    hyperparameters=json.loads(db_model.hyperparameters),
                    metrics=json.loads(db_model.metrics),
                    dataset_hash=db_model.dataset_hash,
                    features=json.loads(db_model.features),
                    target=db_model.target,
                    status=db_model.status,
                    file_path=db_model.file_path,
                    file_size=db_model.file_size,
                    checksum=db_model.checksum
                )
                models.append(metadata)
            
            return models
            
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            raise
    
    def update_model_status(self, model_id: str, version: str, status: str):
        """Update model status"""
        try:
            db_model = (self.db_session.query(ModelRegistry)
                       .filter(ModelRegistry.id == model_id)
                       .filter(ModelRegistry.version == version)
                       .first())
            
            if not db_model:
                raise ValueError(f"Model not found: {model_id} v{version}")
            
            db_model.status = status
            self.db_session.commit()
            
            # Invalidate cache
            self.redis_client.delete(f"model:{model_id}:{version}")
            
            self.logger.info(f"Updated model status: {model_id} v{version} -> {status}")
            
        except Exception as e:
            self.logger.error(f"Failed to update model status: {e}")
            self.db_session.rollback()
            raise
    
    def deploy_model(
        self,
        model_id: str,
        version: str,
        environment: str,
        deployed_by: str
    ) -> str:
        """Deploy model to environment"""
        try:
            deployment_id = f"{model_id}_{version}_{environment}_{int(datetime.now().timestamp())}"
            
            # Create deployment record
            deployment = ModelDeployment(
                id=deployment_id,
                model_id=model_id,
                environment=environment,
                deployed_at=datetime.now(),
                deployed_by=deployed_by,
                status="active",
                endpoint_url=f"/api/v1/predict/{model_id}/{version}",
                health_check_url=f"/api/v1/health/{model_id}/{version}",
                last_health_check=datetime.now(),
                performance_metrics=json.dumps({})
            )
            
            self.db_session.add(deployment)
            self.db_session.commit()
            
            # Update model status
            if environment == "production":
                self.update_model_status(model_id, version, "production")
            
            MODEL_DEPLOYMENTS.inc()
            
            self.logger.info(f"Deployed model: {deployment_id}")
            return deployment_id
            
        except Exception as e:
            self.logger.error(f"Failed to deploy model: {e}")
            self.db_session.rollback()
            raise
    
    def get_production_models(self) -> List[Tuple[str, str]]:
        """Get list of production models"""
        try:
            deployments = (self.db_session.query(ModelDeployment)
                          .filter(ModelDeployment.environment == "production")
                          .filter(ModelDeployment.status == "active")
                          .all())
            
            return [(d.model_id, d.model_id.split('_')[-1]) for d in deployments]
            
        except Exception as e:
            self.logger.error(f"Failed to get production models: {e}")
            raise
    
    def cleanup_old_models(self, days_threshold: int = 30):
        """Cleanup old models that are not in production"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_threshold)
            
            old_models = (self.db_session.query(ModelRegistry)
                         .filter(ModelRegistry.created_at < cutoff_date)
                         .filter(ModelRegistry.status != "production")
                         .all())
            
            for model in old_models:
                # Delete from S3
                try:
                    self.s3_client.delete_object(
                        Bucket=self.model_bucket,
                        Key=model.file_path
                    )
                except ClientError:
                    pass  # File might already be deleted
                
                # Delete from database
                self.db_session.delete(model)
            
            self.db_session.commit()
            
            self.logger.info(f"Cleaned up {len(old_models)} old models")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old models: {e}")
            self.db_session.rollback()
            raise

# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'database_url': 'postgresql://nicegold:password@localhost/nicegold',
        'redis_host': 'localhost',
        'redis_port': 6379,
        'mlflow_uri': 'http://localhost:5000',
        's3_endpoint': 'http://localhost:9000',
        's3_access_key': 'minioadmin',
        's3_secret_key': 'minioadmin',
        'model_bucket': 'nicegold-models'
    }
    
    # Initialize MLOps manager
    mlops = MLOpsManager(config)
    
    # Example: Register a dummy model
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    
    # Train a simple model
    X = pd.DataFrame(np.random.randn(100, 5), columns=['f1', 'f2', 'f3', 'f4', 'f5'])
    y = np.random.choice([0, 1], 100)
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    
    # Create metadata
    metadata = ModelMetadata(
        id="xauusd_predictor",
        name="XAUUSD Prediction Model",
        version="1.0.0",
        algorithm="RandomForest",
        framework="sklearn",
        created_at=datetime.now(),
        created_by="mlops_system",
        description="Random Forest model for XAUUSD price prediction",
        hyperparameters={"n_estimators": 100, "random_state": 42},
        metrics={"accuracy": 0.85, "precision": 0.82, "recall": 0.88},
        dataset_hash="abc123",
        features=list(X.columns),
        target="signal",
        status="staging",
        file_path="",
        file_size=0,
        checksum=""
    )
    
    # Register model
    model_id = mlops.register_model(model, metadata)
    print(f"Registered model: {model_id}")
    
    # Load model
    loaded_model, loaded_metadata = mlops.load_model(model_id)
    print(f"Loaded model: {loaded_metadata.name} v{loaded_metadata.version}")
    
    # Deploy to production
    deployment_id = mlops.deploy_model(model_id, "1.0.0", "production", "mlops_system")
    print(f"Deployed model: {deployment_id}")
