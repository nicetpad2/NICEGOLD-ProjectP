"""
Production-ready ML Pipeline with MLflow, Feature Store, and Model Monitoring
"""
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
import joblib
from datetime import datetime, timedelta
import redis
import asyncio
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ModelMetrics:
    accuracy: float
    auc: float
    f1_score: float
    precision: float
    recall: float
    timestamp: datetime

class ProductionMLPipeline:
    """Production-ready ML Pipeline"""
    
    def __init__(self, 
                 mlflow_tracking_uri: str,
                 redis_client: redis.Redis,
                 model_registry: str = "models",
                 experiment_name: str = "nicegold_trading"):
        
        self.mlflow_client = MlflowClient(mlflow_tracking_uri)
        self.redis_client = redis_client
        self.model_registry = model_registry
        self.experiment_name = experiment_name
        
        # Set MLflow tracking URI and experiment
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        self.logger = logging.getLogger(__name__)
    
    async def train_model(self, 
                         X_train: pd.DataFrame, 
                         y_train: pd.Series,
                         X_val: pd.DataFrame,
                         y_val: pd.Series,
                         model_params: Dict[str, Any]) -> str:
        """Train model with MLflow tracking"""
        
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_params(model_params)
            
            # Import and train model
            from catboost import CatBoostClassifier
            
            model = CatBoostClassifier(**model_params)
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                verbose=False,
                early_stopping_rounds=50
            )
            
            # Make predictions
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
            
            # Log metrics
            mlflow.log_metrics({
                "accuracy": metrics.accuracy,
                "auc": metrics.auc,
                "f1_score": metrics.f1_score,
                "precision": metrics.precision,
                "recall": metrics.recall
            })
            
            # Log model
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name=f"{self.model_registry}_catboost"
            )
            
            # Log feature importance
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            mlflow.log_text(
                feature_importance.to_csv(index=False),
                "feature_importance.csv"
            )
            
            # Cache model performance
            await self._cache_model_performance(run.info.run_id, metrics)
            
            return run.info.run_id
    
    async def deploy_model(self, run_id: str, stage: str = "Production") -> bool:
        """Deploy model to production"""
        try:
            # Get model version
            model_version = self.mlflow_client.get_latest_versions(
                f"{self.model_registry}_catboost", 
                stages=[stage]
            )[0]
            
            # Load model
            model_uri = f"models:/{self.model_registry}_catboost/{model_version.version}"
            model = mlflow.sklearn.load_model(model_uri)
            
            # Save to production cache
            model_bytes = joblib.dumps(model)
            self.redis_client.set(
                f"production_model_{stage.lower()}", 
                model_bytes,
                ex=86400  # 24 hours
            )
            
            # Update deployment status
            deployment_info = {
                "model_version": model_version.version,
                "run_id": run_id,
                "deployed_at": datetime.utcnow().isoformat(),
                "stage": stage
            }
            
            self.redis_client.hset(
                "deployment_status",
                stage.lower(),
                str(deployment_info)
            )
            
            self.logger.info(f"Model deployed to {stage}: version {model_version.version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model deployment failed: {e}")
            return False
    
    async def predict_with_monitoring(self, 
                                    features: pd.DataFrame,
                                    model_stage: str = "production") -> Tuple[np.ndarray, Dict[str, Any]]:
        """Make predictions with monitoring"""
        
        # Load model from cache
        model_bytes = self.redis_client.get(f"production_model_{model_stage}")
        if not model_bytes:
            raise ValueError(f"No model found for stage: {model_stage}")
        
        model = joblib.loads(model_bytes)
        
        # Make predictions
        start_time = datetime.utcnow()
        predictions = model.predict_proba(features)[:, 1]
        prediction_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Monitor predictions
        monitoring_data = {
            "prediction_count": len(predictions),
            "prediction_time_seconds": prediction_time,
            "avg_confidence": float(np.mean(predictions)),
            "min_confidence": float(np.min(predictions)),
            "max_confidence": float(np.max(predictions)),
            "timestamp": start_time.isoformat()
        }
        
        # Log monitoring data
        await self._log_prediction_monitoring(monitoring_data)
        
        return predictions, monitoring_data
    
    async def detect_data_drift(self, 
                               current_features: pd.DataFrame,
                               reference_features: pd.DataFrame,
                               threshold: float = 0.1) -> Dict[str, Any]:
        """Detect data drift between current and reference data"""
        
        from scipy.stats import wasserstein_distance
        
        drift_results = {}
        
        for column in current_features.columns:
            if column in reference_features.columns:
                # Calculate Wasserstein distance
                distance = wasserstein_distance(
                    current_features[column].dropna(),
                    reference_features[column].dropna()
                )
                
                drift_results[column] = {
                    "distance": distance,
                    "drift_detected": distance > threshold
                }
        
        # Calculate overall drift score
        overall_drift = np.mean([result["distance"] for result in drift_results.values()])
        
        drift_summary = {
            "overall_drift_score": overall_drift,
            "drift_detected": overall_drift > threshold,
            "feature_drifts": drift_results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Cache drift results
        self.redis_client.setex(
            "latest_drift_analysis",
            3600,  # 1 hour
            str(drift_summary)
        )
        
        return drift_summary
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba) -> ModelMetrics:
        """Calculate model performance metrics"""
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
        
        return ModelMetrics(
            accuracy=accuracy_score(y_true, y_pred),
            auc=roc_auc_score(y_true, y_pred_proba),
            f1_score=f1_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred),
            recall=recall_score(y_true, y_pred),
            timestamp=datetime.utcnow()
        )
    
    async def _cache_model_performance(self, run_id: str, metrics: ModelMetrics):
        """Cache model performance metrics"""
        performance_data = {
            "run_id": run_id,
            "accuracy": metrics.accuracy,
            "auc": metrics.auc,
            "f1_score": metrics.f1_score,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "timestamp": metrics.timestamp.isoformat()
        }
        
        self.redis_client.lpush("model_performance_history", str(performance_data))
        self.redis_client.ltrim("model_performance_history", 0, 100)  # Keep last 100 records
    
    async def _log_prediction_monitoring(self, monitoring_data: Dict[str, Any]):
        """Log prediction monitoring data"""
        self.redis_client.lpush("prediction_monitoring", str(monitoring_data))
        self.redis_client.ltrim("prediction_monitoring", 0, 1000)  # Keep last 1000 records

# Feature Store Implementation
class FeatureStore:
    """Production Feature Store"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
    
    async def store_features(self, 
                           features: pd.DataFrame, 
                           feature_group: str,
                           ttl: int = 3600):
        """Store features in the feature store"""
        
        feature_key = f"features:{feature_group}:{datetime.utcnow().strftime('%Y%m%d_%H')}"
        
        # Store as compressed pickle
        import pickle
        import gzip
        
        compressed_data = gzip.compress(pickle.dumps(features))
        self.redis_client.setex(feature_key, ttl, compressed_data)
        
        # Update feature metadata
        metadata = {
            "shape": features.shape,
            "columns": list(features.columns),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.redis_client.hset(f"feature_metadata:{feature_group}", feature_key, str(metadata))
        
        self.logger.info(f"Stored features: {feature_group} with shape {features.shape}")
    
    async def get_features(self, 
                          feature_group: str, 
                          timestamp: Optional[datetime] = None) -> pd.DataFrame:
        """Retrieve features from the feature store"""
        
        if timestamp is None:
            # Get latest features
            pattern = f"features:{feature_group}:*"
            keys = self.redis_client.keys(pattern)
            if not keys:
                raise ValueError(f"No features found for group: {feature_group}")
            
            feature_key = sorted(keys)[-1]  # Get latest
        else:
            feature_key = f"features:{feature_group}:{timestamp.strftime('%Y%m%d_%H')}"
        
        # Load features
        import pickle
        import gzip
        
        compressed_data = self.redis_client.get(feature_key)
        if not compressed_data:
            raise ValueError(f"Features not found: {feature_key}")
        
        features = pickle.loads(gzip.decompress(compressed_data))
        return features

# Model Registry with Auto-Retraining
class ModelRegistry:
    """Production Model Registry with Auto-Retraining"""
    
    def __init__(self, 
                 ml_pipeline: ProductionMLPipeline,
                 feature_store: FeatureStore,
                 retrain_threshold: float = 0.05):
        
        self.ml_pipeline = ml_pipeline
        self.feature_store = feature_store
        self.retrain_threshold = retrain_threshold
        self.logger = logging.getLogger(__name__)
    
    async def check_and_retrain(self, current_metrics: ModelMetrics):
        """Check if model needs retraining and trigger if necessary"""
        
        # Get historical performance
        performance_history = self.ml_pipeline.redis_client.lrange("model_performance_history", 0, 10)
        
        if len(performance_history) < 5:
            self.logger.info("Insufficient history for retrain check")
            return False
        
        # Calculate average historical AUC
        historical_aucs = []
        for perf in performance_history:
            import ast
            perf_data = ast.literal_eval(perf.decode() if isinstance(perf, bytes) else perf)
            historical_aucs.append(perf_data["auc"])
        
        avg_historical_auc = np.mean(historical_aucs)
        
        # Check if current performance is significantly worse
        performance_drop = avg_historical_auc - current_metrics.auc
        
        if performance_drop > self.retrain_threshold:
            self.logger.warning(f"Performance drop detected: {performance_drop:.4f}")
            await self._trigger_retraining()
            return True
        
        return False
    
    async def _trigger_retraining(self):
        """Trigger model retraining"""
        try:
            # Get latest training data
            training_features = await self.feature_store.get_features("training_data")
            
            # Split data (implement your logic)
            # train_test_split logic here
            
            # Trigger retraining
            run_id = await self.ml_pipeline.train_model(
                # Your training parameters
            )
            
            self.logger.info(f"Retraining triggered with run_id: {run_id}")
            
        except Exception as e:
            self.logger.error(f"Retraining failed: {e}")

# Usage Example
async def main():
    """Example usage of production ML pipeline"""
    
    # Initialize components
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    
    ml_pipeline = ProductionMLPipeline(
        mlflow_tracking_uri="http://localhost:5000",
        redis_client=redis_client
    )
    
    feature_store = FeatureStore(redis_client)
    model_registry = ModelRegistry(ml_pipeline, feature_store)
    
    # Example workflow
    # 1. Train model
    # 2. Deploy model
    # 3. Make predictions with monitoring
    # 4. Check for data drift
    # 5. Auto-retrain if needed

if __name__ == "__main__":
    asyncio.run(main())
