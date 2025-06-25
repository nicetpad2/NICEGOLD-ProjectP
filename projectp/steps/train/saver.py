from fastapi import FastAPI, HTTPException
from pathlib import Path
from projectp.pro_log import pro_log
from pydantic import BaseModel
from rich.console import Console
from typing import Any, Dict, List, Optional
from typing import List, Dict, Any
import joblib
import json
                import mlflow
import numpy as np
import os
import pandas as pd
            import shutil
    import uvicorn
"""
Model Saving Module
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
Handles model persistence, serialization, and deployment preparation
"""


console = Console()

class ModelSaver:
    """Model saving and serialization for deployment"""

    def __init__(self, output_dir: str = "output_default"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok = True)
        self.models_dir = self.output_dir / "models"
        self.models_dir.mkdir(exist_ok = True)

    def save_model(self, model: Any, model_name: str = "best_model", 
                  features: Optional[List[str]] = None, 
                  metrics: Optional[Dict[str, float]] = None) -> Dict[str, str]:
        """Save model with metadata"""
        try:
            saved_files = {}

            # Save model file
            model_path = self.models_dir / f"{model_name}.pkl"
            joblib.dump(model, model_path)
            saved_files['model'] = str(model_path)
            pro_log(f"[ModelSaver] Model saved to {model_path}", tag = "Save")

            # Save features list
            if features:
                features_path = self.models_dir / f"{model_name}_features.txt"
                with open(features_path, 'w', encoding = 'utf - 8') as f:
                    for feature in features:
                        f.write(f"{feature}\n")
                saved_files['features'] = str(features_path)
                pro_log(f"[ModelSaver] Features saved to {features_path}", tag = "Save")

            # Save model metadata
            metadata = {
                'model_name': model_name, 
                'model_type': str(type(model).__name__), 
                'features_count': len(features) if features else 0, 
                'features': features or [], 
                'metrics': metrics or {}, 
                'save_timestamp': pd.Timestamp.now().isoformat(), 
                'model_file': str(model_path), 
                'features_file': str(features_path) if features else None
            }

            metadata_path = self.models_dir / f"{model_name}_metadata.json"
            with open(metadata_path, 'w', encoding = 'utf - 8') as f:
                json.dump(metadata, f, indent = 2)
            saved_files['metadata'] = str(metadata_path)

            pro_log(f"[ModelSaver] Metadata saved to {metadata_path}", tag = "Save")
            return saved_files

        except Exception as e:
            pro_log(f"[ModelSaver] Model saving failed: {e}", level = "error", tag = "Save")
            raise

    def save_for_serving(self, model: Any, features: List[str], 
                        model_name: str = "serving_model") -> Dict[str, str]:
        """Save model specifically for serving/API deployment"""
        try:
            # Create serving directory
            serving_dir = self.output_dir / "serving"
            serving_dir.mkdir(exist_ok = True)

            saved_files = {}

            # Save model in multiple formats for compatibility

            # 1. Joblib format (primary)
            joblib_path = serving_dir / f"{model_name}.pkl"
            joblib.dump(model, joblib_path)
            saved_files['joblib'] = str(joblib_path)

            # 2. Try to save in MLflow format if available
            try:
                mlflow_path = serving_dir / f"{model_name}_mlflow"

                # Prepare sample data for signature
                sample_data = pd.DataFrame({feature: [0.0] for feature in features})
                signature = mlflow.models.infer_signature(sample_data, model.predict(sample_data))

                if hasattr(mlflow, 'catboost') and 'catboost' in str(type(model)).lower():
                    mlflow.catboost.save_model(model, str(mlflow_path), signature = signature)
                elif hasattr(mlflow, 'lightgbm') and 'lightgbm' in str(type(model)).lower():
                    mlflow.lightgbm.save_model(model, str(mlflow_path), signature = signature)
                elif hasattr(mlflow, 'xgboost') and 'xgboost' in str(type(model)).lower():
                    mlflow.xgboost.save_model(model, str(mlflow_path), signature = signature)
                else:
                    mlflow.sklearn.save_model(model, str(mlflow_path), signature = signature)

                saved_files['mlflow'] = str(mlflow_path)
                pro_log(f"[ModelSaver] MLflow model saved to {mlflow_path}", tag = "Save")

            except ImportError:
                pro_log("[ModelSaver] MLflow not available, skipping MLflow format", level = "warn", tag = "Save")
            except Exception as e:
                pro_log(f"[ModelSaver] MLflow saving failed: {e}", level = "warn", tag = "Save")

            # 3. Save serving configuration
            serving_config = {
                'model_name': model_name, 
                'model_path': str(joblib_path), 
                'features': features, 
                'feature_count': len(features), 
                'model_type': str(type(model).__name__), 
                'serving_timestamp': pd.Timestamp.now().isoformat(), 
                'api_version': '1.0', 
                'input_schema': {feature: 'float' for feature in features}, 
                'output_schema': {'prediction': 'int', 'probability': 'float'}
            }

            config_path = serving_dir / f"{model_name}_config.json"
            with open(config_path, 'w', encoding = 'utf - 8') as f:
                json.dump(serving_config, f, indent = 2)
            saved_files['config'] = str(config_path)

            # 4. Create simple API script template
            api_script = self._create_api_script_template(model_name, features)
            api_path = serving_dir / f"{model_name}_api.py"
            with open(api_path, 'w', encoding = 'utf - 8') as f:
                f.write(api_script)
            saved_files['api_script'] = str(api_path)

            pro_log(f"[ModelSaver] Serving files saved to {serving_dir}", tag = "Save")
            return saved_files

        except Exception as e:
            pro_log(f"[ModelSaver] Serving preparation failed: {e}", level = "error", tag = "Save")
            raise

    def _create_api_script_template(self, model_name: str, features: List[str]) -> str:
        """Create API script template for model serving"""
        return f'''#!/usr/bin/env python3
"""
Auto - generated API script for model serving
Model: {model_name}
Features: {len(features)} features
Generated: {pd.Timestamp.now().isoformat()}
"""


# Load model
MODEL_PATH = "{model_name}.pkl"
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {{MODEL_PATH}}")
except Exception as e:
    print(f"Failed to load model: {{e}}")
    model = None

# Define features
FEATURES = {features}

# Create FastAPI app
app = FastAPI(
    title = "{model_name.title()} Prediction API", 
    description = "Auto - generated API for ML model predictions", 
    version = "1.0.0"
)

class PredictionInput(BaseModel):
    """Input data for prediction"""
    {chr(10).join([f'    {feature}: float = 0.0' for feature in features[:10]])}  # Truncated for readability

    class Config:
        schema_extra = {{
            "example": {{{', '.join([f'"{feature}": 0.0' for feature in features[:5]])}}}
        }}

class PredictionOutput(BaseModel):
    """Output prediction result"""
    prediction: int
    probability: float
    confidence: str

@app.get("/")
async def root():
    return {{"message": "Model API is running", "model": "{model_name}", "features": len(FEATURES)}}

@app.get("/health")
async def health_check():
    return {{"status": "healthy", "model_loaded": model is not None}}

@app.post("/predict", response_model = PredictionOutput)
async def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code = 500, detail = "Model not loaded")

    try:
        # Convert input to DataFrame
        data_dict = input_data.dict()
        df = pd.DataFrame([data_dict])

        # Ensure all features are present
        for feature in FEATURES:
            if feature not in df.columns:
                df[feature] = 0.0

        # Select and order features
        df = df[FEATURES]

        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0, 1] if hasattr(model, 'predict_proba') else 0.5

        # Determine confidence
        confidence = "high" if abs(probability - 0.5) > 0.3 else "medium" if abs(probability - 0.5) > 0.1 else "low"

        return PredictionOutput(
            prediction = int(prediction), 
            probability = float(probability), 
            confidence = confidence
        )

    except Exception as e:
        raise HTTPException(status_code = 400, detail = f"Prediction failed: {{str(e)}}")

if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 8000)
'''

    def save_test_predictions(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                            output_name: str = "test_predictions") -> str:
        """Save test predictions for threshold optimization"""
        try:
            # Generate predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

            # Create predictions DataFrame
            predictions_df = pd.DataFrame({
                'actual': y_test, 
                'prediction': y_pred, 
                'probability': y_pred_proba
            })

            # Add original index if available
            if hasattr(X_test, 'index'):
                predictions_df.index = X_test.index

            # Save to CSV
            predictions_path = self.output_dir / f"{output_name}.csv"
            predictions_df.to_csv(predictions_path, index = True)

            pro_log(f"[ModelSaver] Test predictions saved to {predictions_path}", tag = "Save")
            return str(predictions_path)

        except Exception as e:
            pro_log(f"[ModelSaver] Test predictions saving failed: {e}", level = "error", tag = "Save")
            raise

    def create_deployment_package(self, model: Any, features: List[str], 
                                metrics: Dict[str, float], 
                                package_name: str = "deployment_package") -> str:
        """Create complete deployment package"""
        try:
            # Create package directory
            package_dir = self.output_dir / package_name
            package_dir.mkdir(exist_ok = True)

            # Save all components
            model_files = self.save_model(model, "production_model", features, metrics)
            serving_files = self.save_for_serving(model, features, "production_model")

            # Copy files to package directory

            for file_type, file_path in {**model_files, **serving_files}.items():
                if os.path.exists(file_path):
                    dest_path = package_dir / os.path.basename(file_path)
                    shutil.copy2(file_path, dest_path)

            # Create README
            readme_content = self._create_deployment_readme(features, metrics)
            readme_path = package_dir / "README.md"
            with open(readme_path, 'w', encoding = 'utf - 8') as f:
                f.write(readme_content)

            # Create requirements.txt
            requirements = [
                "pandas> = 1.3.0", 
                "numpy> = 1.21.0", 
                "scikit - learn> = 1.0.0", 
                "joblib> = 1.0.0", 
                "fastapi> = 0.70.0", 
                "uvicorn> = 0.15.0", 
                "pydantic> = 1.8.0"
            ]

            requirements_path = package_dir / "requirements.txt"
            with open(requirements_path, 'w', encoding = 'utf - 8') as f:
                f.write('\n'.join(requirements))

            pro_log(f"[ModelSaver] Deployment package created at {package_dir}", tag = "Save")
            return str(package_dir)

        except Exception as e:
            pro_log(f"[ModelSaver] Deployment package creation failed: {e}", level = "error", tag = "Save")
            raise

    def _create_deployment_readme(self, features: List[str], metrics: Dict[str, float]) -> str:
        """Create deployment README"""
        return f'''# Model Deployment Package

## Model Information
- **Features**: {len(features)} features
- **Model Type**: Production ML Model
- **Performance**: AUC = {metrics.get('test_auc', 'N/A'):.3f}, Accuracy = {metrics.get('test_accuracy', 'N/A'):.3f}
- **Generated**: {pd.Timestamp.now().isoformat()}

## Files Included
- `production_model.pkl` - Main model file
- `production_model_features.txt` - Features list
- `production_model_metadata.json` - Model metadata
- `production_model_config.json` - Serving configuration
- `production_model_api.py` - API script for serving
- `requirements.txt` - Python dependencies

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test Model Loading
```python
model = joblib.load('production_model.pkl')
print("Model loaded successfully!")
```

### 3. Start API Server
```bash
python production_model_api.py
```

### 4. Test API
```bash
curl -X POST "http://localhost:8000/predict" \\
     -H "Content - Type: application/json" \\
     -d '{{"feature1": 0.5, "feature2": 1.0}}'
```

## Features
{chr(10).join([f"- {feature}" for feature in features[:20]])}
{"..." if len(features) > 20 else ""}

## Performance Metrics
{chr(10).join([f"- {metric}: {value:.4f}" for metric, value in metrics.items()])}

## API Endpoints
- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /predict` - Make predictions

## Support
For issues or questions, please refer to the project documentation.
'''

    def verify_saved_model(self, model_path: str, features: List[str]) -> bool:
        """Verify that saved model can be loaded and used"""
        try:
            # Load model
            loaded_model = joblib.load(model_path)

            # Create test data
            test_data = pd.DataFrame({feature: [0.0] for feature in features})

            # Test prediction
            prediction = loaded_model.predict(test_data)

            if hasattr(loaded_model, 'predict_proba'):
                probability = loaded_model.predict_proba(test_data)

            pro_log(f"[ModelSaver] Model verification successful: {model_path}", tag = "Save")
            return True

        except Exception as e:
            pro_log(f"[ModelSaver] Model verification failed: {e}", level = "error", tag = "Save")
            return False