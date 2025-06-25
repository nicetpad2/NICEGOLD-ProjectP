# -*- coding: utf - 8 -* - 
#!/usr/bin/env python3
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
from sklearn.model_selection import (
from sklearn.preprocessing import RobustScaler, StandardScaler
from typing import Any, Dict, List, Optional, Tuple, Union
    import catboost as cb
import joblib
        import json
import logging
import numpy as np
    import optuna
import os
import pandas as pd
import pickle
import warnings
    import xgboost as xgb
"""
Production ML Pipeline for NICEGOLD ProjectP
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

Complete machine learning pipeline with:
- Data preprocessing and validation
- Feature engineering
- Model training and optimization
- Performance evaluation
- Model persistence
- Production - ready deployment

Author: NICEGOLD Team
Version: 3.0 Production
Created: 2025 - 06 - 24
"""


    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    f1_score, 
    precision_score, 
    recall_score, 
    roc_auc_score, 
    roc_curve, 
)
    GridSearchCV, 
    StratifiedKFold, 
    TimeSeriesSplit, 
    cross_val_score, 
    train_test_split, 
)

# Optional advanced ML libraries
try:

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class ProductionMLPipeline:
    """
    Production - ready ML pipeline for XAUUSD trading prediction.

    Features:
    - Multiple ML algorithms (RF, XGBoost, CatBoost, Logistic Regression)
    - Hyperparameter optimization with Optuna
    - Time series aware cross - validation
    - Comprehensive model evaluation
    - Model persistence and versioning
    - Production deployment ready
    """

    def __init__(
        self, 
        output_dir: str = "output_default", 
        model_dir: str = "models", 
        random_state: int = 42, 
        verbose: bool = True, 
    ):
        """
        Initialize the ML pipeline.

        Args:
            output_dir: Directory for outputs
            model_dir: Directory for saved models
            random_state: Random seed for reproducibility
            verbose: Enable verbose logging
        """
        self.output_dir = Path(output_dir)
        self.model_dir = Path(model_dir)
        self.random_state = random_state
        self.verbose = verbose

        # Create directories
        self.output_dir.mkdir(parents = True, exist_ok = True)
        self.model_dir.mkdir(parents = True, exist_ok = True)

        # Pipeline state
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        self.training_history = []

        # Model configurations
        self.model_configs = {
            "random_forest": {
                "class": RandomForestClassifier, 
                "params": {
                    "n_estimators": 100, 
                    "max_depth": 10, 
                    "min_samples_split": 5, 
                    "min_samples_leaf": 2, 
                    "class_weight": "balanced", 
                    "random_state": self.random_state, 
                    "n_jobs": -1, 
                }, 
                "param_grid": {
                    "n_estimators": [50, 100, 200], 
                    "max_depth": [8, 10, 12, None], 
                    "min_samples_split": [2, 5, 10], 
                    "min_samples_leaf": [1, 2, 4], 
                }, 
            }, 
            "logistic_regression": {
                "class": LogisticRegression, 
                "params": {
                    "class_weight": "balanced", 
                    "random_state": self.random_state, 
                    "max_iter": 1000, 
                }, 
                "param_grid": {
                    "C": [0.1, 1.0, 10.0], 
                    "penalty": ["l1", "l2"], 
                    "solver": ["liblinear", "saga"], 
                }, 
            }, 
        }

        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.model_configs["xgboost"] = {
                "class": xgb.XGBClassifier, 
                "params": {
                    "n_estimators": 100, 
                    "max_depth": 6, 
                    "learning_rate": 0.1, 
                    "subsample": 0.8, 
                    "colsample_bytree": 0.8, 
                    "random_state": self.random_state, 
                    "eval_metric": "auc", 
                    "use_label_encoder": False, 
                }, 
                "param_grid": {
                    "n_estimators": [50, 100, 200], 
                    "max_depth": [4, 6, 8], 
                    "learning_rate": [0.05, 0.1, 0.2], 
                    "subsample": [0.7, 0.8, 0.9], 
                    "colsample_bytree": [0.7, 0.8, 0.9], 
                }, 
            }

        # Add CatBoost if available
        if CATBOOST_AVAILABLE:
            self.model_configs["catboost"] = {
                "class": cb.CatBoostClassifier, 
                "params": {
                    "iterations": 100, 
                    "depth": 6, 
                    "learning_rate": 0.1, 
                    "class_weights": [1, 1], 
                    "random_seed": self.random_state, 
                    "verbose": False, 
                }, 
                "param_grid": {
                    "iterations": [50, 100, 200], 
                    "depth": [4, 6, 8], 
                    "learning_rate": [0.05, 0.1, 0.2], 
                }, 
            }

    def validate_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Validate input data for ML training.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Validated X and y

        Raises:
            ValueError: If data validation fails
        """
        logger.info("Validating ML data...")

        # Check shapes
        if len(X) != len(y):
            raise ValueError(
                f"Shape mismatch: X has {len(X)} rows, y has {len(y)} rows"
            )

        # Check for sufficient data
        if len(X) < 1000:
            logger.warning(
                f"Limited data: {len(X)} samples. Results may be unreliable."
            )

        # Check target distribution
        target_counts = y.value_counts()
        minority_ratio = target_counts.min() / target_counts.sum()

        if minority_ratio < 0.05:
            logger.warning(
                f"Imbalanced target: minority class ratio = {minority_ratio:.3f}"
            )

        logger.info(f"Target distribution: {target_counts.to_dict()}")

        # Remove features with too many missing values
        missing_ratio = X.isnull().sum() / len(X)
        high_missing_features = missing_ratio[missing_ratio > 0.5].index.tolist()

        if high_missing_features:
            logger.warning(
                f"Removing {len(high_missing_features)} features with >50% missing values"
            )
            X = X.drop(columns = high_missing_features)

        # Remove constant features
        constant_features = X.columns[X.nunique() <= 1].tolist()
        if constant_features:
            logger.warning(f"Removing {len(constant_features)} constant features")
            X = X.drop(columns = constant_features)

        # Handle remaining missing values
        if X.isnull().sum().sum() > 0:
            logger.info("Filling remaining missing values with median")
            X = X.fillna(X.median())

        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        if X.isnull().sum().sum() > 0:
            X = X.fillna(X.median())

        logger.info(f"✅ Data validation completed. Final shape: {X.shape}")

        return X, y

    def prepare_features(
        self, X: pd.DataFrame, scaler_type: str = "robust"
    ) -> pd.DataFrame:
        """
        Prepare features for ML training.

        Args:
            X: Feature matrix
            scaler_type: Type of scaler ('standard', 'robust', 'none')

        Returns:
            Scaled feature matrix
        """
        # Filter out non - numeric columns and datetime columns
        numeric_columns = []
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                # Check if it's not a datetime column
                if not pd.api.types.is_datetime64_any_dtype(X[col]):
                    numeric_columns.append(col)

        # Keep only numeric columns
        X_numeric = X[numeric_columns].copy()

        logger.info(
            f"Filtered to {len(numeric_columns)} numeric features from {len(X.columns)} total columns"
        )

        if scaler_type == "none":
            return X_numeric

        logger.info(f"Scaling features using {scaler_type} scaler...")

        # Select scaler
        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        # Fit and transform
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_numeric), 
            index = X_numeric.index, 
            columns = X_numeric.columns, 
        )

        # Store scaler
        self.scalers[scaler_type] = scaler

        logger.info(f"✅ Feature scaling completed on {len(X_scaled.columns)} features")

        return X_scaled

    def train_single_model(
        self, 
        model_name: str, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_val: pd.DataFrame = None, 
        y_val: pd.Series = None, 
        optimize_hyperparams: bool = True, 
    ) -> Dict[str, Any]:
        """
        Train a single ML model.

        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            optimize_hyperparams: Whether to optimize hyperparameters

        Returns:
            Training results dictionary
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")

        logger.info(f"Training {model_name}...")

        config = self.model_configs[model_name]

        # Initialize model
        model = config["class"](**config["params"])

        # Hyperparameter optimization
        if optimize_hyperparams and OPTUNA_AVAILABLE:
            logger.info(f"Optimizing hyperparameters for {model_name}...")
            model = self._optimize_hyperparameters(model_name, X_train, y_train)
        elif optimize_hyperparams:
            logger.info(f"Using GridSearchCV for {model_name}...")
            model = self._grid_search_optimization(model_name, X_train, y_train)

        # Train model
        start_time = datetime.now()

        if model_name == "xgboost" and X_val is not None:
            # Use early stopping for XGBoost
            eval_set = [(X_train, y_train), (X_val, y_val)]
            model.fit(
                X_train, 
                y_train, 
                eval_set = eval_set, 
                early_stopping_rounds = 10, 
                verbose = False, 
            )
        else:
            model.fit(X_train, y_train)

        training_time = (datetime.now() - start_time).total_seconds()

        # Evaluate model
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)[:, 1]

        train_metrics = self._calculate_metrics(y_train, y_train_pred, y_train_proba)

        val_metrics = {}
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            y_val_proba = model.predict_proba(X_val)[:, 1]
            val_metrics = self._calculate_metrics(y_val, y_val_pred, y_val_proba)

        # Feature importance
        feature_importance = self._get_feature_importance(model, X_train.columns)

        # Store results
        results = {
            "model": model, 
            "model_name": model_name, 
            "training_time": training_time, 
            "train_metrics": train_metrics, 
            "val_metrics": val_metrics, 
            "feature_importance": feature_importance, 
            "hyperparameters": model.get_params(), 
        }

        self.models[model_name] = model
        self.feature_importance[model_name] = feature_importance
        self.performance_metrics[model_name] = {
            "train": train_metrics, 
            "val": val_metrics, 
        }

        logger.info(f"✅ {model_name} training completed")
        logger.info(f"📊 Train AUC: {train_metrics['auc']:.4f}")
        if val_metrics:
            logger.info(f"📊 Val AUC: {val_metrics['auc']:.4f}")

        return results

    def _optimize_hyperparameters(self, model_name: str, X: pd.DataFrame, y: pd.Series):
        """Optimize hyperparameters using Optuna."""
        config = self.model_configs[model_name]

        def objective(trial):
            # Suggest hyperparameters
            params = config["params"].copy()

            if model_name == "random_forest":
                params.update(
                    {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 200), 
                        "max_depth": trial.suggest_int("max_depth", 8, 15), 
                        "min_samples_split": trial.suggest_int(
                            "min_samples_split", 2, 10
                        ), 
                        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4), 
                    }
                )
            elif model_name == "xgboost":
                params.update(
                    {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 200), 
                        "max_depth": trial.suggest_int("max_depth", 4, 8), 
                        "learning_rate": trial.suggest_float(
                            "learning_rate", 0.05, 0.2
                        ), 
                        "subsample": trial.suggest_float("subsample", 0.7, 0.9), 
                        "colsample_bytree": trial.suggest_float(
                            "colsample_bytree", 0.7, 0.9
                        ), 
                    }
                )
            elif model_name == "catboost":
                params.update(
                    {
                        "iterations": trial.suggest_int("iterations", 50, 200), 
                        "depth": trial.suggest_int("depth", 4, 8), 
                        "learning_rate": trial.suggest_float(
                            "learning_rate", 0.05, 0.2
                        ), 
                    }
                )
            elif model_name == "logistic_regression":
                params.update(
                    {
                        "C": trial.suggest_float("C", 0.1, 10.0, log = True), 
                        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]), 
                        "solver": trial.suggest_categorical(
                            "solver", ["liblinear", "saga"]
                        ), 
                    }
                )

            # Create and evaluate model
            model = config["class"](**params)

            # Use time series cross - validation
            tscv = TimeSeriesSplit(n_splits = 3)
            scores = cross_val_score(model, X, y, cv = tscv, scoring = "roc_auc", n_jobs = -1)

            return scores.mean()

        # Create study
        study = optuna.create_study(direction = "maximize")
        study.optimize(objective, n_trials = 20, show_progress_bar = False)

        # Get best parameters
        best_params = config["params"].copy()
        best_params.update(study.best_params)

        logger.info(f"Best hyperparameters for {model_name}: {study.best_params}")
        logger.info(f"Best AUC: {study.best_value:.4f}")

        return config["class"](**best_params)

    def _grid_search_optimization(self, model_name: str, X: pd.DataFrame, y: pd.Series):
        """Optimize hyperparameters using GridSearchCV."""
        config = self.model_configs[model_name]

        # Use smaller parameter grid for faster execution
        param_grid = {}
        for key, values in config["param_grid"].items():
            if len(values) > 3:
                # Sample 3 values for faster grid search
                param_grid[key] = [values[0], values[len(values) // 2], values[ - 1]]
            else:
                param_grid[key] = values

        # Grid search with time series cross - validation
        tscv = TimeSeriesSplit(n_splits = 3)
        grid_search = GridSearchCV(
            config["class"](**config["params"]), 
            param_grid, 
            cv = tscv, 
            scoring = "roc_auc", 
            n_jobs = -1, 
            verbose = 0, 
        )

        grid_search.fit(X, y)

        logger.info(
            f"Best hyperparameters for {model_name}: {grid_search.best_params_}"
        )
        logger.info(f"Best AUC: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate performance metrics."""
        return {
            "accuracy": accuracy_score(y_true, y_pred), 
            "precision": precision_score(y_true, y_pred, average = "weighted"), 
            "recall": recall_score(y_true, y_pred, average = "weighted"), 
            "f1": f1_score(y_true, y_pred, average = "weighted"), 
            "auc": roc_auc_score(y_true, y_proba), 
        }

    def _get_feature_importance(
        self, model, feature_names: List[str]
    ) -> Dict[str, float]:
        """Extract feature importance from model."""
        try:
            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
            elif hasattr(model, "coef_"):
                importance = np.abs(model.coef_[0])
            else:
                return {}

            return dict(zip(feature_names, importance))
        except:
            return {}

    def train_ensemble(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        test_size: float = 0.2, 
        models_to_train: List[str] = None, 
        optimize_hyperparams: bool = True, 
    ) -> Dict[str, Any]:
        """
        Train ensemble of ML models.

        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of test set
            models_to_train: List of model names to train
            optimize_hyperparams: Whether to optimize hyperparameters

        Returns:
            Training results dictionary
        """
        logger.info("🚀 Starting ensemble training...")

        # Validate data
        X, y = self.validate_data(X, y)

        # Check for datetime columns before filtering
        has_datetime_column = "Time" in X.columns or "datetime" in X.columns

        # Prepare features (this will filter out datetime columns)
        X_scaled = self.prepare_features(X, scaler_type = "robust")

        # Split data with time series awareness
        if has_datetime_column:
            # Time series split
            split_idx = int(len(X_scaled) * (1 - test_size))
            X_train, X_test = X_scaled.iloc[:split_idx], X_scaled.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        else:
            # Random split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, 
                y, 
                test_size = test_size, 
                random_state = self.random_state, 
                stratify = y, 
            )

        # Further split training data for validation
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, 
            y_train, 
            test_size = 0.2, 
            random_state = self.random_state, 
            stratify = y_train, 
        )

        logger.info(
            f"Data split - Train: {len(X_train_split)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )

        # Train models
        if models_to_train is None:
            models_to_train = list(self.model_configs.keys())

        training_results = {}

        for model_name in models_to_train:
            try:
                result = self.train_single_model(
                    model_name, 
                    X_train_split, 
                    y_train_split, 
                    X_val, 
                    y_val, 
                    optimize_hyperparams = optimize_hyperparams, 
                )
                training_results[model_name] = result
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue

        # Evaluate on test set
        test_results = {}
        for model_name, model in self.models.items():
            try:
                y_test_pred = model.predict(X_test)
                y_test_proba = model.predict_proba(X_test)[:, 1]
                test_metrics = self._calculate_metrics(
                    y_test, y_test_pred, y_test_proba
                )
                test_results[model_name] = test_metrics

                logger.info(f"📊 {model_name} Test AUC: {test_metrics['auc']:.4f}")
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name} on test set: {e}")

        # Find best model
        best_model_name = max(test_results.keys(), key = lambda x: test_results[x]["auc"])
        best_model = self.models[best_model_name]

        logger.info(
            f"🏆 Best model: {best_model_name} (AUC: {test_results[best_model_name]['auc']:.4f})"
        )

        # Prepare final results
        ensemble_results = {
            "models": self.models, 
            "training_results": training_results, 
            "test_results": test_results, 
            "best_model_name": best_model_name, 
            "best_model": best_model, 
            "feature_importance": self.feature_importance, 
            "data_split": {
                "train_size": len(X_train_split), 
                "val_size": len(X_val), 
                "test_size": len(X_test), 
            }, 
            "target_distribution": {
                "train": y_train_split.value_counts().to_dict(), 
                "val": y_val.value_counts().to_dict(), 
                "test": y_test.value_counts().to_dict(), 
            }, 
        }

        # Save results
        self._save_ensemble_results(ensemble_results)

        logger.info("✅ Ensemble training completed!")

        return ensemble_results

    def _save_ensemble_results(self, results: Dict[str, Any]):
        """Save ensemble results to disk."""
        logger.info("Saving ensemble results...")

        # Save models
        for model_name, model in results["models"].items():
            model_path = self.model_dir / f"{model_name}_model.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Model saved: {model_path}")

        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            scaler_path = self.model_dir / f"{scaler_name}_scaler.joblib"
            joblib.dump(scaler, scaler_path)

        # Save feature importance
        feature_importance_df = pd.DataFrame(results["feature_importance"]).fillna(0)
        feature_importance_path = self.output_dir / "feature_importance.csv"
        feature_importance_df.to_csv(feature_importance_path)

        # Save performance metrics
        performance_df = pd.DataFrame(
            {
                model: {
                    "train_auc": results["training_results"][model]["train_metrics"][
                        "auc"
                    ], 
                    "val_auc": (
                        results["training_results"][model]["val_metrics"]["auc"]
                        if results["training_results"][model]["val_metrics"]
                        else np.nan
                    ), 
                    "test_auc": results["test_results"][model]["auc"], 
                    "train_accuracy": results["training_results"][model][
                        "train_metrics"
                    ]["accuracy"], 
                    "test_accuracy": results["test_results"][model]["accuracy"], 
                    "training_time": results["training_results"][model][
                        "training_time"
                    ], 
                }
                for model in results["models"].keys()
            }
        ).T

        performance_path = self.output_dir / "model_performance.csv"
        performance_df.to_csv(performance_path)

        # Save metadata
        metadata = {
            "best_model": results["best_model_name"], 
            "training_date": datetime.now().isoformat(), 
            "data_split": results["data_split"], 
            "target_distribution": results["target_distribution"], 
            "models_trained": list(results["models"].keys()), 
        }

        metadata_path = self.output_dir / "training_metadata.json"

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent = 2)

        logger.info(f"✅ Results saved to {self.output_dir}")

    def load_model(self, model_name: str) -> Any:
        """Load a saved model."""
        model_path = self.model_dir / f"{model_name}_model.joblib"
        if model_path.exists():
            return joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")

    def predict(
        self, X: pd.DataFrame, model_name: str = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using trained model.

        Args:
            X: Feature matrix
            model_name: Name of model to use (uses best model if None)

        Returns:
            Predictions and probabilities
        """
        if model_name is None:
            # Use best model
            metadata_path = self.output_dir / "training_metadata.json"
            if metadata_path.exists():

                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                model_name = metadata["best_model"]
            else:
                raise ValueError("No model specified and no metadata found")

        # Load model if not in memory
        if model_name not in self.models:
            self.models[model_name] = self.load_model(model_name)

        model = self.models[model_name]

        # Scale features if scaler exists
        scaler_path = self.model_dir / "robust_scaler.joblib"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            X_scaled = pd.DataFrame(
                scaler.transform(X), index = X.index, columns = X.columns
            )
        else:
            X_scaled = X

        # Make predictions
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]

        return predictions, probabilities


def train_production_models(
    df: pd.DataFrame, 
    target_column: str = "target", 
    output_dir: str = "output_default", 
    models_to_train: List[str] = None, 
    optimize_hyperparams: bool = True, 
) -> Dict[str, Any]:
    """
    Production wrapper function for model training.

    Args:
        df: DataFrame with features and target
        target_column: Name of target column
        output_dir: Output directory
        models_to_train: List of models to train
        optimize_hyperparams: Whether to optimize hyperparameters

    Returns:
        Training results
    """
    logger.info("🚀 Starting production model training...")

    # Prepare data
    feature_columns = [col for col in df.columns if col != target_column]
    X = df[feature_columns]
    y = df[target_column]

    # Initialize pipeline
    pipeline = ProductionMLPipeline(output_dir = output_dir)

    # Train ensemble
    results = pipeline.train_ensemble(
        X, y, models_to_train = models_to_train, optimize_hyperparams = optimize_hyperparams
    )

    logger.info("✅ Production model training completed!")

    return results


if __name__ == "__main__":
    # Test the production ML pipeline
    logger.info("Testing Production ML Pipeline...")

    # Create sample data
    np.random.seed(42)
    n_samples = 5000
    n_features = 20

    # Generate sample features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features), 
        columns = [f"feature_{i}" for i in range(n_features)], 
    )

    # Generate target with some relationship to features
    y = (
        X["feature_0"]
        + X["feature_1"]
        - X["feature_2"]
        + np.random.randn(n_samples) * 0.5
        > 0
    ).astype(int)

    df = X.copy()
    df["target"] = y

    # Train models
    results = train_production_models(
        df, 
        models_to_train = ["random_forest", "logistic_regression"], 
        optimize_hyperparams = False,  # Skip optimization for quick test
    )

    logger.info(f"✅ Test completed. Best model: {results['best_model_name']}")
    logger.info(
        f"📊 Test AUC: {results['test_results'][results['best_model_name']]['auc']:.4f}"
    )