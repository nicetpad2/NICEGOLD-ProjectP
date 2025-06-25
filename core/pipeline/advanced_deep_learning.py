#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NICEGOLD ProjectP - Advanced Deep Learning Module
Neural Networks, LSTM, CNN-LSTM, Transformers for Gold Trading
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import (
        GRU,
        LSTM,
        Conv1D,
        Dense,
        Dropout,
        Flatten,
        LayerNormalization,
        MaxPooling1D,
        MultiHeadAttention,
    )
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    # Fallback for demo purposes
    
    class Model:
        pass
    
    class Sequential:
        pass
    
    class LSTM:
        pass
    
    class GRU:
        pass
    
    class Dense:
        pass
    
    class Dropout:
        pass
    
    class MinMaxScaler:
        pass
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
    )
    from tensorflow.keras.layers import (
        GRU,
        LSTM,
        Attention,
        BatchNormalization,
        Concatenate,
        Conv1D,
        Dense,
        Dropout,
        Flatten,
        GlobalAveragePooling1D,
        Input,
        LayerNormalization,
        MaxPooling1D,
        MultiHeadAttention,
        Reshape,
    )
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.optimizers import Adam, RMSprop

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


class AdvancedDeepLearning:
    """
    🧠 Advanced Deep Learning System for Gold Trading
    - LSTM/GRU Networks สำหรับ Time Series
    - CNN-LSTM Hybrid Models
    - Transformer Architecture
    - Multi-task Learning
    - Ensemble Deep Models
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Advanced Deep Learning System"""
        self.config = config or self._get_default_config()
        self.models = {}
        self.scalers = {}
        self.trained_models = {}
        self.training_history = {}
        self.predictions = {}

        # Model architectures
        self.model_architectures = {
            "lstm": self._build_lstm_model,
            "gru": self._build_gru_model,
            "cnn_lstm": self._build_cnn_lstm_model,
            "transformer": self._build_transformer_model,
            "attention_lstm": self._build_attention_lstm_model,
            "ensemble": self._build_ensemble_model,
        }

        # Check TensorFlow availability
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Installing fallback models.")
            self.fallback_mode = True
        else:
            self.fallback_mode = False
            logger.info("TensorFlow available. Advanced deep learning enabled.")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for deep learning"""
        return {
            # Model settings
            "sequence_length": 60,  # Number of timesteps to look back
            "prediction_horizon": 1,  # Number of steps to predict ahead
            "feature_columns": ["open", "high", "low", "close", "volume"],
            "target_column": "close",
            # Training settings
            "train_test_split": 0.8,
            "validation_split": 0.2,
            "batch_size": 32,
            "epochs": 100,
            "early_stopping_patience": 10,
            "learning_rate": 0.001,
            # Model architectures to train
            "models_to_train": ["lstm", "gru", "cnn_lstm", "attention_lstm"],
            "ensemble_enabled": True,
            # LSTM/GRU settings
            "lstm_units": [64, 32],
            "dropout_rate": 0.2,
            "recurrent_dropout": 0.2,
            # CNN settings
            "cnn_filters": [64, 32],
            "cnn_kernel_size": 3,
            "cnn_pool_size": 2,
            # Transformer settings
            "transformer_heads": 8,
            "transformer_dim": 64,
            "transformer_layers": 2,
            # Advanced settings
            "use_attention": True,
            "use_batch_norm": True,
            "optimizer": "adam",
            "loss_function": "mse",
            "metrics": ["mae", "mape"],
            # Ensemble settings
            "ensemble_weights": "auto",  # 'auto' or custom weights
            "ensemble_method": "weighted_average",  # 'average', 'weighted_average', 'voting'
        }

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        🔧 Prepare data for deep learning training
        Create sequences for time series prediction
        """
        logger.info("Preparing data for deep learning models")

        # Extract features and target
        feature_cols = self.config["feature_columns"]
        target_col = self.config["target_column"]

        # Ensure required columns exist
        missing_cols = [col for col in feature_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")

        if target_col not in data.columns:
            raise ValueError(f"Missing target column: {target_col}")

        # Sort by date if available
        if "date" in data.columns:
            data = data.sort_values("date")

        # Scale features
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

        features_scaled = feature_scaler.fit_transform(data[feature_cols])
        target_scaled = target_scaler.fit_transform(data[[target_col]])

        # Store scalers
        self.scalers["features"] = feature_scaler
        self.scalers["target"] = target_scaler

        # Create sequences
        X, y = self._create_sequences(features_scaled, target_scaled.flatten())

        logger.info(
            f"Created {len(X)} sequences of length {self.config['sequence_length']}"
        )
        logger.info(f"Input shape: {X.shape}, Output shape: {y.shape}")

        return X, y

    def _create_sequences(
        self, features: np.ndarray, target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction"""
        seq_length = self.config["sequence_length"]
        pred_horizon = self.config["prediction_horizon"]

        X, y = [], []

        for i in range(seq_length, len(features) - pred_horizon + 1):
            X.append(features[i - seq_length : i])
            y.append(target[i : i + pred_horizon])

        return np.array(X), np.array(y)

    def _build_lstm_model(self, input_shape: Tuple[int, int]) -> Model:
        """🧠 Build LSTM model for time series prediction"""
        model = Sequential(
            [
                LSTM(
                    self.config["lstm_units"][0],
                    return_sequences=len(self.config["lstm_units"]) > 1,
                    dropout=self.config["dropout_rate"],
                    recurrent_dropout=self.config["recurrent_dropout"],
                    input_shape=input_shape,
                ),
            ]
        )

        # Add additional LSTM layers
        for i, units in enumerate(self.config["lstm_units"][1:], 1):
            return_sequences = i < len(self.config["lstm_units"]) - 1
            model.add(
                LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=self.config["dropout_rate"],
                    recurrent_dropout=self.config["recurrent_dropout"],
                )
            )

        # Add batch normalization if enabled
        if self.config["use_batch_norm"]:
            model.add(BatchNormalization())

        # Output layer
        model.add(Dense(self.config["prediction_horizon"], activation="linear"))

        return model

    def _build_gru_model(self, input_shape: Tuple[int, int]) -> Model:
        """🧠 Build GRU model for time series prediction"""
        model = Sequential(
            [
                GRU(
                    self.config["lstm_units"][0],
                    return_sequences=len(self.config["lstm_units"]) > 1,
                    dropout=self.config["dropout_rate"],
                    recurrent_dropout=self.config["recurrent_dropout"],
                    input_shape=input_shape,
                ),
            ]
        )

        # Add additional GRU layers
        for i, units in enumerate(self.config["lstm_units"][1:], 1):
            return_sequences = i < len(self.config["lstm_units"]) - 1
            model.add(
                GRU(
                    units,
                    return_sequences=return_sequences,
                    dropout=self.config["dropout_rate"],
                    recurrent_dropout=self.config["recurrent_dropout"],
                )
            )

        # Add batch normalization if enabled
        if self.config["use_batch_norm"]:
            model.add(BatchNormalization())

        # Output layer
        model.add(Dense(self.config["prediction_horizon"], activation="linear"))

        return model

    def _build_cnn_lstm_model(self, input_shape: Tuple[int, int]) -> Model:
        """🧠 Build CNN-LSTM hybrid model"""
        model = Sequential(
            [
                # CNN layers for feature extraction
                Conv1D(
                    filters=self.config["cnn_filters"][0],
                    kernel_size=self.config["cnn_kernel_size"],
                    activation="relu",
                    input_shape=input_shape,
                ),
                BatchNormalization(),
                MaxPooling1D(pool_size=self.config["cnn_pool_size"]),
                Dropout(self.config["dropout_rate"]),
                Conv1D(
                    filters=self.config["cnn_filters"][1],
                    kernel_size=self.config["cnn_kernel_size"],
                    activation="relu",
                ),
                BatchNormalization(),
                MaxPooling1D(pool_size=self.config["cnn_pool_size"]),
                Dropout(self.config["dropout_rate"]),
                # LSTM layers for sequence modeling
                LSTM(
                    self.config["lstm_units"][0],
                    return_sequences=len(self.config["lstm_units"]) > 1,
                    dropout=self.config["dropout_rate"],
                ),
            ]
        )

        # Add additional LSTM layers if specified
        for i, units in enumerate(self.config["lstm_units"][1:], 1):
            return_sequences = i < len(self.config["lstm_units"]) - 1
            model.add(
                LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=self.config["dropout_rate"],
                )
            )

        # Output layer
        model.add(Dense(self.config["prediction_horizon"], activation="linear"))

        return model

    def _build_attention_lstm_model(self, input_shape: Tuple[int, int]) -> Model:
        """🧠 Build LSTM with Attention mechanism"""
        inputs = Input(shape=input_shape)

        # LSTM layer
        lstm_out = LSTM(
            self.config["lstm_units"][0],
            return_sequences=True,
            dropout=self.config["dropout_rate"],
        )(inputs)

        # Attention mechanism (simplified)
        attention = Dense(1, activation="tanh")(lstm_out)
        attention = Flatten()(attention)
        attention = tf.keras.layers.Activation("softmax")(attention)
        attention = tf.keras.layers.RepeatVector(self.config["lstm_units"][0])(
            attention
        )
        attention = tf.keras.layers.Permute([2, 1])(attention)

        # Apply attention
        sent_representation = tf.keras.layers.Multiply()([lstm_out, attention])
        sent_representation = tf.keras.layers.Lambda(
            lambda xin: tf.keras.backend.sum(xin, axis=-2),
            output_shape=(self.config["lstm_units"][0],),
        )(sent_representation)

        # Output layer
        outputs = Dense(self.config["prediction_horizon"], activation="linear")(
            sent_representation
        )

        model = Model(inputs=inputs, outputs=outputs)
        return model

    def _build_transformer_model(self, input_shape: Tuple[int, int]) -> Model:
        """🧠 Build Transformer model for time series"""
        inputs = Input(shape=input_shape)

        # Multi-head attention
        attention_out = MultiHeadAttention(
            num_heads=self.config["transformer_heads"],
            key_dim=self.config["transformer_dim"],
        )(inputs, inputs)

        # Add & Norm
        attention_out = LayerNormalization()(inputs + attention_out)

        # Feed Forward
        ff_out = Dense(self.config["transformer_dim"], activation="relu")(attention_out)
        ff_out = Dense(input_shape[-1])(ff_out)

        # Add & Norm
        ff_out = LayerNormalization()(attention_out + ff_out)

        # Global pooling and output
        pooled = GlobalAveragePooling1D()(ff_out)
        outputs = Dense(self.config["prediction_horizon"], activation="linear")(pooled)

        model = Model(inputs=inputs, outputs=outputs)
        return model

    def _build_ensemble_model(self, input_shape: Tuple[int, int]) -> Dict[str, Model]:
        """🧠 Build ensemble of multiple models"""
        ensemble_models = {}

        # Build individual models
        for model_type in ["lstm", "gru", "cnn_lstm"]:
            if model_type in self.model_architectures:
                ensemble_models[model_type] = self.model_architectures[model_type](
                    input_shape
                )

        return ensemble_models

    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        🚀 Train deep learning models
        """
        if self.fallback_mode:
            return self._train_fallback_models(X, y)

        logger.info("Starting deep learning model training")

        # Split data
        split_idx = int(len(X) * self.config["train_test_split"])
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        input_shape = (X_train.shape[1], X_train.shape[2])
        results = {}

        # Train individual models
        for model_name in self.config["models_to_train"]:
            if model_name == "ensemble":
                continue

            logger.info(f"Training {model_name} model")

            try:
                # Build model
                model = self.model_architectures[model_name](input_shape)

                # Compile model
                optimizer = Adam(learning_rate=self.config["learning_rate"])
                model.compile(
                    optimizer=optimizer,
                    loss=self.config["loss_function"],
                    metrics=self.config["metrics"],
                )

                # Callbacks
                callbacks = [
                    EarlyStopping(
                        patience=self.config["early_stopping_patience"],
                        restore_best_weights=True,
                    ),
                    ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.0001),
                ]

                # Train model
                history = model.fit(
                    X_train,
                    y_train,
                    batch_size=self.config["batch_size"],
                    epochs=self.config["epochs"],
                    validation_split=self.config["validation_split"],
                    callbacks=callbacks,
                    verbose=0,
                )

                # Evaluate model
                train_loss = model.evaluate(X_train, y_train, verbose=0)
                test_loss = model.evaluate(X_test, y_test, verbose=0)

                # Store results
                self.trained_models[model_name] = model
                self.training_history[model_name] = history.history

                results[model_name] = {
                    "train_loss": (
                        train_loss[0] if isinstance(train_loss, list) else train_loss
                    ),
                    "test_loss": (
                        test_loss[0] if isinstance(test_loss, list) else test_loss
                    ),
                    "epochs_trained": len(history.history["loss"]),
                    "best_val_loss": min(
                        history.history.get("val_loss", [float("inf")])
                    ),
                }

                logger.info(
                    f"{model_name} training completed - Test Loss: {test_loss[0]:.6f}"
                )

            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                results[model_name] = {"error": str(e)}

        # Train ensemble if enabled
        if self.config["ensemble_enabled"] and len(results) > 1:
            logger.info("Training ensemble model")
            ensemble_result = self._train_ensemble(X_train, y_train, X_test, y_test)
            results["ensemble"] = ensemble_result

        logger.info(f"Deep learning training completed. Trained {len(results)} models.")
        return results

    def _train_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Train ensemble model combining individual models"""
        try:
            # Get predictions from individual models
            train_preds = []
            test_preds = []
            model_names = []

            for name, model in self.trained_models.items():
                train_pred = model.predict(X_train, verbose=0)
                test_pred = model.predict(X_test, verbose=0)

                train_preds.append(train_pred)
                test_preds.append(test_pred)
                model_names.append(name)

            if len(train_preds) == 0:
                return {"error": "No trained models available for ensemble"}

            # Simple ensemble averaging
            ensemble_train_pred = np.mean(train_preds, axis=0)
            ensemble_test_pred = np.mean(test_preds, axis=0)

            # Calculate ensemble performance
            train_mse = np.mean((y_train - ensemble_train_pred) ** 2)
            test_mse = np.mean((y_test - ensemble_test_pred) ** 2)

            # Store ensemble predictions
            self.predictions["ensemble"] = {
                "train": ensemble_train_pred,
                "test": ensemble_test_pred,
            }

            return {
                "train_loss": train_mse,
                "test_loss": test_mse,
                "component_models": model_names,
                "ensemble_method": "simple_average",
            }

        except Exception as e:
            logger.error(f"Error training ensemble: {str(e)}")
            return {"error": str(e)}

    def _train_fallback_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train simple fallback models when TensorFlow is not available"""
        logger.info("Training fallback models (TensorFlow not available)")

        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_absolute_error, mean_squared_error

            # Reshape data for sklearn
            X_reshaped = X.reshape(X.shape[0], -1)

            # Split data
            split_idx = int(len(X_reshaped) * self.config["train_test_split"])
            X_train, X_test = X_reshaped[:split_idx], X_reshaped[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            results = {}

            # Train Random Forest
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train.ravel())

            rf_train_pred = rf_model.predict(X_train)
            rf_test_pred = rf_model.predict(X_test)

            results["random_forest"] = {
                "train_loss": mean_squared_error(y_train, rf_train_pred),
                "test_loss": mean_squared_error(y_test, rf_test_pred),
                "train_mae": mean_absolute_error(y_train, rf_train_pred),
                "test_mae": mean_absolute_error(y_test, rf_test_pred),
            }

            # Train Linear Regression
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train.ravel())

            lr_train_pred = lr_model.predict(X_train)
            lr_test_pred = lr_model.predict(X_test)

            results["linear_regression"] = {
                "train_loss": mean_squared_error(y_train, lr_train_pred),
                "test_loss": mean_squared_error(y_test, lr_test_pred),
                "train_mae": mean_absolute_error(y_train, lr_train_pred),
                "test_mae": mean_absolute_error(y_test, lr_test_pred),
            }

            # Store fallback models
            self.trained_models["random_forest"] = rf_model
            self.trained_models["linear_regression"] = lr_model

            logger.info("Fallback models trained successfully")
            return results

        except Exception as e:
            logger.error(f"Error training fallback models: {str(e)}")
            return {"error": str(e)}

    def predict(self, X: np.ndarray, model_name: str = "best") -> np.ndarray:
        """
        🔮 Make predictions using trained models
        """
        if model_name == "best":
            # Find best model based on validation loss
            model_name = self._get_best_model()

        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found in trained models")

        model = self.trained_models[model_name]

        if self.fallback_mode:
            # Reshape for sklearn models
            X_reshaped = X.reshape(X.shape[0], -1)
            predictions = model.predict(X_reshaped)
        else:
            predictions = model.predict(X, verbose=0)

        # Inverse transform predictions if scaler is available
        if "target" in self.scalers:
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            predictions = self.scalers["target"].inverse_transform(predictions)
            if predictions.shape[1] == 1:
                predictions = predictions.flatten()

        return predictions

    def _get_best_model(self) -> str:
        """Get best performing model based on validation loss"""
        if not self.trained_models:
            raise ValueError("No trained models available")

        best_model = None
        best_loss = float("inf")

        # Check individual models
        for name, model in self.trained_models.items():
            if name in self.training_history:
                val_loss = min(
                    self.training_history[name].get("val_loss", [float("inf")])
                )
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model = name

        # Check ensemble if available
        if "ensemble" in self.predictions:
            ensemble_loss = getattr(self, "ensemble_test_loss", float("inf"))
            if ensemble_loss < best_loss:
                best_model = "ensemble"

        return best_model or list(self.trained_models.keys())[0]

    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of all trained models"""
        summary = {
            "total_models": len(self.trained_models),
            "available_models": list(self.trained_models.keys()),
            "best_model": self._get_best_model() if self.trained_models else None,
            "tensorflow_available": not self.fallback_mode,
            "config": self.config,
        }

        # Add training results if available
        if hasattr(self, "training_results"):
            summary["training_results"] = self.training_results

        return summary

    def save_models(self, directory: str) -> Dict[str, str]:
        """Save trained models to disk"""
        import os

        import joblib

        os.makedirs(directory, exist_ok=True)
        saved_models = {}

        for name, model in self.trained_models.items():
            try:
                if self.fallback_mode:
                    # Save sklearn models
                    model_path = os.path.join(directory, f"{name}_model.joblib")
                    joblib.dump(model, model_path)
                else:
                    # Save Keras models
                    model_path = os.path.join(directory, f"{name}_model.h5")
                    model.save(model_path)

                saved_models[name] = model_path
                logger.info(f"Saved {name} model to {model_path}")

            except Exception as e:
                logger.error(f"Error saving {name} model: {str(e)}")
                saved_models[name] = f"Error: {str(e)}"

        # Save scalers
        if self.scalers:
            scaler_path = os.path.join(directory, "scalers.joblib")
            joblib.dump(self.scalers, scaler_path)
            saved_models["scalers"] = scaler_path

        return saved_models
