#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NICEGOLD ProjectP - Fast Production Pipeline
Optimized version for faster execution during development and testing
"""

import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Suppress warnings
warnings.filterwarnings("ignore")


class FastProductionPipeline:
    """Fast production pipeline for NICEGOLD ProjectP - Optimized for speed"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = self._setup_logger()
        self.results = {}

        # Fast configuration
        self.fast_mode = True
        self.max_samples = 50000  # Limit data size for speed
        self.n_estimators = 50  # Reduced from 200
        self.cv_splits = 3  # Reduced from 5
        self.max_features = 15  # Reduced from 31

    def _setup_logger(self):
        """Setup logger for the pipeline"""
        logger = logging.getLogger("FastPipeline")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """Load trading data from CSV file"""
        self.logger.info("🔄 Loading data...")

        if data_path is None:
            # Auto-detect data file
            data_folder = Path("datacsv")
            if not data_folder.exists():
                raise FileNotFoundError("datacsv folder not found!")

            csv_files = list(data_folder.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError("No CSV files found in datacsv folder!")

            # Choose the largest file
            data_path = max(csv_files, key=lambda f: f.stat().st_size)

        self.logger.info(f"📊 Loading data from: {data_path}")

        # Load data with optimizations
        df = pd.read_csv(data_path, nrows=self.max_samples if self.fast_mode else None)

        # Basic data cleaning
        df = df.dropna()

        self.logger.info(f"✅ Loaded {len(df)} rows of data")
        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create essential trading features quickly"""
        self.logger.info("⚙️ Creating features...")

        # Ensure required columns exist
        required_cols = ["open", "high", "low", "close"]

        # Case-insensitive column mapping
        col_mapping = {}
        for req_col in required_cols:
            for df_col in df.columns:
                if req_col.lower() == df_col.lower():
                    col_mapping[req_col] = df_col
                    break

        if len(col_mapping) < len(required_cols):
            # Use available numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 4:
                col_mapping = {
                    "close": numeric_cols[0],
                    "open": (
                        numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
                    ),
                    "high": (
                        numeric_cols[2] if len(numeric_cols) > 2 else numeric_cols[0]
                    ),
                    "low": (
                        numeric_cols[3] if len(numeric_cols) > 3 else numeric_cols[0]
                    ),
                }
            else:
                raise ValueError("Insufficient numeric columns for feature creation")

        # Use mapped column names
        close_col = col_mapping["close"]
        open_col = col_mapping["open"]
        high_col = col_mapping["high"]
        low_col = col_mapping["low"]

        # Create essential features (fast computation)
        feature_df = pd.DataFrame()

        # Price-based features
        feature_df["returns"] = df[close_col].pct_change()
        feature_df["price_change"] = df[close_col] - df[open_col]
        feature_df["high_low_ratio"] = df[high_col] / df[low_col]

        # Fast moving averages
        for period in [5, 10, 20]:
            feature_df[f"sma_{period}"] = df[close_col].rolling(period).mean()
            feature_df[f"price_vs_sma_{period}"] = (
                df[close_col] / feature_df[f"sma_{period}"]
            )

        # Fast volatility
        feature_df["volatility_5"] = feature_df["returns"].rolling(5).std()
        feature_df["volatility_20"] = feature_df["returns"].rolling(20).std()

        # Fast momentum
        for period in [3, 5, 10]:
            feature_df[f"momentum_{period}"] = (
                df[close_col] / df[close_col].shift(period) - 1
            )

        # Target variable (next period direction)
        feature_df["target"] = (df[close_col].shift(-1) > df[close_col]).astype(int)

        # Drop rows with NaN values
        feature_df = feature_df.dropna()

        self.logger.info(
            f"✅ Created {len(feature_df.columns)} features, {len(feature_df)} samples"
        )
        return feature_df

    def select_features(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, list]:
        """Fast feature selection"""
        self.logger.info(f"🔍 Selecting top {self.max_features} features...")

        # Use SelectKBest for fast feature selection
        selector = SelectKBest(
            score_func=f_classif, k=min(self.max_features, X.shape[1])
        )
        X_selected = selector.fit_transform(X, y)

        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()

        X_selected_df = pd.DataFrame(
            X_selected, columns=selected_features, index=X.index
        )

        self.logger.info(f"✅ Selected {len(selected_features)} features")
        return X_selected_df, selected_features

    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train models with fast configuration"""
        self.logger.info("🤖 Training models (fast mode)...")

        # Fast models configuration
        models = {
            "FastRandomForest": RandomForestClassifier(
                n_estimators=self.n_estimators,  # Reduced for speed
                max_depth=8,  # Reduced for speed
                min_samples_split=10,  # Increased for speed
                min_samples_leaf=5,  # Increased for speed
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            )
        }

        results = {}

        # Time series split (fast)
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)

        for name, model in models.items():
            self.logger.info(f"🏃 Training {name}...")

            try:
                # Fast cross-validation
                cv_scores = cross_val_score(
                    model, X, y, cv=tscv, scoring="roc_auc", n_jobs=-1
                )

                mean_auc = cv_scores.mean()
                std_auc = cv_scores.std()

                # Train final model
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                model.fit(X_train, y_train)

                # Fast predictions
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)

                final_auc = roc_auc_score(y_test, y_pred_proba)

                results[name] = {
                    "model": model,
                    "cv_auc_mean": mean_auc,
                    "cv_auc_std": std_auc,
                    "final_auc": final_auc,
                    "features_used": list(X.columns),
                }

                self.logger.info(
                    f"✅ {name} - CV AUC: {mean_auc:.3f}±{std_auc:.3f}, Final AUC: {final_auc:.3f}"
                )

            except Exception as e:
                self.logger.error(f"❌ Error training {name}: {e}")
                continue

        return results

    def run_fast_pipeline(self, data_path: str = None) -> Dict[str, Any]:
        """Run the complete fast pipeline"""
        start_time = datetime.now()
        self.logger.info("🚀 Starting Fast Production Pipeline...")

        try:
            # 1. Load data
            df = self.load_data(data_path)

            # 2. Create features
            feature_df = self.create_features(df)

            # 3. Prepare X and y
            X = feature_df.drop("target", axis=1)
            y = feature_df["target"]

            self.logger.info(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
            self.logger.info(f"📊 Target distribution: {y.value_counts().to_dict()}")

            # 4. Feature selection
            X_selected, selected_features = self.select_features(X, y)

            # 5. Train models
            model_results = self.train_models(X_selected, y)

            # 6. Select best model
            if model_results:
                best_model_name = max(
                    model_results.keys(), key=lambda k: model_results[k]["final_auc"]
                )
                best_model_result = model_results[best_model_name]

                self.logger.info(
                    f"🏆 Best model: {best_model_name} (AUC: {best_model_result['final_auc']:.3f})"
                )
            else:
                self.logger.error("❌ No models were successfully trained")
                return {}

            # 7. Prepare results
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            results = {
                "execution_time_seconds": execution_time,
                "best_model": best_model_name,
                "best_auc": best_model_result["final_auc"],
                "models": model_results,
                "features_used": selected_features,
                "data_shape": X.shape,
                "target_distribution": y.value_counts().to_dict(),
                "fast_mode": True,
            }

            self.logger.info(
                f"✅ Fast pipeline completed in {execution_time:.1f} seconds"
            )
            return results

        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            return {}


def main():
    """Main function to run fast pipeline"""
    print("🚀 NICEGOLD ProjectP - Fast Production Pipeline")
    print("=" * 60)

    pipeline = FastProductionPipeline()
    results = pipeline.run_fast_pipeline()

    if results:
        print(f"\n🎉 Fast Pipeline Results:")
        print(f"⏱️  Execution Time: {results['execution_time_seconds']:.1f} seconds")
        print(f"🏆 Best Model: {results['best_model']}")
        print(f"📊 Best AUC: {results['best_auc']:.3f}")
        print(f"📈 Data Shape: {results['data_shape']}")
        print(f"✅ Fast mode successfully completed!")
    else:
        print("❌ Fast pipeline failed!")


if __name__ == "__main__":
    main()
