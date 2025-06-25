            from catboost import CatBoostClassifier
from projectp.pro_log import pro_log
from rich.console import Console
            from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score
from typing import List, Tuple, Dict, Any, Optional
            import json
import numpy as np
            import os
import pandas as pd
            import shap
"""
Feature Engineering Module
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
Handles feature selection, engineering, and optimization
"""


console = Console()

class FeatureEngineer:
    """Feature engineering and selection for ML models"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.selected_features = []
        self.dropped_features = []

    def remove_highly_correlated_features(self, X: pd.DataFrame, threshold: float = 0.95) -> Tuple[pd.DataFrame, List[str]]:
        """Remove features with high correlation"""
        try:
            corr_matrix = X.corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k = 1).astype(bool)
            )

            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

            X_filtered = X.drop(columns = to_drop, errors = 'ignore')

            pro_log(f"[FeatureEngineer] Dropped {len(to_drop)} highly correlated features", tag = "Features")
            self.dropped_features.extend(to_drop)

            return X_filtered, to_drop

        except Exception as e:
            pro_log(f"[FeatureEngineer] Correlation filtering failed: {e}", level = "warn", tag = "Features")
            return X, []

    def select_top_features_by_importance(self, X: pd.DataFrame, y: pd.Series, method: str = "mutual_info", top_n: int = 20) -> List[str]:
        """Select top N features by importance"""
        try:
            if method == "mutual_info":
                return self._select_by_mutual_info(X, y, top_n)
            elif method == "shap":
                return self._select_by_shap(X, y, top_n)
            else:
                pro_log(f"[FeatureEngineer] Unknown selection method: {method}", level = "warn", tag = "Features")
                return X.columns.tolist()[:top_n]

        except Exception as e:
            pro_log(f"[FeatureEngineer] Feature selection failed: {e}", level = "warn", tag = "Features")
            return X.columns.tolist()[:top_n]

    def _select_by_mutual_info(self, X: pd.DataFrame, y: pd.Series, top_n: int) -> List[str]:
        """Select features using mutual information"""
        try:

            mi_scores = mutual_info_classif(X, y, random_state = 42)
            mi_df = pd.DataFrame({
                'feature': X.columns, 
                'mi_score': mi_scores
            }).sort_values('mi_score', ascending = False)

            # Save MI scores
            os.makedirs("output_default", exist_ok = True)
            mi_df.to_csv("output_default/feature_mi.csv", index = False)

            top_features = mi_df.head(top_n)['feature'].tolist()
            pro_log(f"[FeatureEngineer] Selected top {top_n} features by mutual info", tag = "Features")

            return top_features

        except Exception as e:
            pro_log(f"[FeatureEngineer] Mutual info selection failed: {e}", level = "warn", tag = "Features")
            return X.columns.tolist()[:top_n]

    def _select_by_shap(self, X: pd.DataFrame, y: pd.Series, top_n: int) -> List[str]:
        """Select features using SHAP values"""
        try:

            # Train a quick model for SHAP
            model = CatBoostClassifier(iterations = 50, verbose = False, random_state = 42)
            model.fit(X, y)

            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            # Get feature importance from SHAP
            feature_importance = np.abs(shap_values).mean(0)
            shap_df = pd.DataFrame({
                'feature': X.columns, 
                'shap_importance': feature_importance
            }).sort_values('shap_importance', ascending = False)

            # Save SHAP scores
            os.makedirs("output_default", exist_ok = True)
            shap_df.to_csv("output_default/feature_shap.csv", index = False)

            top_features = shap_df.head(top_n)['feature'].tolist()
            pro_log(f"[FeatureEngineer] Selected top {top_n} features by SHAP", tag = "Features")

            return top_features

        except Exception as e:
            pro_log(f"[FeatureEngineer] SHAP selection failed: {e}", level = "warn", tag = "Features")
            return X.columns.tolist()[:top_n]

    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for better model performance"""
        try:
            df_enhanced = df.copy()

            # Technical indicators
            if 'Close' in df.columns:
                self._add_technical_indicators(df_enhanced)

            # Statistical features
            self._add_statistical_features(df_enhanced)

            # Interaction features (limited to avoid explosion)
            self._add_interaction_features(df_enhanced)

            pro_log(f"[FeatureEngineer] Created advanced features: {df.shape} -> {df_enhanced.shape}", tag = "Features")
            return df_enhanced

        except Exception as e:
            pro_log(f"[FeatureEngineer] Advanced feature creation failed: {e}", level = "warn", tag = "Features")
            return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> None:
        """Add technical trading indicators"""
        try:
            # Moving averages
            for window in [5, 10, 20]:
                df[f'sma_{window}'] = df['Close'].rolling(window).mean()
                df[f'ema_{window}'] = df['Close'].ewm(span = window).mean()

            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window = 14).mean()
            loss = ( - delta.where(delta < 0, 0)).rolling(window = 14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # MACD
            exp1 = df['Close'].ewm(span = 12).mean()
            exp2 = df['Close'].ewm(span = 26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span = 9).mean()

            pro_log("[FeatureEngineer] Added technical indicators", tag = "Features")

        except Exception as e:
            pro_log(f"[FeatureEngineer] Technical indicators failed: {e}", level = "warn", tag = "Features")

    def _add_statistical_features(self, df: pd.DataFrame) -> None:
        """Add statistical features"""
        try:
            numeric_cols = df.select_dtypes(include = [np.number]).columns[:5]  # Limit to first 5

            for col in numeric_cols:
                # Rolling statistics
                df[f'{col}_rolling_std'] = df[col].rolling(5).std().fillna(0)
                df[f'{col}_rolling_min'] = df[col].rolling(5).min().fillna(0)
                df[f'{col}_rolling_max'] = df[col].rolling(5).max().fillna(0)

                # Lag features
                df[f'{col}_lag1'] = df[col].shift(1).fillna(0)
                df[f'{col}_lag2'] = df[col].shift(2).fillna(0)

                # Percentage change
                df[f'{col}_pct_change'] = df[col].pct_change().fillna(0)

            pro_log("[FeatureEngineer] Added statistical features", tag = "Features")

        except Exception as e:
            pro_log(f"[FeatureEngineer] Statistical features failed: {e}", level = "warn", tag = "Features")

    def _add_interaction_features(self, df: pd.DataFrame) -> None:
        """Add interaction features (limited set)"""
        try:
            numeric_cols = df.select_dtypes(include = [np.number]).columns[:3]  # Limit to prevent explosion

            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i + 1:]:
                    # Product interaction
                    df[f'{col1}_{col2}_product'] = df[col1] * df[col2]

                    # Ratio interaction (avoid division by zero)
                    df[f'{col1}_{col2}_ratio'] = df[col1] / (df[col2] + 1e - 8)

            pro_log("[FeatureEngineer] Added interaction features", tag = "Features")

        except Exception as e:
            pro_log(f"[FeatureEngineer] Interaction features failed: {e}", level = "warn", tag = "Features")

    def export_feature_info(self, features: List[str], output_dir: str = "output_default") -> None:
        """Export feature information for future reference"""
        try:

            os.makedirs(output_dir, exist_ok = True)

            # Export feature list
            features_path = os.path.join(output_dir, "train_features.txt")
            with open(features_path, "w", encoding = "utf - 8") as f:
                for feat in features:
                    f.write(f"{feat}\n")

            # Export feature metadata
            metadata = {
                'total_features': len(features), 
                'selected_features': features, 
                'dropped_features': self.dropped_features, 
                'feature_engineering_applied': True
            }

            metadata_path = os.path.join(output_dir, "feature_metadata.json")
            with open(metadata_path, "w", encoding = "utf - 8") as f:
                json.dump(metadata, f, indent = 2)

            pro_log(f"[FeatureEngineer] Exported feature info to {output_dir}", tag = "Features")

        except Exception as e:
            pro_log(f"[FeatureEngineer] Feature export failed: {e}", level = "warn", tag = "Features")