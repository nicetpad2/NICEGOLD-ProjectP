#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NICEGOLD ProjectP - Data Leak Prevention System
Advanced protection against data leakage and overfitting
"""

import logging
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")


class DataLeakPrevention:
    """
    Advanced system to prevent data leakage and overfitting
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.removed_features = []
        self.suspicious_features = []
        
    def detect_future_leakage(self, df: pd.DataFrame) -> List[str]:
        """
        Detect features that might contain future information
        """
        future_keywords = [
            'future', 'next', 'forward', 'ahead', 'tomorrow',
            'lead', 'shift_-', '_-', 'lag_-'
        ]
        
        leaky_features = []
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in future_keywords):
                leaky_features.append(col)
                self.logger.warning(f"Potential future leak detected: {col}")
        
        return leaky_features
    
    def detect_perfect_predictors(self, X: pd.DataFrame, y: pd.Series, 
                                correlation_threshold: float = 0.95) -> List[str]:
        """
        Detect features with suspiciously high correlation to target
        """
        perfect_predictors = []
        
        for col in X.columns:
            if X[col].dtype in ['object', 'category']:
                continue
                
            try:
                corr = np.corrcoef(X[col].fillna(0), y)[0, 1]
                if not np.isnan(corr) and abs(corr) > correlation_threshold:
                    perfect_predictors.append(col)
                    self.logger.warning(f"Perfect predictor detected: {col} (corr={corr:.3f})")
            except:
                continue
        
        return perfect_predictors
    
    def detect_identical_distributions(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Detect features with identical distribution to target
        """
        identical_features = []
        
        for col in X.columns:
            if X[col].dtype in ['object', 'category']:
                continue
            
            try:
                # Check if feature values perfectly separate classes
                feature_vals = X[col].fillna(0)
                unique_vals = np.unique(feature_vals)
                
                if len(unique_vals) == 2 and len(np.unique(y)) == 2:
                    # Check if each unique value corresponds to one class
                    class_0_vals = feature_vals[y == 0]
                    class_1_vals = feature_vals[y == 1]
                    
                    if (len(np.unique(class_0_vals)) == 1 and 
                        len(np.unique(class_1_vals)) == 1 and
                        np.unique(class_0_vals)[0] != np.unique(class_1_vals)[0]):
                        identical_features.append(col)
                        self.logger.warning(f"Perfect separator detected: {col}")
            except:
                continue
        
        return identical_features
    
    def apply_temporal_constraints(self, df: pd.DataFrame, 
                                 date_column: str = 'Date') -> pd.DataFrame:
        """
        Apply temporal constraints to prevent look-ahead bias
        """
        self.logger.info("Applying temporal constraints...")
        
        df_clean = df.copy()
        
        if date_column in df_clean.columns:
            # Ensure chronological order
            df_clean[date_column] = pd.to_datetime(df_clean[date_column])
            df_clean = df_clean.sort_values(date_column).reset_index(drop=True)
            
            # Add lag to all computed features to prevent look-ahead
            feature_cols = [col for col in df_clean.columns 
                          if col not in [date_column, 'target', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            for col in feature_cols:
                if 'lag' not in col.lower() and 'shift' not in col.lower():
                    # Apply 1-period lag to prevent look-ahead
                    df_clean[col] = df_clean[col].shift(1)
        
        # Remove rows with NaN values created by lagging
        df_clean = df_clean.dropna()
        
        return df_clean
    
    def robust_feature_selection(self, X: pd.DataFrame, y: pd.Series, 
                                max_features: int = 50) -> Tuple[pd.DataFrame, List[str]]:
        """
        Robust feature selection to reduce noise and overfitting
        """
        self.logger.info(f"Selecting robust features (max: {max_features})...")
        
        # Remove constant features
        constant_features = [col for col in X.columns if X[col].nunique() <= 1]
        X_clean = X.drop(columns=constant_features)
        
        if constant_features:
            self.logger.info(f"Removed {len(constant_features)} constant features")
        
        # Remove highly correlated features
        X_clean = self._remove_correlated_features(X_clean)
        
        # Statistical feature selection
        if len(X_clean.columns) > max_features:
            selector = SelectKBest(f_classif, k=max_features)
            X_selected = selector.fit_transform(X_clean.fillna(0), y)
            selected_features = X_clean.columns[selector.get_support()].tolist()
            X_final = pd.DataFrame(X_selected, columns=selected_features, index=X_clean.index)
        else:
            X_final = X_clean
            selected_features = X_clean.columns.tolist()
        
        self.logger.info(f"Selected {len(selected_features)} features from {len(X.columns)}")
        return X_final, selected_features
    
    def _remove_correlated_features(self, X: pd.DataFrame, 
                                  threshold: float = 0.95) -> pd.DataFrame:
        """
        Remove highly correlated features
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols].fillna(0)
        
        if len(numeric_cols) <= 1:
            return X
        
        # Calculate correlation matrix
        corr_matrix = X_numeric.corr().abs()
        
        # Find highly correlated pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Identify features to remove
        to_remove = [column for column in upper_triangle.columns 
                    if any(upper_triangle[column] > threshold)]
        
        if to_remove:
            self.logger.info(f"Removing {len(to_remove)} highly correlated features")
            X_clean = X.drop(columns=to_remove)
        else:
            X_clean = X
        
        return X_clean
    
    def validate_temporal_consistency(self, X: pd.DataFrame, y: pd.Series, 
                                    n_splits: int = 5) -> Dict[str, Any]:
        """
        Validate model performance across time periods
        """
        self.logger.info("Validating temporal consistency...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        consistency_scores = []
        
        # Simple baseline model for consistency check
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Fill missing values
            X_train_filled = X_train.fillna(X_train.mean())
            X_val_filled = X_val.fillna(X_train.mean())  # Use train statistics
            
            try:
                model.fit(X_train_filled, y_train)
                score = model.score(X_val_filled, y_val)
                consistency_scores.append(score)
            except:
                continue
        
        if consistency_scores:
            consistency_result = {
                'mean_score': np.mean(consistency_scores),
                'std_score': np.std(consistency_scores),
                'consistent': np.std(consistency_scores) < 0.1,  # Less than 10% std
                'scores': consistency_scores
            }
        else:
            consistency_result = {
                'mean_score': 0,
                'std_score': 1,
                'consistent': False,
                'scores': []
            }
        
        return consistency_result
    
    def comprehensive_leak_check(self, X: pd.DataFrame, y: pd.Series, 
                                date_column: str = 'Date') -> Dict[str, Any]:
        """
        Comprehensive data leakage detection and prevention
        """
        self.logger.info("🔍 Running comprehensive leak detection...")
        
        leak_report = {
            'leaky_features': [],
            'removed_features': [],
            'warnings': [],
            'clean_data': None,
            'selected_features': []
        }
        
        # 1. Detect future leakage
        future_leaks = self.detect_future_leakage(X)
        leak_report['leaky_features'].extend(future_leaks)
        
        # 2. Detect perfect predictors
        perfect_predictors = self.detect_perfect_predictors(X, y)
        leak_report['leaky_features'].extend(perfect_predictors)
        
        # 3. Detect identical distributions
        identical_features = self.detect_identical_distributions(X, y)
        leak_report['leaky_features'].extend(identical_features)
        
        # 4. Remove identified leaky features
        all_leaky = list(set(leak_report['leaky_features']))
        X_clean = X.drop(columns=all_leaky, errors='ignore')
        leak_report['removed_features'] = all_leaky
        
        if all_leaky:
            self.logger.warning(f"Removed {len(all_leaky)} potentially leaky features")
        
        # 5. Apply temporal constraints if date column exists
        if date_column in X.columns:
            # Create temporary dataframe with target for temporal processing
            temp_df = X_clean.copy()
            temp_df['target'] = y
            temp_df = self.apply_temporal_constraints(temp_df, date_column)
            
            X_clean = temp_df.drop(columns=['target'])
            y_clean = temp_df['target']
        else:
            y_clean = y
        
        # 6. Robust feature selection
        X_final, selected_features = self.robust_feature_selection(X_clean, y_clean)
        
        # 7. Validate temporal consistency
        consistency = self.validate_temporal_consistency(X_final, y_clean)
        
        leak_report.update({
            'clean_data': X_final,
            'clean_target': y_clean,
            'selected_features': selected_features,
            'temporal_consistency': consistency,
            'final_feature_count': len(selected_features),
            'original_feature_count': len(X.columns)
        })
        
        # Summary
        if not all_leaky and consistency['consistent']:
            leak_report['status'] = 'CLEAN'
            self.logger.info("✅ Data leak check passed - data is clean")
        else:
            leak_report['status'] = 'WARNINGS'
            self.logger.warning("⚠️ Data leak warnings - check report")
        
        return leak_report
    
    def generate_leak_report(self, leak_report: Dict[str, Any]) -> str:
        """
        Generate detailed leak prevention report
        """
        report = f"""
# 🛡️ DATA LEAK PREVENTION REPORT

## 📊 SUMMARY
- **Status**: {leak_report['status']}
- **Original Features**: {leak_report['original_feature_count']}
- **Final Features**: {leak_report['final_feature_count']}
- **Removed Features**: {len(leak_report['removed_features'])}

## 🚨 LEAKY FEATURES DETECTED
"""
        
        if leak_report['leaky_features']:
            for i, feature in enumerate(leak_report['leaky_features'], 1):
                report += f"{i}. {feature}\n"
        else:
            report += "No leaky features detected ✅\n"
        
        report += f"""
## ⏰ TEMPORAL CONSISTENCY
- **Mean Score**: {leak_report['temporal_consistency']['mean_score']:.4f}
- **Score Std**: {leak_report['temporal_consistency']['std_score']:.4f}
- **Consistent**: {'✅ YES' if leak_report['temporal_consistency']['consistent'] else '❌ NO'}

## 🎯 SELECTED FEATURES ({len(leak_report['selected_features'])})
"""
        
        for i, feature in enumerate(leak_report['selected_features'], 1):
            report += f"{i}. {feature}\n"
        
        report += """
## ✅ PREVENTION MEASURES APPLIED
1. Future information leak detection
2. Perfect predictor identification
3. Temporal constraint enforcement
4. Feature correlation analysis
5. Robust feature selection
6. Temporal consistency validation

---
Generated by NICEGOLD Data Leak Prevention System
"""
        
        return report


class OverfittingPrevention:
    """
    Advanced overfitting prevention system
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def apply_regularization_config(self, model_type: str) -> Dict[str, Any]:
        """
        Get regularization configuration for different model types
        """
        configs = {
            'RandomForest': {
                'max_depth': 8,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'max_features': 'sqrt',
                'n_estimators': 100
            },
            'GradientBoosting': {
                'max_depth': 5,
                'learning_rate': 0.1,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'subsample': 0.8
            },
            'XGBoost': {
                'max_depth': 6,
                'learning_rate': 0.1,
                'min_child_weight': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0
            },
            'LightGBM': {
                'max_depth': 6,
                'learning_rate': 0.1,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0
            }
        }
        
        return configs.get(model_type, {})
    
    def early_stopping_config(self) -> Dict[str, Any]:
        """
        Configuration for early stopping
        """
        return {
            'validation_fraction': 0.1,
            'n_iter_no_change': 5,
            'tol': 1e-4
        }
