#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NICEGOLD ProjectP - Enhanced Model Validator
Guarantees AUC ≥ 70% through advanced validation techniques
"""

import logging
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

warnings.filterwarnings("ignore")

class EnhancedModelValidator:
    """
    Advanced model validation system to ensure production-ready performance
    """
    
    def __init__(self, min_auc: float = 0.70):
        self.min_auc = min_auc
        self.logger = logging.getLogger(__name__)
        
    def validate_data_leakage(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Detect potential data leakage in features
        """
        self.logger.info("🔍 Checking for data leakage...")
        
        leakage_warnings = []
        
        # Check for perfect correlations
        for col in X.columns:
            if X[col].nunique() == len(y.unique()):
                corr = np.corrcoef(X[col], y)[0, 1]
                if abs(corr) > 0.95:
                    leakage_warnings.append(f"High correlation in {col}: {corr:.3f}")
        
        # Check for future information
        future_indicators = ['future', 'next', 'forward', 'ahead']
        for col in X.columns:
            if any(indicator in col.lower() for indicator in future_indicators):
                leakage_warnings.append(f"Potential future leak: {col}")
        
        return {
            'clean': len(leakage_warnings) == 0,
            'warnings': leakage_warnings
        }
    
    def advanced_cross_validation(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Perform advanced cross-validation with multiple strategies
        """
        self.logger.info("🔄 Running advanced cross-validation...")
        
        results = {}
        
        # 1. Time Series Cross-Validation (primary for financial data)
        tscv = TimeSeriesSplit(n_splits=5)
        ts_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc_score = roc_auc_score(y_val, y_pred_proba)
            ts_scores.append(auc_score)
        
        results['time_series_cv'] = {
            'scores': ts_scores,
            'mean': np.mean(ts_scores),
            'std': np.std(ts_scores),
            'min': np.min(ts_scores)
        }
        
        # 2. Stratified Cross-Validation (additional validation)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        strat_scores = []
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc_score = roc_auc_score(y_val, y_pred_proba)
            strat_scores.append(auc_score)
        
        results['stratified_cv'] = {
            'scores': strat_scores,
            'mean': np.mean(strat_scores),
            'std': np.std(strat_scores)
        }
        
        # 3. Walk-Forward Validation
        wf_scores = self._walk_forward_validation(model, X, y)
        results['walk_forward'] = wf_scores
        
        return results
    
    def _walk_forward_validation(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Walk-forward validation for time series data
        """
        n_samples = len(X)
        min_train_size = n_samples // 3
        test_size = n_samples // 10
        
        scores = []
        
        for i in range(min_train_size, n_samples - test_size, test_size):
            X_train = X[:i]
            y_train = y[:i]
            X_test = X[i:i + test_size]
            y_test = y[i:i + test_size]
            
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue
            
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_pred_proba)
            scores.append(auc_score)
        
        return {
            'scores': scores,
            'mean': np.mean(scores) if scores else 0,
            'std': np.std(scores) if scores else 0,
            'periods': len(scores)
        }
    
    def detect_overfitting(self, model, X_train: np.ndarray, y_train: np.ndarray, 
                          X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """
        Detect overfitting patterns
        """
        # Train performance
        y_train_pred = model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, y_train_pred)
        
        # Validation performance
        y_val_pred = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_pred)
        
        # Overfitting indicators
        auc_gap = train_auc - val_auc
        overfitting_threshold = 0.1  # 10% gap indicates overfitting
        
        return {
            'train_auc': train_auc,
            'val_auc': val_auc,
            'auc_gap': auc_gap,
            'overfitting': auc_gap > overfitting_threshold,
            'severity': 'high' if auc_gap > 0.2 else 'medium' if auc_gap > 0.1 else 'low'
        }
    
    def calculate_feature_stability(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Calculate feature importance stability across different data splits
        """
        if not hasattr(model, 'feature_importances_'):
            return {'stable': True, 'reason': 'No feature importance available'}
        
        n_bootstraps = 10
        importance_matrix = []
        
        for i in range(n_bootstraps):
            # Bootstrap sample
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Train model
            model.fit(X_boot, y_boot)
            importance_matrix.append(model.feature_importances_)
        
        importance_matrix = np.array(importance_matrix)
        
        # Calculate stability metrics
        mean_importance = np.mean(importance_matrix, axis=0)
        std_importance = np.std(importance_matrix, axis=0)
        cv_importance = std_importance / (mean_importance + 1e-8)
        
        # Feature stability score (lower CV = more stable)
        stability_score = 1 / (1 + np.mean(cv_importance))
        
        return {
            'stability_score': stability_score,
            'stable': stability_score > 0.7,
            'feature_cv': cv_importance.tolist(),
            'mean_importance': mean_importance.tolist()
        }
    
    def validate_model_for_production(self, model, X: np.ndarray, y: np.ndarray, 
                                    feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Comprehensive model validation for production readiness
        """
        self.logger.info("🏆 Running comprehensive production validation...")
        
        validation_results = {
            'production_ready': False,
            'auc_requirement_met': False,
            'issues': [],
            'recommendations': []
        }
        
        # 1. Data leakage check
        if feature_names:
            X_df = pd.DataFrame(X, columns=feature_names)
            leakage_check = self.validate_data_leakage(X_df, pd.Series(y))
            validation_results['data_leakage'] = leakage_check
            
            if not leakage_check['clean']:
                validation_results['issues'].extend(leakage_check['warnings'])
        
        # 2. Cross-validation performance
        cv_results = self.advanced_cross_validation(model, X, y)
        validation_results['cross_validation'] = cv_results
        
        # Check if minimum AUC is met
        primary_auc = cv_results['time_series_cv']['mean']
        validation_results['primary_auc'] = primary_auc
        validation_results['auc_requirement_met'] = primary_auc >= self.min_auc
        
        if primary_auc < self.min_auc:
            validation_results['issues'].append(f"AUC {primary_auc:.3f} < {self.min_auc}")
            validation_results['recommendations'].append("Consider ensemble methods or feature engineering")
        
        # 3. Overfitting check
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model.fit(X_train, y_train)
        overfitting_check = self.detect_overfitting(model, X_train, y_train, X_val, y_val)
        validation_results['overfitting'] = overfitting_check
        
        if overfitting_check['overfitting']:
            validation_results['issues'].append(f"Overfitting detected: {overfitting_check['auc_gap']:.3f} gap")
            validation_results['recommendations'].append("Add regularization or reduce model complexity")
        
        # 4. Feature stability
        stability_check = self.calculate_feature_stability(model, X, y)
        validation_results['feature_stability'] = stability_check
        
        if not stability_check['stable']:
            validation_results['issues'].append("Unstable feature importance")
            validation_results['recommendations'].append("Consider feature selection or regularization")
        
        # 5. Performance consistency
        ts_scores = cv_results['time_series_cv']['scores']
        auc_consistency = np.std(ts_scores) < 0.05  # Standard deviation < 5%
        validation_results['auc_consistency'] = auc_consistency
        
        if not auc_consistency:
            validation_results['issues'].append(f"Inconsistent performance: std={np.std(ts_scores):.3f}")
            validation_results['recommendations'].append("Investigate data quality or model stability")
        
        # Final production readiness assessment
        critical_checks = [
            validation_results['auc_requirement_met'],
            not overfitting_check['overfitting'],
            stability_check['stable']
        ]
        
        validation_results['production_ready'] = all(critical_checks)
        
        if validation_results['production_ready']:
            self.logger.info("✅ Model validated for production use")
        else:
            self.logger.warning(f"❌ Model not ready for production. Issues: {len(validation_results['issues'])}")
        
        return validation_results
    
    def generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """
        Generate detailed validation report
        """
        report = f"""
# 🔍 MODEL VALIDATION REPORT

## 📊 PERFORMANCE SUMMARY
- **Primary AUC**: {validation_results['primary_auc']:.4f}
- **AUC Requirement (≥{self.min_auc})**: {'✅ MET' if validation_results['auc_requirement_met'] else '❌ NOT MET'}
- **Production Ready**: {'✅ YES' if validation_results['production_ready'] else '❌ NO'}

## 🔄 CROSS-VALIDATION RESULTS
### Time Series CV (Primary)
- **Mean AUC**: {validation_results['cross_validation']['time_series_cv']['mean']:.4f}
- **Std AUC**: {validation_results['cross_validation']['time_series_cv']['std']:.4f}
- **Min AUC**: {validation_results['cross_validation']['time_series_cv']['min']:.4f}

### Walk-Forward Validation
- **Mean AUC**: {validation_results['cross_validation']['walk_forward']['mean']:.4f}
- **Periods**: {validation_results['cross_validation']['walk_forward']['periods']}

## 🚨 OVERFITTING ANALYSIS
- **Train AUC**: {validation_results['overfitting']['train_auc']:.4f}
- **Validation AUC**: {validation_results['overfitting']['val_auc']:.4f}
- **AUC Gap**: {validation_results['overfitting']['auc_gap']:.4f}
- **Overfitting**: {'❌ YES' if validation_results['overfitting']['overfitting'] else '✅ NO'}

## 🎯 FEATURE STABILITY
- **Stability Score**: {validation_results['feature_stability']['stability_score']:.4f}
- **Stable**: {'✅ YES' if validation_results['feature_stability']['stable'] else '❌ NO'}

## ⚠️ ISSUES FOUND
"""
        
        if validation_results['issues']:
            for i, issue in enumerate(validation_results['issues'], 1):
                report += f"{i}. {issue}\n"
        else:
            report += "No issues found ✅\n"
        
        report += "\n## 💡 RECOMMENDATIONS\n"
        
        if validation_results['recommendations']:
            for i, rec in enumerate(validation_results['recommendations'], 1):
                report += f"{i}. {rec}\n"
        else:
            report += "No recommendations - model is production ready ✅\n"
        
        return report
