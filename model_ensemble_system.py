#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 ADVANCED MODEL ENSEMBLE SYSTEM
Intelligent model stacking, adaptive weighting, and ensemble optimization
for NICEGOLD ProjectP v2.1

Author: NICEGOLD Enterprise
Date: June 25, 2025
"""

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import joblib
import lightgbm as lgb
import numpy as np
import xgboost as xgb

# ML imports
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

warnings.filterwarnings('ignore')

# Rich imports with fallback
RICH_AVAILABLE = False
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import track
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    pass


class ModelEnsemble:
    """🚀 Advanced Model Ensemble System with Adaptive Weighting"""
    
    def __init__(self, console_output: bool = True):
        self.console = Console() if RICH_AVAILABLE else None
        self.console_output = console_output
        self.base_models = {}
        self.ensemble_model = None
        self.model_performances = {}
        self.adaptive_weights = {}
        self.ensemble_history = []
        
    def initialize_base_models(self) -> Dict[str, Any]:
        """
        🔧 สร้างและกำหนดค่า base models สำหรับ ensemble
        """
        if self.console_output and RICH_AVAILABLE:
            self._show_initialization_header()
        
        # Define base models with optimized parameters
        self.base_models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='auc'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        }
        
        if self.console_output:
            if RICH_AVAILABLE:
                self._display_base_models()
            else:
                print("✅ Initialized base models:")
                for name in self.base_models.keys():
                    print(f"  • {name}")
        
        return self.base_models
    
    def stack_models(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray = None, y_test: np.ndarray = None,
                    cv_folds: int = 5) -> Dict[str, Any]:
        """
        🎯 สร้าง Stacking Ensemble
        """
        if self.console_output and RICH_AVAILABLE:
            self._show_stacking_header()
        
        # Initialize base models if not done
        if not self.base_models:
            self.initialize_base_models()
        
        # Create list of (name, model) tuples for StackingClassifier
        estimators = [(name, model) for name, model in self.base_models.items()]
        
        # Create meta-learner (Level-2 model)
        meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        
        # Create stacking classifier
        self.ensemble_model = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=TimeSeriesSplit(n_splits=cv_folds),
            n_jobs=-1
        )
        
        # Train the ensemble
        if self.console_output and RICH_AVAILABLE:
            with self.console.status("[bold green]Training stacking ensemble..."):
                self.ensemble_model.fit(X_train, y_train)
        else:
            if self.console_output:
                print("🔄 Training stacking ensemble...")
            self.ensemble_model.fit(X_train, y_train)
        
        # Evaluate individual models and ensemble
        results = self._evaluate_models(X_train, y_train, X_test, y_test)
        
        if self.console_output:
            self._display_stacking_results(results)
        
        return results
    
    def adaptive_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray = None, y_test: np.ndarray = None,
                         adaptation_window: int = 100) -> Dict[str, Any]:
        """
        🧠 สร้าง Adaptive Ensemble ที่ปรับน้ำหนักตามประสิทธิภาพ
        """
        if self.console_output and RICH_AVAILABLE:
            self._show_adaptive_header()
        
        # Initialize base models if not done
        if not self.base_models:
            self.initialize_base_models()
        
        # Train individual models
        trained_models = {}
        model_scores = {}
        
        models_to_process = list(self.base_models.items())
        if RICH_AVAILABLE and self.console_output:
            models_to_process = track(models_to_process, 
                                    description="Training base models...")
        
        for name, model in models_to_process:
            try:
                model.fit(X_train, y_train)
                trained_models[name] = model
                
                # Calculate performance score
                if X_test is not None and y_test is not None:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    score = roc_auc_score(y_test, y_pred_proba)
                else:
                    # Use cross-validation score
                    cv_scores = cross_val_score(
                        model, X_train, y_train, cv=3, scoring='roc_auc'
                    )
                    score = np.mean(cv_scores)
                
                model_scores[name] = score
                
                if self.console_output and not RICH_AVAILABLE:
                    print(f"✅ {name}: AUC = {score:.3f}")
                    
            except Exception as e:
                if self.console_output:
                    print(f"❌ Error training {name}: {str(e)}")
                continue
        
        # Calculate adaptive weights based on performance
        self.adaptive_weights = self._calculate_adaptive_weights(model_scores)
        
        # Store results
        results = {
            'trained_models': trained_models,
            'model_scores': model_scores,
            'adaptive_weights': self.adaptive_weights,
            'ensemble_score': self._calculate_ensemble_score(
                X_test, y_test, trained_models) if X_test is not None else None
        }
        
        if self.console_output:
            self._display_adaptive_results(results)
        
        return results
    
    def predict_ensemble(self, X: np.ndarray, 
                        method: str = 'adaptive') -> np.ndarray:
        """
        🎯 ทำนายผลด้วย ensemble method
        """
        if method == 'stacking' and self.ensemble_model is not None:
            return self.ensemble_model.predict_proba(X)[:, 1]
        
        elif method == 'adaptive' and self.adaptive_weights:
            predictions = []
            for name, model in self.base_models.items():
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)[:, 1]
                    weight = self.adaptive_weights.get(name, 0)
                    predictions.append(pred * weight)
            
            if predictions:
                return np.sum(predictions, axis=0)
        
        # Fallback to simple voting
        return self._simple_voting_predict(X)
    
    def _calculate_adaptive_weights(self, model_scores: Dict[str, float],
                                   min_weight: float = 0.1) -> Dict[str, float]:
        """คำนวณน้ำหนักแบบ adaptive ตามประสิทธิภาพ"""
        if not model_scores:
            return {}
        
        # Convert scores to weights using softmax
        scores_array = np.array(list(model_scores.values()))
        # Apply temperature scaling for sharper/softer distribution
        temperature = 2.0
        exp_scores = np.exp(scores_array / temperature)
        weights = exp_scores / np.sum(exp_scores)
        
        # Ensure minimum weight for each model
        weights = np.maximum(weights, min_weight)
        weights = weights / np.sum(weights)  # Renormalize
        
        return dict(zip(model_scores.keys(), weights))
    
    def _calculate_ensemble_score(self, X_test: np.ndarray, y_test: np.ndarray,
                                 trained_models: Dict[str, Any]) -> float:
        """คำนวณคะแนนของ ensemble"""
        if X_test is None or y_test is None:
            return 0.0
        
        try:
            ensemble_pred = self.predict_ensemble(X_test, method='adaptive')
            return roc_auc_score(y_test, ensemble_pred)
        except Exception:
            return 0.0
    
    def _evaluate_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray = None,
                        y_test: np.ndarray = None) -> Dict[str, Any]:
        """ประเมินผลประสิทธิภาพของ models"""
        results = {
            'individual_scores': {},
            'ensemble_score': None,
            'cross_val_scores': {}
        }
        
        # Evaluate base models
        for name, model in self.base_models.items():
            try:
                # Cross-validation score
                cv_scores = cross_val_score(
                    model, X_train, y_train, cv=3, scoring='roc_auc'
                )
                results['cross_val_scores'][name] = {
                    'mean': np.mean(cv_scores),
                    'std': np.std(cv_scores)
                }
                
                # Test score if available
                if X_test is not None and y_test is not None:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    test_score = roc_auc_score(y_test, y_pred_proba)
                    results['individual_scores'][name] = test_score
                    
            except Exception as e:
                if self.console_output:
                    print(f"⚠️ Error evaluating {name}: {str(e)}")
        
        # Evaluate ensemble
        if X_test is not None and y_test is not None:
            try:
                ensemble_pred = self.ensemble_model.predict_proba(X_test)[:, 1]
                results['ensemble_score'] = roc_auc_score(y_test, ensemble_pred)
            except Exception:
                pass
        
        return results
    
    def _simple_voting_predict(self, X: np.ndarray) -> np.ndarray:
        """Simple voting prediction fallback"""
        predictions = []
        for model in self.base_models.values():
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
                predictions.append(pred)
        
        if predictions:
            return np.mean(predictions, axis=0)
        else:
            return np.zeros(len(X))
    
    def save_ensemble(self, filepath: str) -> bool:
        """บันทึก ensemble model"""
        try:
            ensemble_data = {
                'base_models': self.base_models,
                'ensemble_model': self.ensemble_model,
                'adaptive_weights': self.adaptive_weights,
                'model_performances': self.model_performances,
                'timestamp': datetime.now().isoformat()
            }
            joblib.dump(ensemble_data, filepath)
            if self.console_output:
                print(f"✅ Ensemble saved to: {filepath}")
            return True
        except Exception as e:
            if self.console_output:
                print(f"❌ Error saving ensemble: {str(e)}")
            return False
    
    def load_ensemble(self, filepath: str) -> bool:
        """โหลด ensemble model"""
        try:
            ensemble_data = joblib.load(filepath)
            self.base_models = ensemble_data.get('base_models', {})
            self.ensemble_model = ensemble_data.get('ensemble_model')
            self.adaptive_weights = ensemble_data.get('adaptive_weights', {})
            self.model_performances = ensemble_data.get('model_performances', {})
            
            if self.console_output:
                print(f"✅ Ensemble loaded from: {filepath}")
            return True
        except Exception as e:
            if self.console_output:
                print(f"❌ Error loading ensemble: {str(e)}")
            return False
    
    # Display methods for Rich UI
    def _show_initialization_header(self):
        """แสดงหัวข้อการเริ่มต้น"""
        if RICH_AVAILABLE:
            header = Panel(
                "[bold cyan]🤖 MODEL ENSEMBLE INITIALIZATION[/bold cyan]\n"
                "[yellow]Setting up base models for ensemble learning[/yellow]",
                border_style="cyan"
            )
            self.console.print(header)
    
    def _show_stacking_header(self):
        """แสดงหัวข้อ stacking"""
        if RICH_AVAILABLE:
            header = Panel(
                "[bold green]🎯 STACKING ENSEMBLE[/bold green]\n"
                "[yellow]Training multi-level ensemble with meta-learner[/yellow]",
                border_style="green"
            )
            self.console.print(header)
    
    def _show_adaptive_header(self):
        """แสดงหัวข้อ adaptive ensemble"""
        if RICH_AVAILABLE:
            header = Panel(
                "[bold magenta]🧠 ADAPTIVE ENSEMBLE[/bold magenta]\n"
                "[yellow]Dynamic model weighting based on performance[/yellow]",
                border_style="magenta"
            )
            self.console.print(header)
    
    def _display_base_models(self):
        """แสดงรายการ base models"""
        if RICH_AVAILABLE:
            table = Table(title="🔧 Base Models Configuration", border_style="blue")
            table.add_column("Model", style="cyan")
            table.add_column("Type", style="yellow")
            table.add_column("Parameters", style="green")
            
            for name, model in self.base_models.items():
                model_type = type(model).__name__
                # Get key parameters
                params = []
                if hasattr(model, 'n_estimators'):
                    params.append(f"n_estimators={model.n_estimators}")
                if hasattr(model, 'max_depth'):
                    params.append(f"max_depth={model.max_depth}")
                param_str = ", ".join(params[:2])  # Show first 2 params
                
                table.add_row(name, model_type, param_str)
            
            self.console.print(table)
    
    def _display_stacking_results(self, results: Dict[str, Any]):
        """แสดงผลลัพธ์ stacking"""
        if RICH_AVAILABLE:
            table = Table(title="📊 Stacking Ensemble Results", border_style="green")
            table.add_column("Model", style="cyan")
            table.add_column("CV Score", style="yellow")
            table.add_column("Test Score", style="green")
            
            # Add individual model scores
            for name, cv_data in results.get('cross_val_scores', {}).items():
                cv_score = f"{cv_data['mean']:.3f} ± {cv_data['std']:.3f}"
                test_score = f"{results['individual_scores'].get(name, 0):.3f}"
                table.add_row(name, cv_score, test_score)
            
            # Add ensemble score
            if results.get('ensemble_score'):
                table.add_row(
                    "[bold]Stacking Ensemble[/bold]",
                    "N/A",
                    f"[bold green]{results['ensemble_score']:.3f}[/bold green]"
                )
            
            self.console.print(table)
    
    def _display_adaptive_results(self, results: Dict[str, Any]):
        """แสดงผลลัพธ์ adaptive ensemble"""
        if RICH_AVAILABLE:
            table = Table(title="🧠 Adaptive Ensemble Results", border_style="magenta")
            table.add_column("Model", style="cyan")
            table.add_column("Performance", style="yellow")
            table.add_column("Weight", style="green")
            
            model_scores = results.get('model_scores', {})
            adaptive_weights = results.get('adaptive_weights', {})
            
            for name in model_scores.keys():
                score = f"{model_scores[name]:.3f}"
                weight = f"{adaptive_weights.get(name, 0):.3f}"
                table.add_row(name, score, weight)
            
            # Add ensemble score
            if results.get('ensemble_score'):
                table.add_row(
                    "[bold]Adaptive Ensemble[/bold]",
                    f"[bold green]{results['ensemble_score']:.3f}[/bold green]",
                    "1.000"
                )
            
            self.console.print(table)


if __name__ == "__main__":
    # Demo/Test the Model Ensemble System
    print("🚀 NICEGOLD ProjectP - Model Ensemble Demo")
    
    # Create sample data for testing
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    # Create a challenging binary classification problem
    y = (X[:, 0] + X[:, 1] * 0.5 + X[:, 2] * 0.3 + 
         np.random.randn(n_samples) * 0.1 > 0).astype(int)
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Test the ensemble system
    ensemble = ModelEnsemble()
    
    # 1. Test stacking ensemble
    print("\n1. Testing Stacking Ensemble...")
    stacking_results = ensemble.stack_models(X_train, y_train, X_test, y_test)
    
    # 2. Test adaptive ensemble
    print("\n2. Testing Adaptive Ensemble...")
    adaptive_results = ensemble.adaptive_ensemble(X_train, y_train, X_test, y_test)
    
    # 3. Test predictions
    print("\n3. Testing Predictions...")
    stacking_pred = ensemble.predict_ensemble(X_test, method='stacking')
    adaptive_pred = ensemble.predict_ensemble(X_test, method='adaptive')
    
    # 4. Test saving/loading
    print("\n4. Testing Save/Load...")
    save_path = "test_ensemble.joblib"
    ensemble.save_ensemble(save_path)
    
    new_ensemble = ModelEnsemble()
    new_ensemble.load_ensemble(save_path)
    
    print(f"\n✅ Demo completed successfully!")
    print(f"🎯 Stacking AUC: {stacking_results.get('ensemble_score', 'N/A')}")
    print(f"🧠 Adaptive AUC: {adaptive_results.get('ensemble_score', 'N/A')}")
    
    # Cleanup
    if Path(save_path).exists():
        Path(save_path).unlink()
