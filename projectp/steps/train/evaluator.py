"""
Model Evaluation Module
======================
Handles model evaluation, metrics calculation, and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple, List
from projectp.pro_log import pro_log
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    roc_curve, precision_recall_curve, confusion_matrix, classification_report
)
from rich.console import Console
from rich.table import Table

console = Console()

class ModelEvaluator:
    """Model evaluation and metrics calculation"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics = {}
        self.plots_saved = []
        
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                      X_train: Optional[pd.DataFrame] = None, y_train: Optional[pd.Series] = None) -> Dict[str, float]:
        """Comprehensive model evaluation"""
        try:
            # Test predictions
            y_test_pred_proba = model.predict_proba(X_test)[:, 1]
            y_test_pred = model.predict(X_test)
            
            # Test metrics
            test_metrics = {
                'test_auc': roc_auc_score(y_test, y_test_pred_proba),
                'test_accuracy': accuracy_score(y_test, y_test_pred),
                'test_precision': precision_score(y_test, y_test_pred, zero_division=0),
                'test_recall': recall_score(y_test, y_test_pred, zero_division=0)
            }
            
            # Train metrics (if provided)
            if X_train is not None and y_train is not None:
                y_train_pred_proba = model.predict_proba(X_train)[:, 1]
                y_train_pred = model.predict(X_train)
                
                train_metrics = {
                    'train_auc': roc_auc_score(y_train, y_train_pred_proba),
                    'train_accuracy': accuracy_score(y_train, y_train_pred),
                    'train_precision': precision_score(y_train, y_train_pred, zero_division=0),
                    'train_recall': recall_score(y_train, y_train_pred, zero_division=0)
                }
                test_metrics.update(train_metrics)
            
            # Calculate F1 score
            if test_metrics['test_precision'] + test_metrics['test_recall'] > 0:
                test_metrics['test_f1'] = 2 * (test_metrics['test_precision'] * test_metrics['test_recall']) / \
                                         (test_metrics['test_precision'] + test_metrics['test_recall'])
            else:
                test_metrics['test_f1'] = 0.0
            
            self.metrics = test_metrics
            self._log_metrics()
            
            return test_metrics
            
        except Exception as e:
            pro_log(f"[ModelEvaluator] Evaluation failed: {e}", level="error", tag="Eval")
            return {'test_auc': 0.0, 'test_accuracy': 0.0}
    
    def _log_metrics(self) -> None:
        """Log metrics in a formatted table"""
        table = Table(title="[bold blue]Model Performance Metrics", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for metric, value in self.metrics.items():
            table.add_row(metric, f"{value:.4f}")
        
        console.print(table)
        
        # Log to pro_log as well
        for metric, value in self.metrics.items():
            pro_log(f"[ModelEvaluator] {metric}: {value:.4f}", tag="Eval")
    
    def generate_classification_report(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                                     output_dir: str = "output_default") -> str:
        """Generate detailed classification report"""
        try:
            import os
            import json
            
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Save to file
            os.makedirs(output_dir, exist_ok=True)
            report_path = os.path.join(output_dir, "classification_report.json")
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            pro_log(f"[ModelEvaluator] Classification report saved to {report_path}", tag="Eval")
            return report_path
            
        except Exception as e:
            pro_log(f"[ModelEvaluator] Classification report generation failed: {e}", level="warn", tag="Eval")
            return ""
    
    def plot_roc_curve(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                      output_dir: str = "output_default", title: str = "ROC Curve") -> str:
        """Plot and save ROC curve"""
        try:
            import os
            
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(title)
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, "roc_curve.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.plots_saved.append(plot_path)
            pro_log(f"[ModelEvaluator] ROC curve saved to {plot_path}", tag="Eval")
            return plot_path
            
        except Exception as e:
            pro_log(f"[ModelEvaluator] ROC curve plotting failed: {e}", level="warn", tag="Eval")
            return ""
    
    def plot_precision_recall_curve(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                                   output_dir: str = "output_default", title: str = "Precision-Recall Curve") -> str:
        """Plot and save Precision-Recall curve"""
        try:
            import os
            
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, linewidth=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(title)
            plt.grid(True, alpha=0.3)
            
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, "precision_recall_curve.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.plots_saved.append(plot_path)
            pro_log(f"[ModelEvaluator] PR curve saved to {plot_path}", tag="Eval")
            return plot_path
            
        except Exception as e:
            pro_log(f"[ModelEvaluator] PR curve plotting failed: {e}", level="warn", tag="Eval")
            return ""
    
    def plot_confusion_matrix(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                            output_dir: str = "output_default", title: str = "Confusion Matrix") -> str:
        """Plot and save confusion matrix"""
        try:
            import os
            import seaborn as sns
            
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Predicted 0', 'Predicted 1'],
                       yticklabels=['Actual 0', 'Actual 1'])
            plt.title(title)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, "confusion_matrix.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.plots_saved.append(plot_path)
            pro_log(f"[ModelEvaluator] Confusion matrix saved to {plot_path}", tag="Eval")
            return plot_path
            
        except Exception as e:
            pro_log(f"[ModelEvaluator] Confusion matrix plotting failed: {e}", level="warn", tag="Eval")
            return ""
    
    def plot_feature_importance(self, model: Any, feature_names: List[str], 
                              output_dir: str = "output_default", top_n: int = 20) -> str:
        """Plot and save feature importance"""
        try:
            import os
            
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'get_feature_importance'):
                importances = model.get_feature_importance()
            else:
                pro_log("[ModelEvaluator] Model doesn't support feature importance", level="warn", tag="Eval")
                return ""
            
            # Create dataframe and sort
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Plot top N features
            top_features = importance_df.head(top_n)
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance')
            plt.title(f'Top {top_n} Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, "feature_importance.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save importance data
            importance_path = os.path.join(output_dir, "feature_importance.csv")
            importance_df.to_csv(importance_path, index=False)
            
            self.plots_saved.append(plot_path)
            pro_log(f"[ModelEvaluator] Feature importance saved to {plot_path}", tag="Eval")
            return plot_path
            
        except Exception as e:
            pro_log(f"[ModelEvaluator] Feature importance plotting failed: {e}", level="warn", tag="Eval")
            return ""
    
    def optimize_threshold(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, Dict[str, float]]:
        """Find optimal prediction threshold"""
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            
            # Youden's J statistic (max(tpr - fpr))
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            # Calculate metrics at optimal threshold
            y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
            
            optimal_metrics = {
                'threshold': optimal_threshold,
                'accuracy': accuracy_score(y_test, y_pred_optimal),
                'precision': precision_score(y_test, y_pred_optimal, zero_division=0),
                'recall': recall_score(y_test, y_pred_optimal, zero_division=0),
                'tpr': tpr[optimal_idx],
                'fpr': fpr[optimal_idx]
            }
            
            pro_log(f"[ModelEvaluator] Optimal threshold: {optimal_threshold:.3f}", tag="Eval")
            return optimal_threshold, optimal_metrics
            
        except Exception as e:
            pro_log(f"[ModelEvaluator] Threshold optimization failed: {e}", level="warn", tag="Eval")
            return 0.5, {}
    
    def check_model_quality(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """Check model quality against thresholds"""
        quality_checks = {
            'auc_acceptable': metrics.get('test_auc', 0) >= 0.7,
            'auc_good': metrics.get('test_auc', 0) >= 0.8,
            'accuracy_acceptable': metrics.get('test_accuracy', 0) >= 0.65,
            'no_overfitting': True,  # Will be checked if train metrics available
            'f1_acceptable': metrics.get('test_f1', 0) >= 0.6
        }
        
        # Check overfitting if train metrics are available
        if 'train_auc' in metrics and 'test_auc' in metrics:
            auc_diff = metrics['train_auc'] - metrics['test_auc']
            quality_checks['no_overfitting'] = auc_diff <= 0.1
        
        # Log quality assessment
        for check, passed in quality_checks.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            pro_log(f"[ModelEvaluator] {check}: {status}", tag="Eval")
        
        return quality_checks
    
    def generate_evaluation_summary(self, output_dir: str = "output_default") -> str:
        """Generate comprehensive evaluation summary"""
        try:
            import os
            import json
            
            summary = {
                'metrics': self.metrics,
                'plots_generated': self.plots_saved,
                'quality_checks': self.check_model_quality(self.metrics),
                'evaluation_timestamp': pd.Timestamp.now().isoformat()
            }
            
            os.makedirs(output_dir, exist_ok=True)
            summary_path = os.path.join(output_dir, "evaluation_summary.json")
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            pro_log(f"[ModelEvaluator] Evaluation summary saved to {summary_path}", tag="Eval")
            return summary_path
            
        except Exception as e:
            pro_log(f"[ModelEvaluator] Summary generation failed: {e}", level="warn", tag="Eval")
            return ""
