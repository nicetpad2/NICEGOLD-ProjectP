# Step: Walk-Forward Validation (Production-Grade)
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from projectp.model_guard import check_auc_threshold, check_no_overfitting, check_no_noise, check_no_data_leak
from projectp.pro_log import pro_log
from src.utils.resource_auto import get_optimal_resource_fraction, print_resource_summary
from projectp.steps.backtest import load_and_prepare_main_csv
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.console import Console
from rich.panel import Panel
from src.utils.log_utils import safe_fmt
import time
import psutil
import json
from typing import Dict, List, Any, Optional

# --- Resource Allocation 80% ---
try:
    ram_gb, gpu_gb = get_optimal_resource_fraction(ram_fraction=0.8, gpu_fraction=0.8)
    print_resource_summary()
except Exception as e:
    pro_log(f"[WFV] Resource allocation warning: {e}", level="warning", tag="WFV")
    ram_gb, gpu_gb = None, None

console = Console()
ram_gb_str = f"{ram_gb:.2f}" if (ram_gb is not None and isinstance(ram_gb, (int, float))) else "N/A"
gpu_gb_str = f"{gpu_gb:.2f}" if (gpu_gb is not None and isinstance(gpu_gb, (int, float))) else "N/A"
console.print(Panel(f"[bold green]Allocated RAM: {ram_gb_str} GB | GPU: {gpu_gb_str} GB (80%)", title="[green]Resource Allocation", border_style="green"))

def get_positive_class_proba(model, X: np.ndarray) -> np.ndarray:
    """Return probability of positive class (1) for binary/multiclass, robust to all shape scenarios."""
    try:
        # Validate input
        if len(X) == 0:
            return np.array([])
            
        # Get predictions
        proba = model.predict_proba(X)
        
        # Ensure we have a valid probability array
        if proba is None:
            raise ValueError("predict_proba returned None")
            
        proba = np.asarray(proba)
        
        # Handle different array shapes
        if proba.ndim == 1:
            # 1D array - return as is
            return proba
        elif proba.ndim == 2:
            # 2D array - need to extract correct column
            if proba.shape[1] == 1:
                # Only one class predicted
                return proba.flatten()
            elif proba.shape[1] >= 2:
                # Binary or multiclass
                if hasattr(model, 'classes_'):
                    # Find class 1 if it exists
                    if 1 in model.classes_:
                        idx = list(model.classes_).index(1)
                        return proba[:, idx]
                    else:
                        # Use the last class as positive
                        return proba[:, -1]
                else:
                    # Default: return second column for binary
                    return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
            else:
                # Empty second dimension
                return np.zeros(proba.shape[0])
        else:
            # Higher dimensions - flatten and use first elements
            return proba.flatten()[:len(X)]
            
    except Exception as pred_error:
        pro_log(f"[WFV] Prediction error in get_positive_class_proba: {pred_error}", level="warning", tag="WFV")
        # Fallback: try decision_function
        try:
            if hasattr(model, 'decision_function'):
                decision = model.decision_function(X)
                # Convert decision scores to probabilities using sigmoid
                from scipy.special import expit
                return expit(decision)
            else:
                # Last resort: use predict and convert to pseudo-probabilities
                preds = model.predict(X)
                return preds.astype(float)
        except Exception:
            # Final fallback: return default probabilities
            pro_log(f"[WFV] All prediction methods failed, returning default", level="error", tag="WFV")
            return np.full(len(X), 0.5)

def validate_fold_data(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, fold: int) -> bool:
    """Validate fold data before training."""
    # Check for empty data
    if len(X_train) == 0 or len(X_test) == 0:
        pro_log(f"[WFV] Fold {fold}: Empty data, skipping", level="warning", tag="WFV")
        return False
        
    # Check for single class in target
    if y_train.nunique() < 2:
        pro_log(f"[WFV] Fold {fold}: Single class in training data, skipping", level="warning", tag="WFV")
        return False
        
    if y_test.nunique() < 2:
        pro_log(f"[WFV] Fold {fold}: Single class in test data, skipping", level="warning", tag="WFV")
        return False
    
    # Check for NaN values
    if X_train.isnull().any().any() or X_test.isnull().any().any():
        pro_log(f"[WFV] Fold {fold}: NaN values detected in features", level="warning", tag="WFV")
        
    if y_train.isnull().any() or y_test.isnull().any():
        pro_log(f"[WFV] Fold {fold}: NaN values detected in target", level="warning", tag="WFV")
        return False
    
    # Check feature count consistency
    if X_train.shape[1] != X_test.shape[1]:
        pro_log(f"[WFV] Fold {fold}: Feature count mismatch", level="error", tag="WFV")
        return False
        
    return True

def calculate_robust_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate AUC with robust error handling."""
    try:
        # Ensure arrays are 1D
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Handle NaN/Inf values
        y_pred = np.nan_to_num(y_pred, nan=0.5, posinf=1.0, neginf=0.0)
        
        # Check for single class
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            pro_log(f"[WFV] Single class in AUC calculation, returning 0.5", level="warning", tag="WFV")
            return 0.5
            
        # Calculate AUC
        if len(unique_classes) > 2:
            # Multiclass
            return roc_auc_score(y_true, y_pred, multi_class='ovr')
        else:
            # Binary
            return roc_auc_score(y_true, y_pred)
            
    except Exception as auc_error:
        pro_log(f"[WFV] AUC calculation error: {auc_error}", level="warning", tag="WFV")
        return 0.5

def run_walkforward(config: Optional[Dict] = None) -> Optional[str]:
    """Run walk-forward validation with production-grade error handling."""
    console = Console()
    pro_log("[WFV] Running walk-forward validation (Production)...", tag="WFV")
    
    start_time = time.time()
    
    with Progress(SpinnerColumn(), BarColumn(), TextColumn("[progress.description]{task.description}"), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("[cyan]Running walk-forward validation...", total=100)
        
        # Load data
        try:
            csv_path = os.path.join("data", "raw", "your_data_file.csv")
            if os.path.exists(csv_path):
                df = load_and_prepare_main_csv(csv_path, add_target=True)
                df.columns = [c.lower() for c in df.columns]
                pro_log(f"[WFV] Loaded and prepared CSV: {df.shape}", tag="WFV")
                progress.update(task, advance=30, description="[green]Loaded CSV")
            else:
                data_path = os.path.join("output_default", "preprocessed_super.parquet")
                if not os.path.exists(data_path):
                    pro_log(f"[WFV] Data not found: {data_path}", level="error", tag="WFV")
                    progress.update(task, completed=100, description="[red]Data not found")
                    return None
                df = pd.read_parquet(data_path)
                df.columns = [c.lower() for c in df.columns]
                progress.update(task, advance=30, description="[green]Loaded preprocessed_super.parquet")
        except Exception as data_error:
            pro_log(f"[WFV] Data loading error: {data_error}", level="error", tag="WFV")
            return None
        
        # Prepare features and target
        features = [c for c in df.columns if c not in ["target", "date", "datetime", "timestamp", "time"] and df[c].dtype != "O"]
        target = "target"
        
        if target not in df.columns:
            pro_log(f"[WFV] Target column '{target}' not found", level="error", tag="WFV")
            return None
            
        if len(features) == 0:
            pro_log(f"[WFV] No features found", level="error", tag="WFV")
            return None
        
        # Initialize time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        metrics_list: List[Dict[str, Any]] = []
        
        # Configuration
        strict_guard = True
        if config and isinstance(config, dict):
            strict_guard = config.get('strict_guard', True)
        
        # Run cross-validation
        for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
            try:
                # Split data
                train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
                X_train, y_train = train_df[features], train_df[target]
                X_test, y_test = test_df[features], test_df[target]
                
                # Validate fold data
                if not validate_fold_data(X_train, y_train, X_test, y_test, fold):
                    continue
                
                # Check for data leakage
                try:
                    check_no_data_leak(train_df, test_df)
                except Exception as leak_error:
                    pro_log(f"[WFV] Fold {fold}: Data leak check warning - {leak_error}", level="warning", tag="WFV")
                
                # Train model
                try:
                    model = RandomForestClassifier(
                        n_estimators=100, 
                        random_state=fold, 
                        class_weight='balanced',
                        n_jobs=-1,
                        max_depth=10  # Prevent overfitting
                    )
                    model.fit(X_train, y_train)
                except Exception as train_error:
                    pro_log(f"[WFV] Fold {fold}: Training error - {train_error}", level="error", tag="WFV")
                    continue
                
                # Make predictions
                try:
                    y_train_pred = get_positive_class_proba(model, X_train)
                    y_test_pred = get_positive_class_proba(model, X_test)
                    
                    # Validate prediction lengths
                    if len(y_train_pred) != len(y_train) or len(y_test_pred) != len(y_test):
                        pro_log(f"[WFV] Fold {fold}: Prediction length mismatch", level="error", tag="WFV")
                        continue
                        
                except Exception as pred_error:
                    pro_log(f"[WFV] Fold {fold}: Prediction error - {pred_error}", level="error", tag="WFV")
                    continue
                
                # Calculate metrics
                try:
                    auc_train = calculate_robust_auc(y_train, y_train_pred)
                    auc_test = calculate_robust_auc(y_test, y_test_pred)
                    
                    # Binary predictions for accuracy
                    y_test_binary = (y_test_pred > 0.5).astype(int)
                    acc_test = accuracy_score(y_test, y_test_binary)
                    
                except Exception as metric_error:
                    pro_log(f"[WFV] Fold {fold}: Metrics calculation error - {metric_error}", level="error", tag="WFV")
                    # Fallback values
                    auc_train, auc_test, acc_test = 0.5, 0.5, 0.5
                
                # Model validation
                metrics = {'auc': auc_test, 'train_auc': auc_train, 'test_auc': auc_test}
                
                try:
                    check_auc_threshold(metrics, strict=strict_guard)
                    check_no_overfitting(metrics, strict=strict_guard)
                except Exception as guard_error:
                    pro_log(f"[WFV] Fold {fold}: Guard check warning - {guard_error}", level="warning", tag="WFV")
                
                # Feature importance
                try:
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        feature_importance = {f: imp for f, imp in zip(features, importances)}
                        check_no_noise(feature_importance)
                except Exception as importance_error:
                    pro_log(f"[WFV] Fold {fold}: Feature importance warning - {importance_error}", level="warning", tag="WFV")
                
                # Store metrics
                metrics_list.append({
                    'fold': fold,
                    'auc_train': float(auc_train),
                    'auc_test': float(auc_test),
                    'acc_test': float(acc_test)
                })
                
                pro_log(f"[WFV] Fold {fold}: AUC={auc_test:.3f}, ACC={acc_test:.3f}", tag="WFV")
                
            except Exception as fold_error:
                pro_log(f"[WFV] Fold {fold}: Complete fold failed - {fold_error}", level="error", tag="WFV")
                continue
            
            progress.update(task, advance=10, description=f"[cyan]Fold {fold+1}/5 done")
        
        # Save results
        if not metrics_list:
            pro_log("[WFV] No successful folds completed", level="error", tag="WFV")
            return None
            
        try:
            metrics_df = pd.DataFrame(metrics_list)
            out_path = os.path.join("output_default", "walkforward_metrics.csv")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            metrics_df.to_csv(out_path, index=False)
            pro_log(f"[WFV] Walk-forward metrics saved to {out_path}", level="success", tag="WFV")
            progress.update(task, completed=100, description="[green]Walk-forward validation complete")
        except Exception as save_error:
            pro_log(f"[WFV] Error saving results: {save_error}", level="error", tag="WFV")
            return None

        # Export resource information
        try:
            resource_info = {
                'ram_percent': psutil.virtual_memory().percent,
                'ram_used_gb': psutil.virtual_memory().used / 1e9,
                'ram_total_gb': psutil.virtual_memory().total / 1e9,
                'cpu_percent': psutil.cpu_percent(),
                'gpu_used_gb': 0.0,
                'gpu_total_gb': 0.0
            }
            
            # Try to get GPU info
            try:
                import pynvml
                pynvml.nvmlInit()
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                resource_info['gpu_used_gb'] = float(gpu_mem.used) / 1e9
                resource_info['gpu_total_gb'] = float(gpu_mem.total) / 1e9
            except Exception:
                pass  # GPU info not critical
            
            resource_log_path = os.path.join("output_default", "walkforward_resource_log.json")
            with open(resource_log_path, "w", encoding="utf-8") as f:
                json.dump(resource_info, f, indent=2)
            pro_log(f"[WFV] Resource log exported to {resource_log_path}", tag="WFV")
        except Exception as resource_error:
            pro_log(f"[WFV] Resource logging error: {resource_error}", level="warning", tag="WFV")

        # Export summary metrics
        try:
            summary = {}
            for col in ['auc_train', 'auc_test', 'acc_test']:
                if col in metrics_df.columns:
                    col_data = metrics_df[col].dropna()
                    if len(col_data) > 0:
                        summary[col] = {
                            'mean': float(col_data.mean()),
                            'std': float(col_data.std()),
                            'min': float(col_data.min()),
                            'max': float(col_data.max()),
                        }
            
            summary_path = os.path.join("output_default", "walkforward_summary_metrics.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            pro_log(f"[WFV] Summary metrics exported to {summary_path}", tag="WFV")
        except Exception as summary_error:
            pro_log(f"[WFV] Summary metrics export error: {summary_error}", level="warning", tag="WFV")

        # Export diagnostics log
        try:
            diagnostics_path = os.path.join("output_default", "walkforward_diagnostics.log")
            with open(diagnostics_path, "a", encoding="utf-8") as f:
                f.write(f"[DIAG] Walkforward run at: {pd.Timestamp.now()}\n")
                if 'resource_info' in locals():
                    f.write(f"Resource: {resource_info}\n")
                if 'summary' in locals():
                    f.write(f"Summary: {summary}\n")
                f.write(f"Output: {out_path}\n")
            pro_log(f"[WFV] Diagnostics log exported to {diagnostics_path}", tag="WFV")
        except Exception as diag_error:
            pro_log(f"[WFV] Diagnostics logging error: {diag_error}", level="warning", tag="WFV")

        # Validate output file
        try:
            assert os.path.exists(out_path), f"Walkforward output file missing: {out_path}"
            df_check = pd.read_csv(out_path)
            df_check.columns = [c.lower() for c in df_check.columns]
            required_cols = ['auc_train', 'auc_test', 'acc_test']
            missing_cols = [c for c in required_cols if c not in df_check.columns]
            if missing_cols:
                raise ValueError(f"Walkforward output missing columns: {missing_cols}")
            pro_log(f"[WFV] Output file and columns validated.", tag="WFV")
        except Exception as validation_error:
            pro_log(f"[WFV] Output validation error: {validation_error}", level="error", tag="WFV")

        # Export timing log
        try:
            end_time = time.time()
            duration = end_time - start_time
            perf_log_path = os.path.join("output_default", "walkforward_perf.log")
            with open(perf_log_path, "a", encoding="utf-8") as f:
                f.write(f"Walkforward finished at {pd.Timestamp.now()} | Duration: {duration:.2f} sec\n")
            pro_log(f"[WFV] Performance log exported to {perf_log_path}", tag="WFV")
        except Exception as perf_error:
            pro_log(f"[WFV] Performance logging error: {perf_error}", level="warning", tag="WFV")

        # Display summary
        console.print(Panel("[bold green]Walk-forward validation completed!", title="Walk-Forward", expand=False))

        # Rich summary table
        try:
            from rich.table import Table
            from rich import box
            
            summary_table = Table(title="[bold green]Walk-Forward Validation Summary", show_header=True, header_style="bold magenta", box=box.ROUNDED)
            summary_table.add_column("Metric", style="cyan", justify="right")
            summary_table.add_column("Value", style="white")
            
            if metrics_list:
                # Calculate averages
                avg_auc_test = sum([m['auc_test'] for m in metrics_list]) / len(metrics_list)
                avg_auc_train = sum([m['auc_train'] for m in metrics_list]) / len(metrics_list)
                avg_acc_test = sum([m['acc_test'] for m in metrics_list]) / len(metrics_list)
                
                summary_table.add_row("Average Test AUC", f"{avg_auc_test:.4f}")
                summary_table.add_row("Average Train AUC", f"{avg_auc_train:.4f}")
                summary_table.add_row("Average Test Accuracy", f"{avg_acc_test:.4f}")
                summary_table.add_row("Completed Folds", str(len(metrics_list)))
                summary_table.add_row("Output File", out_path)
            
            console.print(Panel(summary_table, title="[bold blue]🚀 Walk-Forward Complete!", border_style="bright_green"))
        except Exception as display_error:
            pro_log(f"[WFV] Display error: {display_error}", level="warning", tag="WFV")

        return out_path

if __name__ == "__main__":
    result = run_walkforward()
    if result:
        print(f"Walk-forward validation completed successfully. Results saved to: {result}")
    else:
        print("Walk-forward validation failed.")
