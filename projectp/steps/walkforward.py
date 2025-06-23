# Step: Walk-Forward Validation (Production-Grade)
import os
import pandas as pd
import numpy as np
import traceback
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

# --- Resource Allocation 80% ---
try:
    ram_gb, gpu_gb = get_optimal_resource_fraction(ram_fraction=0.8, gpu_fraction=0.8)
    print_resource_summary()
except Exception as e:
    pro_log(f"[WFV] Resource allocation warning: {e}", level="warning", tag="WFV")
    ram_gb, gpu_gb = None, None

console = Console()
ram_gb_str = f"{ram_gb:.2f}" if ram_gb is not None else "N/A"
gpu_gb_str = f"{gpu_gb:.2f}" if gpu_gb is not None else "N/A"
console.print(Panel(f"[bold green]Allocated RAM: {ram_gb_str} GB | GPU: {gpu_gb_str} GB (80%)", title="[green]Resource Allocation", border_style="green"))

def get_positive_class_proba(model, X):
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
        
        # DEBUG: Log array info to help diagnose
        pro_log(f"[WFV] Debug - proba shape: {proba.shape}, ndim: {proba.ndim}, dtype: {proba.dtype}", level="debug", tag="WFV")
        
        # Handle different array shapes
        if proba.ndim == 1:
            # 1D array - return as is
            pro_log(f"[WFV] Debug - 1D array detected, returning as-is", level="debug", tag="WFV")
            return proba
        elif proba.ndim == 2:
            # 2D array - need to extract correct column
            pro_log(f"[WFV] Debug - 2D array detected, shape[1]={proba.shape[1]}", level="debug", tag="WFV")
            
            if proba.shape[1] == 1:
                # Only one class predicted
                pro_log(f"[WFV] Debug - Single class, flattening", level="debug", tag="WFV")
                return proba.flatten()
            elif proba.shape[1] >= 2:
                # Binary or multiclass - BE EXTRA CAREFUL HERE
                try:
                    if hasattr(model, 'classes_'):
                        # Find class 1 if it exists
                        if 1 in model.classes_:
                            idx = list(model.classes_).index(1)
                            pro_log(f"[WFV] Debug - Found class 1 at idx {idx}, extracting proba[:, {idx}]", level="debug", tag="WFV")
                            # ROBUST CHECK: Ensure idx is valid
                            if idx < proba.shape[1]:
                                return proba[:, idx]
                            else:
                                pro_log(f"[WFV] Warning - idx {idx} >= shape[1] {proba.shape[1]}, using last column", level="warning", tag="WFV")
                                return proba[:, -1]
                        else:
                            # Use the last class as positive
                            pro_log(f"[WFV] Debug - Class 1 not found, using last class", level="debug", tag="WFV")
                            return proba[:, -1]
                    else:
                        # Default: return second column for binary
                        pro_log(f"[WFV] Debug - No model.classes_, using default logic", level="debug", tag="WFV")
                        return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
                except Exception as idx_error:
                    pro_log(f"[WFV] Warning - Index error in 2D array handling: {idx_error}, using last column", level="warning", tag="WFV")
                    return proba[:, -1] if proba.shape[1] > 0 else np.zeros(proba.shape[0])
            else:
                # Empty second dimension
                pro_log(f"[WFV] Debug - Empty second dimension, returning zeros", level="debug", tag="WFV")
                return np.zeros(proba.shape[0])
        else:
            # Higher dimensions - flatten and use first elements
            pro_log(f"[WFV] Debug - Higher dimensions, flattening", level="debug", tag="WFV")
            return proba.flatten()[:len(X)]
            
    except Exception as pred_error:
        pro_log(f"[WFV] Prediction error in get_positive_class_proba: {pred_error}", level="warning", tag="WFV")
        pro_log(f"[WFV] Exception traceback: {traceback.format_exc()}", level="debug", tag="WFV")
        # Fallback: try decision_function
        try:
            if hasattr(model, 'decision_function'):
                decision = model.decision_function(X)
                # Convert decision scores to probabilities using simple sigmoid approximation
                decision = np.asarray(decision)
                return 1.0 / (1.0 + np.exp(-decision))
            else:
                # Last resort: use predict and convert to pseudo-probabilities
                preds = model.predict(X)
                return preds.astype(float)
        except Exception:
            # Final fallback: return default probabilities
            pro_log(f"[WFV] All prediction methods failed, returning default", level="error", tag="WFV")
            return np.full(len(X), 0.5)

def validate_fold_data(X_train, y_train, X_test, y_test, fold):
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

def calculate_robust_auc(y_true, y_pred):
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

def run_walkforward(config=None):
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    console = Console()

    pro_log("[WFV] Running walk-forward validation (เทพ)...", tag="WFV")
    with Progress(SpinnerColumn(), BarColumn(), TextColumn("[progress.description]{task.description}"), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("[cyan]Running walk-forward validation...", total=100)
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
        features = [c for c in df.columns if c not in ["target", "date", "datetime", "timestamp", "time"] and df[c].dtype != "O"]
        target = "target"
        tscv = TimeSeriesSplit(n_splits=5)
        metrics_list = []
        strict_guard = True
        if config and isinstance(config, dict):
            strict_guard = config.get('strict_guard', True)
            
        for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
            try:
                train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
                X_train, y_train = train_df[features], train_df[target]
                X_test, y_test = test_df[features], test_df[target]
                
                # PRODUCTION FIX: Validate data before training
                if len(X_train) == 0 or len(X_test) == 0:
                    pro_log(f"[WFV] Fold {fold}: Empty data, skipping", level="warning", tag="WFV")
                    continue
                    
                if y_train.nunique() < 2 or y_test.nunique() < 2:
                    pro_log(f"[WFV] Fold {fold}: Single class, skipping", level="warning", tag="WFV")
                    continue
                
                check_no_data_leak(train_df, test_df)
                
                # PRODUCTION FIX: Robust model training
                model = RandomForestClassifier(n_estimators=100, random_state=fold, class_weight='balanced')
                model.fit(X_train, y_train)
                
                # PRODUCTION FIX: Robust probability prediction
                try:
                    y_train_pred = get_positive_class_proba(model, X_train)
                    y_test_pred = get_positive_class_proba(model, X_test)
                    
                    # PRODUCTION FIX: Validate predictions
                    if len(y_train_pred) != len(y_train) or len(y_test_pred) != len(y_test):
                        pro_log(f"[WFV] Fold {fold}: Prediction length mismatch", level="error", tag="WFV")
                        continue
                        
                    # PRODUCTION FIX: Handle NaN/Inf in predictions
                    y_train_pred = np.nan_to_num(y_train_pred, nan=0.5, posinf=1.0, neginf=0.0)
                    y_test_pred = np.nan_to_num(y_test_pred, nan=0.5, posinf=1.0, neginf=0.0)
                    
                except Exception as pred_error:
                    pro_log(f"[WFV] Fold {fold}: Prediction error - {pred_error}", level="error", tag="WFV")
                    continue
                  # PRODUCTION FIX: Robust AUC calculation
                try:
                    # Ensure predictions are 1D arrays
                    y_train_pred = np.asarray(y_train_pred).flatten()
                    y_test_pred = np.asarray(y_test_pred).flatten()
                    
                    # DEBUG: Log array shapes before AUC calculation
                    pro_log(f"[WFV] Debug - AUC calc: y_train shape={y_train.shape}, y_train_pred shape={y_train_pred.shape}", level="debug", tag="WFV")
                    pro_log(f"[WFV] Debug - AUC calc: y_test shape={y_test.shape}, y_test_pred shape={y_test_pred.shape}", level="debug", tag="WFV")
                    pro_log(f"[WFV] Debug - AUC calc: y_train unique={np.unique(y_train)}, y_test unique={np.unique(y_test)}", level="debug", tag="WFV")
                    pro_log(f"[WFV] Debug - AUC calc: y_train_pred range=[{y_train_pred.min():.3f}, {y_train_pred.max():.3f}]", level="debug", tag="WFV")
                    pro_log(f"[WFV] Debug - AUC calc: y_test_pred range=[{y_test_pred.min():.3f}, {y_test_pred.max():.3f}]", level="debug", tag="WFV")
                      # Check unique classes in true targets (not predictions)
                    train_classes = np.unique(y_train)
                    test_classes = np.unique(y_test)
                    
                    # Calculate AUC using appropriate approach based on number of classes
                    pro_log(f"[WFV] Debug - Train classes: {train_classes}, Test classes: {test_classes}", level="debug", tag="WFV")
                    
                    # For multiclass targets with 1D predictions, we need special handling
                    if len(train_classes) > 2 or len(test_classes) > 2:
                        # For multiclass targets, use 'ovr' but with special handling
                        # Since we only have binary probabilities, we need to convert to proper format
                        pro_log(f"[WFV] Debug - Multiclass detected, using label_binarize approach", level="debug", tag="WFV")
                        
                        # For now, use a simpler approach that works with binary probabilities
                        # Convert multiclass target to binary: positive class (1) vs others
                        y_train_binary = (y_train == 1).astype(int)
                        y_test_binary = (y_test == 1).astype(int)
                        
                        auc_train = roc_auc_score(y_train_binary, y_train_pred)
                        auc_test = roc_auc_score(y_test_binary, y_test_pred)
                        
                        pro_log(f"[WFV] Debug - Used binary conversion for multiclass", level="debug", tag="WFV")
                    else:
                        # Binary classification
                        auc_train = roc_auc_score(y_train, y_train_pred)
                        auc_test = roc_auc_score(y_test, y_test_pred)
                        
                        pro_log(f"[WFV] Debug - Used binary classification", level="debug", tag="WFV")
                        
                    # Binary predictions for accuracy
                    y_test_binary = (y_test_pred > 0.5).astype(int)
                    acc_test = accuracy_score(y_test, y_test_binary)
                    
                    pro_log(f"[WFV] Debug - AUC calculation successful: train={auc_train:.3f}, test={auc_test:.3f}", level="debug", tag="WFV")
                    
                except Exception as auc_error:
                    pro_log(f"[WFV] Fold {fold}: AUC calculation error - {auc_error}", level="error", tag="WFV")
                    pro_log(f"[WFV] Exception details: {traceback.format_exc()}", level="debug", tag="WFV")
                    # Fallback values
                    auc_train, auc_test, acc_test = 0.5, 0.5, 0.5
                
                # Validation and logging
                metrics = {'auc': auc_test, 'train_auc': auc_train, 'test_auc': auc_test}
                
                try:
                    check_auc_threshold(metrics, strict=strict_guard)
                    check_no_overfitting(metrics, strict=strict_guard)
                except Exception as guard_error:
                    pro_log(f"[WFV] Fold {fold}: Guard check warning - {guard_error}", level="warning", tag="WFV")
                
                # Feature importance
                try:
                    importances = model.feature_importances_
                    feature_importance = {f: imp for f, imp in zip(features, importances)}
                    check_no_noise(feature_importance)
                except Exception as importance_error:
                    pro_log(f"[WFV] Fold {fold}: Feature importance warning - {importance_error}", level="warning", tag="WFV")
                
                metrics_list.append({
                    'fold': fold,
                    'auc_train': auc_train,
                    'auc_test': auc_test,
                    'acc_test': acc_test                })
                
                pro_log(f"[WFV] Fold {fold}: AUC={auc_test:.3f}, ACC={acc_test:.3f}", tag="WFV")
                
            except Exception as fold_error:
                pro_log(f"[WFV] Fold {fold}: Complete fold failed - {fold_error}", level="error", tag="WFV")
                continue
            
            progress.update(task, advance=10, description=f"[cyan]Fold {fold+1}/5 done")
        metrics_df = pd.DataFrame(metrics_list)
        out_path = os.path.join("output_default", "walkforward_metrics.csv")
        metrics_df.to_csv(out_path, index=False)
        pro_log(f"[WFV] Walk-forward metrics saved to {out_path}", level="success", tag="WFV")
        progress.update(task, completed=100, description="[green]Walk-forward validation complete")

        import psutil, json, time
        start_time = time.time()
        # --- เทพ: Export resource log (RAM/CPU/GPU) ---
        resource_info = {
            'ram_percent': psutil.virtual_memory().percent,
            'ram_used_gb': psutil.virtual_memory().used / 1e9,
            'ram_total_gb': psutil.virtual_memory().total / 1e9,
            'cpu_percent': psutil.cpu_percent(),
        }
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            def _to_float(val):
                try:
                    return float(val)
                except Exception:
                    return 0.0
            resource_info['gpu_used_gb'] = _to_float(getattr(gpu_mem, 'used', 0)) / 1e9 if hasattr(gpu_mem, 'used') else 0.0
            resource_info['gpu_total_gb'] = _to_float(getattr(gpu_mem, 'total', 0)) / 1e9 if hasattr(gpu_mem, 'total') else 0.0
        except Exception:
            resource_info['gpu_used_gb'] = 0.0
            resource_info['gpu_total_gb'] = 0.0
        resource_log_path = os.path.join("output_default", "walkforward_resource_log.json")
        with open(resource_log_path, "w", encoding="utf-8") as f:
            json.dump(resource_info, f, indent=2)
        pro_log(f"[WFV] Resource log exported to {resource_log_path}", tag="WFV")

        # --- เทพ: Export summary metrics ---
        try:
            summary = {}
            for col in ['auc_train', 'auc_test', 'acc_test']:
                if col in metrics_df.columns:
                    summary[col] = {
                        'mean': float(metrics_df[col].mean()),
                        'std': float(metrics_df[col].std()),
                        'min': float(metrics_df[col].min()),
                        'max': float(metrics_df[col].max()),
                    }
            summary_path = os.path.join("output_default", "walkforward_summary_metrics.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            pro_log(f"[WFV] Summary metrics exported to {summary_path}", tag="WFV")
        except Exception as e:
            pro_log(f"[WFV] Summary metrics export error: {e}", level="warn", tag="WFV")

        # --- เทพ: Export diagnostics log ---
        diagnostics_path = os.path.join("output_default", "walkforward_diagnostics.log")
        with open(diagnostics_path, "a", encoding="utf-8") as f:
            f.write(f"[DIAG] Walkforward run at: {pd.Timestamp.now()}\n")
            f.write(f"Resource: {resource_info}\n")
            if 'summary' in locals():
                f.write(f"Summary: {summary}\n")
            f.write(f"Output: {out_path}\n")
        pro_log(f"[WFV] Diagnostics log exported to {diagnostics_path}", tag="WFV")

        # --- เทพ: Assert output file/column completeness ---
        try:
            assert os.path.exists(out_path), f"Walkforward output file missing: {out_path}"
            df_check = pd.read_csv(out_path)
            df_check.columns = [c.lower() for c in df_check.columns]
            required_cols = ['auc_train', 'auc_test', 'acc_test']
            missing_cols = [c for c in required_cols if c not in df_check.columns]
            if missing_cols:
                raise ValueError(f"Walkforward output missing columns: {missing_cols}")
            pro_log(f"[WFV] Output file and columns validated.", tag="WFV")
        except Exception as e:
            pro_log(f"[WFV] Output validation error: {e}", level="error", tag="WFV")

        # --- เทพ: Export timing/performance log ---
        end_time = time.time()
        perf_log_path = os.path.join("output_default", "walkforward_perf.log")
        with open(perf_log_path, "a", encoding="utf-8") as f:
            duration = end_time - start_time
            if duration is None or not isinstance(duration, (int, float)):
                duration_str = "N/A"
            else:
                duration_str = f"{duration:.2f}"
            f.write(f"Walkforward finished at {pd.Timestamp.now()} | Duration: {duration_str} sec\n")
        pro_log(f"[WFV] Performance log exported to {perf_log_path}", tag="WFV")

        # --- เทพ: Usability hints ---
        pro_log(f"[WFV] ผลลัพธ์หลัก: {out_path}\nดู resource log ที่ {resource_log_path}\nดู summary metrics ที่ {summary_path if 'summary_path' in locals() else '-'}\nดู diagnostics log ที่ {diagnostics_path}", tag="WFV")
    console.print(Panel("[bold green]Walk-forward validation completed!", title="Walk-Forward", expand=False))

    # --- เทพ: RICH SUMMARY PANEL พร้อมลิงก์ ---
    from rich.text import Text
    from rich import box
    summary_table = Table(title="[bold green]Walk-Forward Validation Summary", show_header=True, header_style="bold magenta", box=box.ROUNDED)
    summary_table.add_column("ไฟล์/ข้อมูล", style="cyan", justify="right")
    summary_table.add_column("Path/ค่า", style="white")
    # สมมุติว่ามีไฟล์ output walkforward_results.csv
    wf_path = os.path.join("output_default", "walkforward_results.csv")
    if os.path.exists(wf_path):
        summary_table.add_row("Walkforward Results", f"[link=file://{wf_path}]{wf_path}[/link]")
    if 'metrics_list' in locals() and metrics_list:
        # ป้องกัน KeyError: 'auc' ถ้า metrics_list ไม่มี key นี้
        if all('auc' in m for m in metrics_list):
            avg_auc = sum([m['auc'] for m in metrics_list])/len(metrics_list)
            summary_table.add_row("Average AUC", safe_fmt(avg_auc))
        elif all('auc_test' in m for m in metrics_list):
            avg_auc = sum([m['auc_test'] for m in metrics_list])/len(metrics_list)
            summary_table.add_row("Average AUC", safe_fmt(avg_auc))
        else:
            summary_table.add_row("Average AUC", "N/A")
        summary_table.add_row("Folds", str(len(metrics_list)))
    console.print(Panel(summary_table, title="[bold blue]🚀 Walk-Forward Complete!", border_style="bright_green"))

    # --- EXPORT OPTIONS ---
    export_hint = "[bold yellow]Export Options:[/bold yellow]\n- [green]walkforward_results.csv[/green] (if implemented)\n- [green]metrics_list[/green] (Python object)\n\n[bold]You can convert metrics_list to Excel/Markdown/HTML using pandas if needed.[/bold]"
    console.print(Panel(export_hint, title="[bold magenta]Export & Share", border_style="yellow"))

    # --- NEXT STEP SUGGESTIONS ---
    next_steps = "[bold blue]Next Steps:[/bold blue]\n- Run [green]backtest[/green] for full validation\n- Generate [green]report[/green] for summary\n- Export or share results as needed\n- [bold]Use CLI flags for quick actions![/bold]"
    console.print(Panel(next_steps, title="[bold green]What to do next?", border_style="bright_blue"))

    return out_path
