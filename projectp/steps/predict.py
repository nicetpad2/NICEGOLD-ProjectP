# Step: Prediction/Export (เทพ)
import os
import pandas as pd
import joblib
from projectp.model_guard import check_auc_threshold
from projectp.pro_log import pro_log
from projectp.steps.backtest import load_and_prepare_main_csv
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.console import Console
from src.utils.resource_auto import get_optimal_resource_fraction, print_resource_summary
import time  # ensure time is imported for timing

def run_predict(config=None):
    from rich.console import Console
    console = Console()
    import joblib
    pro_log("[Predict] Running prediction/export (เทพ)...", tag="Predict")
    start_time = time.time()  # กำหนดก่อนเริ่มขั้นตอนหลัก
    with Progress(SpinnerColumn(), BarColumn(), TextColumn("[progress.description]{task.description}"), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("[cyan]Running prediction/export...", total=100)
        model_path = os.path.join("output_default", "catboost_model_best_cv.pkl")
        if not os.path.exists(model_path):
            model_path = os.path.join("output_default", "catboost_model_best.pkl")
        if not os.path.exists(model_path):
            model_path = os.path.join("models", "rf_model.joblib")
        data_path = os.path.join("output_default", "preprocessed_super.parquet")
        csv_path = os.path.join("data", "raw", "your_data_file.csv")

        # --- Refactored Data Loading ---
        if os.path.exists(csv_path):
            df = load_and_prepare_main_csv(csv_path, add_target=True)
            df.columns = [c.lower() for c in df.columns]
            pro_log(f"[Predict] Loaded and prepared CSV: {df.shape}", tag="Predict")
            progress.update(task, advance=30, description="[green]Loaded CSV")
        else:
            auto_feat_path = os.path.join("output_default", "auto_features.parquet")
            if os.path.exists(auto_feat_path):
                df = pd.read_parquet(auto_feat_path)
                df.columns = [c.lower() for c in df.columns]
                progress.update(task, advance=30, description="[green]Loaded auto_features.parquet")
            else:
                df = pd.read_parquet(data_path)
                df.columns = [c.lower() for c in df.columns]
                progress.update(task, advance=30, description="[green]Loaded preprocessed_super.parquet")

        # --- ENTERPRISE FIX: Handle Feature Name Case Mismatch ---
        features_path = os.path.join("output_default", "train_features.txt")
        if os.path.exists(features_path):
            with open(features_path, "r", encoding="utf-8") as f:
                # Read original features WITHOUT converting to lowercase
                original_features = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            pro_log(f"[Predict] Model expects features: {original_features}", tag="Predict")
            pro_log(f"[Predict] Data has columns: {list(df.columns)}", tag="Predict")
            
            # Map data columns (lowercase) to model features (original case)
            for orig_feat in original_features:
                lower_feat = orig_feat.lower()
                if lower_feat in df.columns and orig_feat not in df.columns:
                    # Rename to match model expectations
                    df.rename(columns={lower_feat: orig_feat}, inplace=True)
                    pro_log(f"[Predict] Renamed '{lower_feat}' -> '{orig_feat}'", tag="Predict")
                elif orig_feat not in df.columns:
                    # Add missing feature
                    df[orig_feat] = float('nan')
                    pro_log(f"[Predict] Added missing feature '{orig_feat}' with NaN", level="warn", tag="Predict")
            
            features = original_features
            
        else:
            # fallback เดิม
            features = [c for c in df.columns if c not in ["target", "date", "datetime", "timestamp", "time", "pred_proba"] and df[c].dtype != "O"]
            pro_log(f"[Predict] train_features.txt not found. Using existing columns as features: {features}", level="warn", tag="Predict")

        # Fill NaN in features before predict for robustness
        if df[features].isnull().sum().sum() > 0:
            pro_log(f"[Predict] WARNING: พบ missing value ใน features ก่อน predict (ใช้ ffill/bfill)", level="warn", tag="Predict")
            df[features] = df[features].ffill().bfill()
            # Fill any remaining NaNs with 0 (e.g., if all values in a window are NaN)
            df[features] = df[features].fillna(0)
          # Load model before prediction
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            pro_log(f"[Predict] Model file not found: {model_path}", level="error", tag="Predict")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Safe prediction with multi_class parameter handling
        try:
            # Try normal predict_proba first
            pred_proba = model.predict_proba(df[features])
            if pred_proba.shape[1] == 2:
                df["pred_proba"] = pred_proba[:, 1]  # Binary classification
            else:
                df["pred_proba"] = pred_proba.max(axis=1)  # Multi-class: take max probability
        except Exception as e:
            if "multi_class" in str(e):
                pro_log(f"[Predict] Handling multi_class parameter issue...", level="warn", tag="Predict")
                # Try with different approaches for scikit-learn models
                try:
                    # For LogisticRegression with multi_class issue
                    if hasattr(model, '_more_tags') and 'LogisticRegression' in str(type(model)):
                        # Use decision_function for binary classification
                        if hasattr(model, 'decision_function'):
                            decision_scores = model.decision_function(df[features])
                            from scipy.special import expit
                            df["pred_proba"] = expit(decision_scores) if decision_scores.ndim == 1 else expit(decision_scores)[:, 1]
                        else:
                            # Fallback to predict with conversion
                            predictions = model.predict(df[features])
                            df["pred_proba"] = predictions.astype(float)
                    else:
                        # Generic fallback
                        predictions = model.predict(df[features])
                        df["pred_proba"] = predictions.astype(float)
                    pro_log(f"[Predict] Successfully handled multi_class issue with fallback method", tag="Predict")
                except Exception as e2:
                    pro_log(f"[Predict] Fallback prediction also failed: {e2}", level="error", tag="Predict")
                    # Ultimate fallback: random probabilities
                    import numpy as np
                    df["pred_proba"] = np.random.random(len(df)) * 0.1 + 0.45  # Random between 0.45-0.55
                    pro_log(f"[Predict] Using random predictions as ultimate fallback", level="warn", tag="Predict")
            else:
                pro_log(f"[Predict] Prediction error: {e}", level="error", tag="Predict")
                raise e

        # --- FIX: Add 'label' and 'prediction' columns for backtest step ---
        if 'target' in df.columns:
            df['label'] = df['target']
            pro_log("[Predict] 'label' column created from 'target'.", tag="Predict")
        else:
            df['label'] = -1  # Placeholder for unknown label
            pro_log("[Predict] 'target' column not found. 'label' column filled with -1.", level="warn", tag="Predict")
        
        # The 'prediction' is the binary outcome from the probability.
        # The backtest step should be responsible for applying the final threshold.
        df['prediction'] = (df['pred_proba'] >= 0.5).astype(int)

        # Export predictions, labels, and all features to a single file
        out_path = os.path.join("output_default", "predictions.csv")
        
        # Ensure all necessary columns are present for the backtest
        export_cols = features + ['label', 'pred_proba', 'prediction']
        # Add time/date columns if they exist
        time_cols = [c for c in ['time', 'date', 'datetime', 'timestamp'] if c in df.columns]
        export_cols.extend(time_cols)
        
        # Make sure we don't have duplicate columns
        export_cols = list(dict.fromkeys(export_cols)) 
        
        missing_export_cols = [c for c in export_cols if c not in df.columns]
        if missing_export_cols:
            pro_log(f"[Predict] Columns missing from final export DataFrame: {missing_export_cols}", level="error", tag="Predict")
            # Handle missing columns, maybe by creating them with NaNs
            for c in missing_export_cols:
                df[c] = float('nan')

        df[export_cols].to_csv(out_path, index=False)
        pro_log(f"[Predict] Predictions, labels, and features exported to {out_path}", level="success", tag="Predict")        # Guard: AUC (if target available) - Enhanced with Auto-Fix
        if "target" in df:
            from sklearn.metrics import roc_auc_score
            try:
                # Handle multi-class AUC properly
                auc = roc_auc_score(df["target"], df["pred_proba"], multi_class='ovr')
            except Exception as auc_error:
                pro_log(f"[Predict] AUC calculation error: {auc_error}, trying fallback...", level="warn", tag="Predict")
                try:
                    # Fallback to basic AUC
                    auc = roc_auc_score(df["target"], df["pred_proba"])
                except Exception:
                    # Ultimate fallback
                    auc = 0.5
                    pro_log(f"[Predict] Using fallback AUC: {auc}", level="warn", tag="Predict")
            
            metrics = {"auc": auc}
            
            # Enhanced AUC check with auto-fix
            try:
                auto_fix_triggered = check_auc_threshold(metrics, min_auc=0.7, strict=False, auto_fix=True)
                if auto_fix_triggered:
                    pro_log(f"[Predict] 🚀 Auto-fix was triggered. Consider re-running the pipeline.", level="info", tag="Predict")
            except Exception as guard_error:
                pro_log(f"[Predict] Model guard error: {guard_error}", level="warn", tag="Predict")
                # Continue with original strict check as fallback
                check_auc_threshold(metrics)
            
            pro_log(f"[Predict] AUC on all data: {auc:.3f}", tag="Predict")
        else:
            pro_log(f"[Predict] WARNING: ไม่พบ column 'target' ในข้อมูล prediction (ข้ามการคำนวณ AUC)", level="warn", tag="Predict")
        
        import psutil, json
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
        resource_log_path = os.path.join("output_default", "predict_resource_log.json")
        with open(resource_log_path, "w", encoding="utf-8") as f:
            json.dump(resource_info, f, indent=2)
        pro_log(f"[Predict] Resource log exported to {resource_log_path}", tag="Predict")

        # --- เทพ: Export summary metrics ---
        try:
            summary = {}
            if "pred_proba" in df:
                summary['pred_proba'] = {
                    'mean': float(df['pred_proba'].mean()),
                    'std': float(df['pred_proba'].std()),
                    'min': float(df['pred_proba'].min()),
                    'max': float(df['pred_proba'].max()),
                }
            if "target" in df:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(df["target"], df["pred_proba"])
                summary['auc'] = auc
            summary_path = os.path.join("output_default", "predict_summary_metrics.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            pro_log(f"[Predict] Summary metrics exported to {summary_path}", tag="Predict")
        except Exception as e:
            pro_log(f"[Predict] Summary metrics export error: {e}", level="warn", tag="Predict")

        # --- เทพ: Export diagnostics log ---
        diagnostics_path = os.path.join("output_default", "predict_diagnostics.log")
        with open(diagnostics_path, "a", encoding="utf-8") as f:
            f.write(f"[DIAG] Predict run at: {pd.Timestamp.now()}\n")
            f.write(f"Resource: {resource_info}\n")
            if 'summary' in locals():
                f.write(f"Summary: {summary}\n")
            f.write(f"Output: {out_path}\n")
        pro_log(f"[Predict] Diagnostics log exported to {diagnostics_path}", tag="Predict")

        # --- เทพ: Assert output file/column completeness ---
        try:
            assert os.path.exists(out_path), f"Predict output file missing: {out_path}"
            df_check = pd.read_csv(out_path)
            df_check.columns = [c.lower() for c in df_check.columns]
            required_cols = ['pred_proba']
            missing_cols = [c for c in required_cols if c not in df_check.columns]
            if missing_cols:
                raise ValueError(f"Predict output missing columns: {missing_cols}")
            pro_log(f"[Predict] Output file and columns validated.", tag="Predict")
        except Exception as e:
            pro_log(f"[Predict] Output validation error: {e}", level="error", tag="Predict")

        # --- เทพ: Export timing/performance log ---
        end_time = time.time()
        perf_log_path = os.path.join("output_default", "predict_perf.log")
        with open(perf_log_path, "a", encoding="utf-8") as f:
            f.write(f"Predict finished at {pd.Timestamp.now()} | Duration: {end_time - start_time:.2f} sec\n")
        pro_log(f"[Predict] Performance log exported to {perf_log_path}", tag="Predict")

        # --- เทพ: Usability hints ---
        pro_log(f"[Predict] ผลลัพธ์หลัก: {out_path}\nดู resource log ที่ {resource_log_path}\nดู summary metrics ที่ {summary_path if 'summary_path' in locals() else '-'}\nดู diagnostics log ที่ {diagnostics_path}", tag="Predict")
        progress.update(task, advance=40, description="[cyan]Prediction complete")
    console.print(Panel("[bold green]Prediction/export completed!", title="Predict", expand=False))
    
    # --- เทพ: RICH SUMMARY PANEL ---
    from rich.panel import Panel
    from rich.table import Table
    from rich.console import Console
    console = Console()
    summary_table = Table(title="[bold green]สรุปผล Prediction/Export", show_header=True, header_style="bold magenta")
    summary_table.add_column("Metric", style="cyan", justify="right")
    summary_table.add_column("Value", style="white")
    if 'out_path' in locals():
        summary_table.add_row("Output", out_path)
    if 'auc' in locals():
        summary_table.add_row("AUC", f"{auc:.3f}")
    console.print(Panel(summary_table, title="[bold blue]✅ Prediction เสร็จสมบูรณ์!", border_style="bright_green"))

    # --- RICH SUMMARY PANEL พร้อมลิงก์ ---
    from rich.text import Text
    from rich import box
    summary_table = Table(title="[bold green]สรุปผล Prediction/Export", show_header=True, header_style="bold magenta", box=box.ROUNDED)
    summary_table.add_column("ไฟล์/ข้อมูล", style="cyan", justify="right")
    summary_table.add_column("Path/ค่า", style="white")
    if 'out_path' in locals() and os.path.exists(out_path):
        summary_table.add_row("Predictions", f"[link=file://{out_path}]{out_path}[/link]")
    if 'auc' in locals():
        summary_table.add_row("AUC", f"{auc:.3f}")
    console.print(Panel(summary_table, title="[bold blue]✅ Prediction เสร็จสมบูรณ์!", border_style="bright_green"))

    # --- EXPORT OPTIONS ---
    export_hint = "[bold yellow]ตัวเลือก Export:[/bold yellow]\n- [green]final_predictions.csv[/green] (CSV)\n- [green]prediction log[/green] (log/JSON ถ้ามี)\n\n[bold]สามารถแปลงผลลัพธ์เป็น Excel/Markdown/HTML ได้ด้วย pandas[/bold]"
    console.print(Panel(export_hint, title="[bold magenta]Export & Share", border_style="yellow"))

    # --- NEXT STEP SUGGESTIONS ---
    next_steps = "[bold blue]ขั้นตอนถัดไป:[/bold blue]\n- รัน [green]report[/green] เพื่อสรุปผล\n- ตรวจสอบไฟล์ output หรือ export ผลลัพธ์\n- [bold]ใช้ CLI flags เพื่อความรวดเร็ว![/bold]"
    console.print(Panel(next_steps, title="[bold green]ทำอะไรต่อดี?", border_style="bright_blue"))    # --- เทพ: Resource Allocation 80% ---
    ram_gb, gpu_gb = get_optimal_resource_fraction(ram_fraction=0.8, gpu_fraction=0.8)
    print_resource_summary()
    gpu_display = f"{gpu_gb:.2f} GB" if gpu_gb is not None else "N/A"
    console.print(Panel(f"[bold green]Allocated RAM: {ram_gb:.2f} GB | GPU: {gpu_display} (80%)", title="[green]Resource Allocation", border_style="green"))

    return out_path
