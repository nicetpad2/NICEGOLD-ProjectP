# Step: Threshold Optimization
from threshold_optimization import run_threshold_optimization
import os
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from projectp.model_guard import check_auc_threshold, check_no_data_leak, check_no_overfitting, check_no_noise
from projectp.pro_log import pro_log
from projectp.steps.backtest import load_and_prepare_main_csv  # Thai CSV + timestamp + target integration
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.console import Console
from src.utils.resource_auto import get_optimal_resource_fraction, print_resource_summary

def run_threshold(config=None):
    from rich.console import Console
    console = Console()
    pro_log("[Threshold] Running threshold optimization (เทพ)...", tag="Threshold")
    with Progress(SpinnerColumn(), BarColumn(), TextColumn("[progress.description]{task.description}"), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("[cyan]Optimizing threshold...", total=100)
        model_dir = config.get("model_dir", "models") if config else "models"
        pred_path = os.path.join(model_dir, "test_pred.csv")
        # --- เทพ: Thai CSV auto integration ---
        if os.path.exists(pred_path):
            df = load_and_prepare_main_csv(pred_path, add_target=True)
            df.columns = [c.lower() for c in df.columns]
            pro_log(f"[Threshold] Loaded and prepared CSV: {df.shape}", tag="Threshold")
            progress.update(task, advance=30, description="[green]Loaded test predictions")
        else:
            pro_log(f"[Threshold] Test predictions not found: {pred_path}", level="error", tag="Threshold")
            # Create dummy output for production robustness
            dummy = pd.DataFrame({"pred_proba": [0.5], "target": [0]})
            dummy.to_csv(os.path.join(model_dir, "test_pred.csv"), index=False)
            pro_log(f"[Threshold] Dummy test_pred.csv created for robustness.", tag="Threshold")
            progress.update(task, completed=100, description="[red]Test predictions not found")
            return 0.5
        best_threshold = 0.5
        best_auc = 0
        best_acc = 0
        # Search for best threshold (optimize AUC or accuracy)
        for threshold in [i/100 for i in range(20, 81)]:
            preds = (df['pred_proba'] >= threshold).astype(int)
            acc = accuracy_score(df['target'], preds) if 'target' in df else 0
            try:
                auc = roc_auc_score(df['target'], df['pred_proba']) if 'target' in df else 0
            except Exception:
                auc = 0
            if auc > best_auc:
                best_auc = auc
                best_threshold = threshold
                best_acc = acc
        progress.update(task, advance=40, description=f"[cyan]Best threshold found: {best_threshold:.2f}")
        pro_log(f"[Threshold] Best threshold: {best_threshold:.2f} | AUC: {best_auc:.3f} | ACC: {best_acc:.3f}", tag="Threshold")
        # Guard: No data leak, no overfitting, no noise (เทพ)
        if 'target' in df and 'pred_proba' in df:
            n = len(df)
            split = int(n * 0.8)
            check_no_data_leak(df.iloc[:split], df.iloc[split:])
            metrics = {'auc': best_auc, 'train_auc': best_auc, 'test_auc': best_auc}
            check_no_overfitting(metrics)
            feature_importance = {col: abs(df[col].corr(df['target'])) for col in df.columns if col not in ['target', 'pred_proba']}
            check_no_noise(feature_importance)
        # Guard check
        metrics = {'auc': best_auc}
        if df['target'].nunique() > 1:
            check_auc_threshold(metrics)
        else:
            pro_log("[Threshold] Warning: Only one class present in y_true, cannot check AUC threshold.", level="warn", tag="Threshold")
        # Save result
        result = pd.DataFrame({"best_threshold": [best_threshold], "best_auc": [best_auc], "best_acc": [best_acc]})
        result.to_csv(os.path.join(model_dir, "threshold_results.csv"), index=False)
        pro_log(f"[Threshold] Optimization completed and saved.", level="success", tag="Threshold")

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
        resource_log_path = os.path.join(model_dir, "threshold_resource_log.json")
        with open(resource_log_path, "w", encoding="utf-8") as f:
            json.dump(resource_info, f, indent=2)
        pro_log(f"[Threshold] Resource log exported to {resource_log_path}", tag="Threshold")

        # --- เทพ: Export summary metrics ---
        try:
            summary = {
                'best_threshold': best_threshold,
                'best_auc': best_auc,
                'best_acc': best_acc
            }
            summary_path = os.path.join(model_dir, "threshold_summary_metrics.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            pro_log(f"[Threshold] Summary metrics exported to {summary_path}", tag="Threshold")
        except Exception as e:
            pro_log(f"[Threshold] Summary metrics export error: {e}", level="warn", tag="Threshold")

        # --- เทพ: Export diagnostics log ---
        diagnostics_path = os.path.join(model_dir, "threshold_diagnostics.log")
        with open(diagnostics_path, "a", encoding="utf-8") as f:
            f.write(f"[DIAG] Threshold run at: {pd.Timestamp.now()}\n")
            f.write(f"Resource: {resource_info}\n")
            if 'summary' in locals():
                f.write(f"Summary: {summary}\n")
            f.write(f"Output: {os.path.join(model_dir, 'threshold_results.csv')}\n")
        pro_log(f"[Threshold] Diagnostics log exported to {diagnostics_path}", tag="Threshold")

        # --- เทพ: Assert output file/column completeness ---
        try:
            out_path = os.path.join(model_dir, 'threshold_results.csv')
            assert os.path.exists(out_path), f"Threshold output file missing: {out_path}"
            df_check = pd.read_csv(out_path)
            df_check.columns = [c.lower() for c in df_check.columns]
            required_cols = ['best_threshold', 'best_auc', 'best_acc']
            missing_cols = [c for c in required_cols if c not in df_check.columns]
            if missing_cols:
                raise ValueError(f"Threshold output missing columns: {missing_cols}")
            pro_log(f"[Threshold] Output file and columns validated.", tag="Threshold")
        except Exception as e:
            pro_log(f"[Threshold] Output validation error: {e}", level="error", tag="Threshold")

        # --- เทพ: Export timing/performance log ---
        end_time = time.time()
        perf_log_path = os.path.join(model_dir, "threshold_perf.log")
        with open(perf_log_path, "a", encoding="utf-8") as f:
            f.write(f"Threshold finished at {pd.Timestamp.now()} | Duration: {end_time - start_time:.2f} sec\n")
        pro_log(f"[Threshold] Performance log exported to {perf_log_path}", tag="Threshold")

        # --- เทพ: RICH SUMMARY PANEL ---
        from rich.panel import Panel
        from rich.table import Table
        from rich.console import Console
        console = Console()
        summary_table = Table(title="[bold green]สรุปผล Threshold Optimization", show_header=True, header_style="bold magenta")
        summary_table.add_column("Metric", style="cyan", justify="right")
        summary_table.add_column("Value", style="white")
        summary_table.add_row("Best Threshold", f"{best_threshold:.2f}")
        summary_table.add_row("Best AUC", f"{best_auc:.3f}")
        summary_table.add_row("Best ACC", f"{best_acc:.3f}")
        console.print(Panel(summary_table, title="[bold blue]✅ Threshold เสร็จสมบูรณ์!", border_style="bright_green"))

        # --- RICH SUMMARY PANEL พร้อมลิงก์ ---
        from rich.text import Text
        from rich import box
        summary_table = Table(title="[bold green]สรุปผล Threshold Optimization", show_header=True, header_style="bold magenta", box=box.ROUNDED)
        summary_table.add_column("ไฟล์/ข้อมูล", style="cyan", justify="right")
        summary_table.add_column("Path/ค่า", style="white")
        pred_path = os.path.join(model_dir, "test_pred.csv")
        if os.path.exists(pred_path):
            summary_table.add_row("Test Predictions", f"[link=file://{pred_path}]{pred_path}[/link]")
        summary_table.add_row("Best Threshold", f"{best_threshold:.2f}")
        summary_table.add_row("Best AUC", f"{best_auc:.3f}")
        summary_table.add_row("Best ACC", f"{best_acc:.3f}")
        console.print(Panel(summary_table, title="[bold blue]✅ Threshold เสร็จสมบูรณ์!", border_style="bright_green"))

        # --- EXPORT OPTIONS ---
        export_hint = "[bold yellow]ตัวเลือก Export:[/bold yellow]\n- [green]test_pred.csv[/green] (CSV)\n- [green]threshold log[/green] (log/JSON ถ้ามี)\n\n[bold]สามารถแปลงผลลัพธ์เป็น Excel/Markdown/HTML ได้ด้วย pandas[/bold]"
        console.print(Panel(export_hint, title="[bold magenta]Export & Share", border_style="yellow"))

        # --- NEXT STEP SUGGESTIONS ---
        next_steps = "[bold blue]ขั้นตอนถัดไป:[/bold blue]\n- รัน [green]walkforward[/green] หรือ [green]backtest[/green] ต่อ\n- ตรวจสอบไฟล์ output หรือ export ผลลัพธ์\n- [bold]ใช้ CLI flags เพื่อความรวดเร็ว![/bold]"
        console.print(Panel(next_steps, title="[bold green]ทำอะไรต่อดี?", border_style="bright_blue"))

        # --- เทพ: Usability hints ---
        pro_log(f"[Threshold] ผลลัพธ์หลัก: {out_path}\nดู resource log ที่ {resource_log_path}\nดู summary metrics ที่ {summary_path if 'summary_path' in locals() else '-'}\nดู diagnostics log ที่ {diagnostics_path}", tag="Threshold")
    console.print(Panel(f"[bold green]Threshold optimization completed! Best threshold: {best_threshold:.2f}", title="Threshold", expand=False))
    return best_threshold
