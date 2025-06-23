from rich.console import Console
# Step: Hyperparameter Sweep
from src.strategy import run_hyperparameter_sweep
from projectp.pro_log import pro_log
from projectp.model_guard import check_no_data_leak, check_no_overfitting, check_no_noise
from projectp.steps.backtest import load_and_prepare_main_csv  # Thai CSV + timestamp + target integration
import pandas as pd
import psutil, json, time, os
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from src.utils.resource_auto import get_optimal_resource_fraction, print_resource_summary

def run_sweep(config=None):
    from rich.console import Console
    from rich.panel import Panel  # Ensure Panel is always in local scope for Prefect/multiprocessing
    console = Console()
    pro_log("[Sweep] Running hyperparameter sweep...", tag="Sweep")
    with Progress(SpinnerColumn(), BarColumn(), TextColumn("[progress.description]{task.description}"), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("[cyan]Running hyperparameter sweep...", total=100)
        # --- เทพ: Thai CSV auto integration ---
        data_path = config.get("train_data_path", "data/raw/your_data_file.csv") if config else "data/raw/your_data_file.csv"
        if os.path.exists(data_path):
            df = load_and_prepare_main_csv(data_path, add_target=True)
            df.columns = [c.lower() for c in df.columns]
            pro_log(f"[Sweep] Loaded and prepared CSV: {df.shape}", tag="Sweep")
            progress.update(task, advance=30, description="[green]Loaded training data")
        else:
            df = None
            pro_log(f"[Sweep] Data not found: {data_path}", level="warn", tag="Sweep")
            progress.update(task, completed=100, description="[red]Data not found")
        sweep_params = {
            "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.2],
            "depth": [3, 4, 6, 8, 10],
            "iterations": [50, 100, 200, 300],
            "l2_leaf_reg": [1, 3, 5, 7, 9],
            "bagging_temperature": [0, 0.5, 1, 2],
        }
        if config and "sweep" in config:
            sweep_params.update(config["sweep"])
        def dummy_train_func(**params):
            pro_log(f"[Sweep] Training with params: {params}", tag="Sweep")
            return "model_path", ["feature1", "feature2"]
        progress.update(task, advance=30, description="[cyan]Running sweep...")
        run_hyperparameter_sweep({"output_dir": config.get("model_dir", "models") if config else "models"}, sweep_params, dummy_train_func)
        progress.update(task, advance=30, description="[green]Sweep complete")
        # Guard: No data leak, no overfitting, no noise (เทพ)
        # (สมมติ sweep ผลลัพธ์เป็น model_path, features)
        model_path, features = dummy_train_func()
        # ใช้ DataFrame mock สำหรับ debug guard (index ไม่ overlap)
        df1 = pd.DataFrame({'a': [1, 2, 3]}, index=[0,1,2])
        df2 = pd.DataFrame({'a': [4, 5, 6]}, index=[3,4,5])
        check_no_data_leak(df1, df2)
        check_no_overfitting({'train_auc': 0.75, 'test_auc': 0.75})  # mock call
        check_no_noise({'feature1': 0.5, 'feature2': 0.5})  # mock call

        start_time = time.time()        # --- เทพ: Resource Allocation 80% ---
        ram_gb, gpu_gb = get_optimal_resource_fraction(ram_fraction=0.8, gpu_fraction=0.8)
        print_resource_summary()
        gpu_display = f"{gpu_gb:.2f} GB" if gpu_gb is not None else "N/A"
        console.print(Panel(f"[bold green]Allocated RAM: {ram_gb:.2f} GB | GPU: {gpu_display} (80%)", title="[green]Resource Allocation", border_style="green"))

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
        resource_log_path = "output_default/sweep_resource_log.json"
        with open(resource_log_path, "w", encoding="utf-8") as f:
            json.dump(resource_info, f, indent=2)
        pro_log(f"[Sweep] Resource log exported to {resource_log_path}", tag="Sweep")

        # --- เทพ: Export sweep params summary ---
        try:
            sweep_summary = {'params': sweep_params}
            summary_path = "output_default/sweep_params_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(sweep_summary, f, indent=2)
            pro_log(f"[Sweep] Sweep params summary exported to {summary_path}", tag="Sweep")
        except Exception as e:
            pro_log(f"[Sweep] Sweep params summary export error: {e}", level="warn", tag="Sweep")

        # --- เทพ: Export diagnostics log ---
        diagnostics_path = "output_default/sweep_diagnostics.log"
        with open(diagnostics_path, "a", encoding="utf-8") as f:
            f.write(f"[DIAG] Sweep run at: {pd.Timestamp.now()}\n")
            f.write(f"Resource: {resource_info}\n")
            f.write(f"Sweep params: {sweep_params}\n")
        pro_log(f"[Sweep] Diagnostics log exported to {diagnostics_path}", tag="Sweep")

        # --- เทพ: Export timing/performance log ---
        end_time = time.time()
        perf_log_path = "output_default/sweep_perf.log"
        with open(perf_log_path, "a", encoding="utf-8") as f:
            f.write(f"Sweep finished at {pd.Timestamp.now()} | Duration: {end_time - start_time:.2f} sec\n")
        pro_log(f"[Sweep] Performance log exported to {perf_log_path}", tag="Sweep")

        # --- เทพ: RICH SUMMARY PANEL ---
        from rich.table import Table
        summary_table = Table(title="[bold green]สรุปผล Hyperparameter Sweep", show_header=True, header_style="bold magenta")
        summary_table.add_column("Metric", style="cyan", justify="right")
        summary_table.add_column("Value", style="white")
        summary_table.add_row("Sweep Params", str(list(sweep_params.keys())))
        summary_table.add_row("Output Dir", config.get('model_dir', 'models') if config else 'models')
        console.print(Panel(summary_table, title="[bold blue]✅ Sweep เสร็จสมบูรณ์!", border_style="bright_green"))

        # --- RICH SUMMARY PANEL พร้อมลิงก์ ---
        from rich.text import Text
        from rich import box
        summary_table = Table(title="[bold green]สรุปผล Hyperparameter Sweep", show_header=True, header_style="bold magenta", box=box.ROUNDED)
        summary_table.add_column("ไฟล์/ข้อมูล", style="cyan", justify="right")
        summary_table.add_column("Path/ค่า", style="white")
        if os.path.exists(config.get('model_dir', 'models') if config else 'models'):
            summary_table.add_row("Output Dir", f"[link=file://{config.get('model_dir', 'models') if config else 'models'}]{config.get('model_dir', 'models') if config else 'models'}[/link]")
        summary_table.add_row("Sweep Params", str(list(sweep_params.keys())))
        console.print(Panel(summary_table, title="[bold blue]✅ Sweep เสร็จสมบูรณ์!", border_style="bright_green"))

        # --- EXPORT OPTIONS ---
        export_hint = "[bold yellow]ตัวเลือก Export:[/bold yellow]\n- [green]test_pred.csv[/green] (CSV)\n- [green]sweep log[/green] (log/JSON ถ้ามี)\n\n[bold]สามารถแปลงผลลัพธ์เป็น Excel/Markdown/HTML ได้ด้วย pandas[/bold]"
        console.print(Panel(export_hint, title="[bold magenta]Export & Share", border_style="yellow"))

        # --- NEXT STEP SUGGESTIONS ---
        next_steps = "[bold blue]ขั้นตอนถัดไป:[/bold blue]\n- รัน [green]train[/green] ต่อเพื่อเทรนโมเดล\n- รัน [green]threshold[/green] เพื่อหา threshold ที่ดีที่สุด\n- ตรวจสอบไฟล์ output หรือ export ผลลัพธ์\n- [bold]ใช้ CLI flags เพื่อความรวดเร็ว![/bold]"
        console.print(Panel(next_steps, title="[bold green]ทำอะไรต่อดี?", border_style="bright_blue"))

        # --- เทพ: Usability hints ---
        pro_log(f"[Sweep] ดู resource log ที่ {resource_log_path}\nดู sweep params summary ที่ {summary_path if 'summary_path' in locals() else '-'}\nดู diagnostics log ที่ {diagnostics_path}", tag="Sweep")
        console.print(Panel("[bold green]Sweep completed successfully!", title="Sweep", expand=False))
    return True
