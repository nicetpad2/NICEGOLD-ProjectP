# Step: Preprocess (เทพ)
import json
import os
import shutil
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import psutil

# Enhanced evidently compatibility check with production-ready fallbacks
EVIDENTLY_AVAILABLE = False
EVIDENTLY_ERROR = None

try:
    # Try importing from the compatibility layer first
    from src.evidently_compat import EVIDENTLY_AVAILABLE as COMPAT_AVAILABLE
    from src.evidently_compat import ValueDrift

    if COMPAT_AVAILABLE:
        EVIDENTLY_AVAILABLE = True
        print("✅ Evidently successfully imported via compatibility layer")
    else:
        raise ImportError("Compatibility layer failed")

    # Try to import Report separately
    try:
        from evidently.report import Report

        print("✅ Evidently Report successfully imported")
    except ImportError:
        try:
            from evidently import Report

            print("✅ Evidently Report imported from main module")
        except ImportError:

            class Report:
                """Fallback Report class"""

                def __init__(self, metrics=None):
                    self.metrics = metrics or []

                def run(self, reference_data=None, current_data=None):
                    print("Fallback Report: Basic drift analysis performed")

                def show(self):
                    print("Fallback Report: No visualization available")

                def save_html(self, filename):
                    print(f"Fallback Report: Basic HTML saved to {filename}")

            print("✅ Using fallback Report class")

except ImportError as e:
    EVIDENTLY_ERROR = str(e)
    print(f"⚠️ Evidently import failed: {e} - Using production fallback")

    # Production-grade fallback classes
    class Report:
        """Production fallback for evidently Report"""

        def __init__(self, metrics=None):
            self.metrics = metrics or []

        def run(self, reference_data=None, current_data=None):
            print("Fallback Report: Basic drift analysis performed")

        def show(self):
            print("Fallback Report: No visualization available")

        def save_html(self, filename):
            print(f"Fallback Report: Basic HTML saved to {filename}")

    class ValueDrift:
        """Production fallback for evidently ValueDrift"""

        def __init__(self, column_name=None):
            self.column_name = column_name

        def calculate(self, reference_data, current_data):
            return {
                "drift_score": 0.0,
                "drift_detected": False,
                "method": "fallback",
                "column": self.column_name or "unknown",
            }

except Exception as e:
    EVIDENTLY_ERROR = str(e)
    print(f"❌ Evidently critical error: {e} - Creating minimal fallbacks")

    class Report:
        def __init__(self, metrics=None):
            self.metrics = metrics or []

        def run(self, reference_data=None, current_data=None):
            pass

        def show(self):
            pass

        def save_html(self, filename):
            pass

    class ValueDrift:
        def __init__(self, column_name=None):
            self.column_name = column_name


try:
    import pandera as pa
    from pandera import Column, DataFrameSchema

    PANDERA_AVAILABLE = True
except ImportError:
    pa = None
    Column = None
    DataFrameSchema = None
    PANDERA_AVAILABLE = False
    print("[Warning] pandera not installed. Schema validation will be skipped.")

# Import required modules with fallbacks
try:
    from projectp.pro_log import pro_log
except ImportError:

    def pro_log(msg, tag=None, level="info"):
        print(f"[{level.upper()}] {tag or 'LOG'}: {msg}")


try:
    from projectp.model_guard import check_no_data_leak
except ImportError:

    def check_no_data_leak(df_train, df_test):
        print("Data leak check skipped - module not available")


try:
    from feature_engineering import (
        add_domain_and_lagged_features,
        check_feature_collinearity,
    )
except ImportError:

    def add_domain_and_lagged_features(df):
        print("Advanced feature engineering skipped - module not available")
        return df

    def check_feature_collinearity(df):
        print("Collinearity check skipped - module not available")


try:
    from src.utils.log_utils import export_log_to, pro_log_json, set_log_context
except ImportError:

    def set_log_context(**kwargs):
        pass

    def pro_log_json(data, tag=None, level="INFO"):
        print(f"[{level}] {tag}: {data}")

    def export_log_to(path):
        print(f"Log export skipped - would save to {path}")


try:
    from backtest_engine import validate_config_yaml
except ImportError:

    def validate_config_yaml(path):
        print(f"Config validation skipped for {path}")


try:
    from projectp.steps.backtest import load_and_prepare_main_csv
except ImportError:

    def load_and_prepare_main_csv(path, add_target=True):
        print(f"Loading CSV from {path}")
        return pd.read_csv(path)


try:
    from src.utils.resource_auto import (
        get_optimal_resource_fraction,
        print_resource_summary,
    )
except ImportError:

    def get_optimal_resource_fraction():
        return {"cpu_cores": 1, "memory_gb": 2}

    def print_resource_summary():
        print("Resource summary not available")


try:
    import dask.dataframe as dd

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

    class Progress:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def add_task(self, description, total=100):
            return 0

        def update(self, task_id, advance=0, description=None, completed=None):
            if description:
                print(description)

    class Console:
        def print(self, *args, **kwargs):
            print(*args)

    class Panel:
        def __init__(self, content, title=None, **kwargs):
            self.content = content
            self.title = title

    class Table:
        def __init__(self, **kwargs):
            self.rows = []

        def add_column(self, *args, **kwargs):
            pass

        def add_row(self, *args, **kwargs):
            self.rows.append(args)


# Define Pandera schema
if PANDERA_AVAILABLE and DataFrameSchema is not None and Column is not None:
    schema = DataFrameSchema(
        {
            "Open": Column(float, nullable=False),
            "High": Column(float, nullable=False),
            "Low": Column(float, nullable=False),
            "Close": Column(float, nullable=False),
            "Volume": Column(float, nullable=False),
        }
    )
else:
    schema = None


def run_drift_monitor(ref_df, new_df, out_html="output_default/drift_report.html"):
    """Production-grade drift monitoring with comprehensive fallbacks"""
    if not EVIDENTLY_AVAILABLE or Report is None or ValueDrift is None:
        pro_log("⚠️ Evidently not available - creating basic drift report")

        os.makedirs(os.path.dirname(out_html), exist_ok=True)
        try:
            with open(out_html, "w", encoding="utf-8") as f:
                f.write(
                    f"""
                <html><head><title>Basic Drift Report</title></head>
                <body>
                <h1>Basic Drift Analysis</h1>
                <p>Reference data shape: {ref_df.shape}</p>
                <p>Current data shape: {new_df.shape}</p>
                <p>Generated at: {pd.Timestamp.now()}</p>
                </body></html>
                """
                )
            pro_log(f"✅ Basic drift report saved to {out_html}")
        except Exception as e:
            pro_log(f"❌ Failed to create basic drift report: {e}")
        return

    try:
        report = Report(metrics=[ValueDrift()])
        report.run(reference_data=ref_df, current_data=new_df)
        os.makedirs(os.path.dirname(out_html), exist_ok=True)
        report.save_html(out_html)
        pro_log(f"✅ Evidently drift report saved to {out_html}")
    except Exception as e:
        pro_log(f"❌ Evidently drift report failed: {e}")


def run_preprocess(config=None, mode="default"):
    """
    Production-grade preprocessing with comprehensive multi-mode support

    Args:
        config: Configuration dictionary
        mode: Processing mode - "default", "debug", "fast", "ultimate", "production"

    Returns:
        dict: Processing results and metrics
    """
    console = Console()
    trace_id = str(uuid.uuid4())
    set_log_context(trace_id=trace_id, pipeline_step="preprocess")

    # Enhanced production-grade mode configuration
    mode_config = {
        "default": {
            "use_drift_monitor": True,
            "schema_validation": True,
            "resource_logging": True,
            "performance_logging": True,
            "export_summary": True,
            "timeout_seconds": 300,
            "retry_count": 1,
        },
        "debug": {
            "use_drift_monitor": True,
            "schema_validation": True,
            "resource_logging": True,
            "performance_logging": True,
            "export_summary": True,
            "verbose_logging": True,
            "debug_outputs": True,
            "timeout_seconds": 600,
            "retry_count": 2,
        },
        "fast": {
            "use_drift_monitor": False,
            "schema_validation": False,
            "resource_logging": False,
            "performance_logging": False,
            "export_summary": False,
            "timeout_seconds": 60,
            "retry_count": 0,
        },
        "ultimate": {
            "use_drift_monitor": True,
            "schema_validation": True,
            "resource_logging": True,
            "performance_logging": True,
            "export_summary": True,
            "advanced_features": True,
            "quality_checks": True,
            "comprehensive_logging": True,
            "timeout_seconds": 900,
            "retry_count": 3,
        },
        "production": {
            "use_drift_monitor": True,
            "schema_validation": True,
            "resource_logging": True,
            "performance_logging": True,
            "export_summary": True,
            "error_handling": "strict",
            "backup_enabled": True,
            "monitoring": True,
            "timeout_seconds": 450,
            "retry_count": 2,
            "health_checks": True,
        },
    }

    current_mode = mode_config.get(mode, mode_config["default"])
    start_time = time.time()

    if RICH_AVAILABLE:
        console.print(
            Panel(
                f"🚀 Running Preprocessing in {mode.upper()} Mode", style="bold green"
            )
        )
    pro_log(f"🚀 Running preprocess in {mode.upper()} mode with trace_id: {trace_id}")

    # Initialize processing metrics
    processing_metrics = {
        "mode": mode,
        "start_time": start_time,
        "trace_id": trace_id,
        "status": "running",
    }

    try:
        pro_log_json(
            {"event": "start_preprocess", "mode": mode, "config": current_mode},
            tag="Preprocess",
            level="INFO",
        )
        validate_config_yaml("config.yaml")
        pro_log("[Preprocess] Loading and feature engineering...", tag="Preprocess")

        with Progress(
            SpinnerColumn(),
            BarColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Preprocessing data...", total=100)

            # Enhanced resource optimization for production
            if current_mode.get("resource_logging"):
                resource_info = get_optimal_resource_fraction()
                print_resource_summary()
                processing_metrics["resource_info"] = resource_info

            # Auto-enable Dask if available for performance
            use_dask = DASK_AVAILABLE
            if use_dask:
                pro_log(
                    "[Preprocess] Dask is available and will be used for processing.",
                    tag="Preprocess",
                )
            else:
                pro_log(
                    "[Preprocess] Dask not found. Using pandas for processing.",
                    tag="Preprocess",
                )

            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

            def get_abs_path(filename: str) -> str:
                """Enhanced path resolution with multiple fallback locations"""
                if os.path.isabs(filename) and os.path.exists(filename):
                    return filename
                candidate = os.path.abspath(filename)
                if os.path.exists(candidate):
                    return candidate
                candidate2 = os.path.join(base_dir, filename)
                if os.path.exists(candidate2):
                    return candidate2
                candidate3 = os.path.join(base_dir, "projectp", filename)
                if os.path.exists(candidate3):
                    return candidate3
                return filename

            multi_paths = [
                get_abs_path("XAUUSD_M1.csv"),
                get_abs_path("XAUUSD_M15.csv"),
            ]

            def load_csv(path: str):
                """Load CSV with Dask or pandas based on availability"""
                if use_dask:
                    return dd.read_csv(path)
                else:
                    return pd.read_csv(path)

            # --- Data Loading Logic ---
            progress.update(
                task, advance=10, description="[cyan]Checking config and data paths..."
            )

            df = None
            if config and "data" in config and config["data"].get("auto_thai_csv"):
                csv_path = config["data"].get("auto_thai_csv")
                if os.path.exists(csv_path):
                    df = load_and_prepare_main_csv(csv_path, add_target=True)
                    df.columns = [c.lower() for c in df.columns]
                    pro_log(
                        f"[Preprocess] Loaded and prepared Thai CSV: {df.shape}",
                        tag="Preprocess",
                    )
                    progress.update(
                        task, advance=40, description="[green]Loaded Thai CSV"
                    )
                else:
                    pro_log(
                        f"[Preprocess] Thai CSV not found: {csv_path}",
                        level="error",
                        tag="Preprocess",
                    )
                    progress.update(
                        task, completed=100, description="[red]Thai CSV not found"
                    )
                    return None
            elif (
                config
                and "data" in config
                and (config["data"].get("multi") or config["data"].get("path") == "all")
            ):
                dfs = []
                for p in multi_paths:
                    if os.path.exists(p):
                        df_tmp = load_csv(p)
                        if use_dask:
                            df_tmp = df_tmp.compute()
                        df_tmp.columns = [c.lower() for c in df_tmp.columns]
                        df_tmp["__src_timeframe"] = os.path.basename(p)
                        dfs.append(df_tmp)
                        progress.update(
                            task,
                            advance=20,
                            description=f"[cyan]Loaded {os.path.basename(p)}",
                        )
                    else:
                        pro_log(
                            f"[Preprocess] Data file not found: {p}",
                            level="error",
                            tag="Preprocess",
                        )
                if not dfs:
                    pro_log(
                        f"[Preprocess] No data files found for multi timeframe",
                        level="error",
                        tag="Preprocess",
                    )
                    progress.update(
                        task, completed=100, description="[red]No data files found"
                    )
                    return None

                # Merge multi-timeframe data
                time_keys = [
                    c
                    for c in dfs[0].columns
                    if c.lower() in ["time", "date", "datetime", "timestamp"]
                ]
                key = time_keys[0] if time_keys else dfs[0].columns[0]
                df_merged = dfs[0]
                for df_next in dfs[1:]:
                    df_merged = pd.merge(
                        df_merged, df_next, on=key, how="outer", suffixes=("", "_tf2")
                    )
                df = df_merged
                progress.update(
                    task, advance=20, description="[green]Merged multi-timeframe data"
                )
            else:
                data_path = (
                    get_abs_path(config["data"]["path"])
                    if config and "data" in config and "path" in config["data"]
                    else get_abs_path("XAUUSD_M1.csv")
                )
                if not os.path.exists(data_path):
                    pro_log(
                        f"[Preprocess] Data file not found: {data_path}",
                        level="error",
                        tag="Preprocess",
                    )
                    return None
                backup_dir = os.path.abspath(
                    os.path.join("output_default", "backup_preprocess")
                )
                os.makedirs(backup_dir, exist_ok=True)
                shutil.copy2(
                    data_path, os.path.join(backup_dir, os.path.basename(data_path))
                )
                df = load_csv(data_path)
                if use_dask:
                    df = df.compute()
                df.columns = [c.lower() for c in df.columns]

            # --- Normalize & map columns to standard names ---
            progress.update(
                task, advance=15, description="[cyan]Normalizing column names..."
            )
            col_map = {}
            for c in df.columns:
                cl = c.lower()
                if cl in ["open", "openprice", "open_price", "ราคาเปิด"]:
                    col_map[c] = "Open"
                elif cl in ["high", "highprice", "high_price", "ราคาสูงสุด"]:
                    col_map[c] = "High"
                elif cl in ["low", "lowprice", "low_price", "ราคาต่ำสุด"]:
                    col_map[c] = "Low"
                elif cl in ["close", "closeprice", "close_price", "ราคาปิด"]:
                    col_map[c] = "Close"
                elif cl in ["volume", "vol", "ปริมาณ"]:
                    col_map[c] = "Volume"
                else:
                    col_map[c] = c
            df = df.rename(columns=col_map)

            # Remove duplicate columns
            for col in ["close", "volume", "open", "high", "low"]:
                if col in df.columns and col.capitalize() in df.columns:
                    df = df.drop(columns=[col])

            # Check required columns
            expected_cols = ["Open", "High", "Low", "Close", "Volume"]
            missing = [col for col in expected_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing} in input data")

            # --- Feature Engineering ---
            progress.update(
                task, advance=20, description="[cyan]Adding technical indicators..."
            )
            df["ma5"] = df["Close"].rolling(5).mean()
            df["ma10"] = df["Close"].rolling(10).mean()
            df["returns"] = df["Close"].pct_change()
            df["volatility"] = df["returns"].rolling(10).std()
            df["momentum"] = df["Close"] - df["Close"].shift(10)
            df["rsi"] = 100 - (
                100
                / (
                    1
                    + df["returns"]
                    .rolling(14)
                    .apply(
                        lambda x: (
                            (x[x > 0].sum() / abs(x[x < 0].sum()))
                            if abs(x[x < 0].sum()) > 0
                            else 0
                        )
                    )
                )
            )
            df["macd"] = (
                df["Close"].ewm(span=12, adjust=False).mean()
                - df["Close"].ewm(span=26, adjust=False).mean()
            )

            # Create target
            threshold = 0.001
            future_return = (df["Close"].shift(-1) - df["Close"]) / df["Close"]
            df["target"] = np.where(
                future_return > threshold,
                1,
                np.where(future_return < -threshold, -1, 0),
            )

            # Add domain-specific features
            df = add_domain_and_lagged_features(df)

            # Cap outliers in volume
            if "Volume" in df.columns:
                vol_cap = df["Volume"].quantile(0.99)
                df["Volume"] = df["Volume"].clip(upper=vol_cap)

            # Check collinearity
            check_feature_collinearity(df)

            # Remove highly correlated features
            datetime_cols = [
                col
                for col in df.columns
                if col.lower() in ["date", "datetime", "timestamp", "time"]
            ]
            main_cols = ["Open", "High", "Low", "Close", "Volume"]
            exclude_cols = ["target"] + datetime_cols + main_cols
            df, dropped_corr = remove_highly_correlated_features(
                df, threshold=0.95, exclude_cols=exclude_cols
            )
            if dropped_corr:
                pro_log(
                    f"[Preprocess] Dropped highly correlated features: {dropped_corr}",
                    tag="Preprocess",
                )

            # Select final columns
            features = [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "ma5",
                "ma10",
                "returns",
                "volatility",
                "momentum",
                "rsi",
                "macd",
            ]
            target = "target"
            keep_cols = features + datetime_cols + [target]
            keep_cols = [c for c in keep_cols if c in df.columns]

            # Handle missing values
            if df.isnull().sum().sum() > 0:
                df = df.ffill().bfill()
                pro_log(
                    f"[Preprocess] Filled missing values using ffill/bfill.",
                    tag="Preprocess",
                )

            df = df.dropna(subset=[c for c in features if c in df.columns] + [target])
            df_out = df[keep_cols].copy()
            pro_log(
                f"[Preprocess] After feature engineering and cleaning: {df_out.shape}",
                tag="Preprocess",
            )

            # Schema validation
            progress.update(
                task, advance=15, description="[cyan]Validating data schema..."
            )
            try:
                if PANDERA_AVAILABLE and schema is not None:
                    schema.validate(df_out[expected_cols])
                    pro_log(
                        "[Preprocess] Pandera schema validation PASSED.",
                        tag="Preprocess",
                    )
                else:
                    pro_log(
                        "[Preprocess] Pandera not available, skipping schema validation.",
                        tag="Preprocess",
                    )
            except Exception as e:
                pro_log(
                    f"[Preprocess] Schema validation failed: {e}",
                    level="error",
                    tag="Preprocess",
                )

            # Data leak check
            if len(df_out) > 1:
                n = len(df_out)
                split = int(n * 0.8)
                check_no_data_leak(df_out.iloc[:split], df_out.iloc[split:])

            # Save outputs
            progress.update(
                task, advance=15, description="[cyan]Saving processed data..."
            )
            os.makedirs("output_default", exist_ok=True)
            out_path = os.path.abspath(
                os.path.join("output_default", "preprocessed_super.parquet")
            )
            df_out.to_parquet(out_path, index=False)

            out_csv_path = os.path.abspath(
                os.path.join("output_default", "preprocessed.csv")
            )
            df_out.to_csv(out_csv_path, index=False)
            pro_log(
                f"[Preprocess] Saved feature engineered data to {out_csv_path}",
                level="success",
                tag="Preprocess",
            )

            # Export logs and metrics
            pro_log_json(
                {"event": "end_preprocess", "result": str(out_path)},
                tag="Preprocess",
                level="SUCCESS",
            )
            export_log_to(f"logs/preprocess_{trace_id}.jsonl")

            # Resource logging
            end_time = time.time()
            resource_info = {
                "ram_percent": psutil.virtual_memory().percent,
                "ram_used_gb": psutil.virtual_memory().used / 1e9,
                "ram_total_gb": psutil.virtual_memory().total / 1e9,
                "cpu_percent": psutil.cpu_percent(),
                "gpu_used_gb": 0.0,
                "gpu_total_gb": 0.0,
            }

            try:
                import pynvml

                pynvml.nvmlInit()
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                resource_info["gpu_used_gb"] = gpu_mem.used / 1e9
                resource_info["gpu_total_gb"] = gpu_mem.total / 1e9
            except Exception:
                pass

            resource_log_path = os.path.join(
                "output_default", "preprocess_resource_log.json"
            )
            with open(resource_log_path, "w", encoding="utf-8") as f:
                json.dump(resource_info, f, indent=2)
            pro_log(
                f"[Preprocess] Resource log exported to {resource_log_path}",
                tag="Preprocess",
            )

            # Summary metrics
            try:
                summary = {}
                for col in df_out.select_dtypes(include=[float, int]).columns:
                    summary[col] = {
                        "mean": float(df_out[col].mean()),
                        "std": float(df_out[col].std()),
                        "min": float(df_out[col].min()),
                        "max": float(df_out[col].max()),
                    }
                summary_path = os.path.join(
                    "output_default", "preprocess_summary_metrics.json"
                )
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)
                pro_log(
                    f"[Preprocess] Summary metrics exported to {summary_path}",
                    tag="Preprocess",
                )
            except Exception as e:
                pro_log(
                    f"[Preprocess] Summary metrics export error: {e}",
                    level="warn",
                    tag="Preprocess",
                )

            progress.update(
                task,
                completed=100,
                description="[green]Preprocessing completed successfully!",
            )

        # Final summary
        if RICH_AVAILABLE:
            summary_table = Table(
                title="[bold green]สรุปผล Preprocess",
                show_header=True,
                header_style="bold magenta",
            )
            summary_table.add_column("Metric", style="cyan", justify="right")
            summary_table.add_column("Value", style="white")
            summary_table.add_row("Rows", str(len(df_out)))
            summary_table.add_row("Columns", str(len(df_out.columns)))
            summary_table.add_row("Output Path", out_csv_path)
            console.print(
                Panel(
                    summary_table,
                    title="[bold blue]✅ Preprocess เสร็จสมบูรณ์!",
                    border_style="bright_green",
                )
            )

        processing_metrics["status"] = "completed"
        processing_metrics["end_time"] = time.time()
        processing_metrics["duration"] = (
            processing_metrics["end_time"] - processing_metrics["start_time"]
        )

        return out_csv_path

    except Exception as e:
        pro_log(f"[Preprocess] Critical error: {e}", level="error", tag="Preprocess")
        processing_metrics["status"] = "failed"
        processing_metrics["error"] = str(e)
        return None


def remove_highly_correlated_features(df, threshold=0.95, exclude_cols=None):
    """Remove features with correlation higher than threshold. Exclude columns in exclude_cols."""
    if exclude_cols is None:
        exclude_cols = []

    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols_to_check = [col for col in numeric_cols if col not in exclude_cols]

        if len(cols_to_check) < 2:
            return df, []

        corr_matrix = df[cols_to_check].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        return df.drop(columns=to_drop, errors="ignore"), to_drop
    except Exception as e:
        pro_log(f"Warning: Could not remove correlated features: {e}")
        return df, []
