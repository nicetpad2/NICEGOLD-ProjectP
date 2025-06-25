

# Enhanced evidently compatibility check with production - ready fallbacks
# Enhanced Production - Ready Preprocess Module
# Step: Preprocess (เทพ) with comprehensive multi - mode support
from backtest_engine import validate_config_yaml
    from evidently.metrics import ValueDrift
    from evidently.report import Report
from feature_engineering import add_domain_and_lagged_features, check_feature_collinearity
    from pandera import Column, DataFrameSchema
from projectp.model_guard import check_no_data_leak
from projectp.pro_log import pro_log
from projectp.steps.backtest import load_and_prepare_main_csv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from src.utils.log_utils import set_log_context, pro_log_json, export_log_to
from src.utils.resource_auto import get_optimal_resource_fraction, print_resource_summary
from typing import Dict, Optional, Any, List
    import dask.dataframe as dd
                import json
import numpy as np
import os
import pandas as pd
    import pandera as pa
import shutil
    import sys
import time
import traceback
import uuid
EVIDENTLY_AVAILABLE = False
EVIDENTLY_ERROR = None

try:
    # Production - grade evidently import with version compatibility
    python_version = sys.version_info

    if python_version >= (3, 11):
        pro_log("Python 3.11+ detected, checking evidently compatibility...")

    EVIDENTLY_AVAILABLE = True
    pro_log("✅ Evidently successfully imported")

except ImportError as e:
    EVIDENTLY_ERROR = str(e)
    pro_log(f"⚠️ Evidently import failed: {e} - Using production fallback")

    # Production - grade fallback classes
    class Report:
        """Production fallback for evidently Report"""
        def __init__(self, metrics = None):
            self.metrics = metrics or []
            self._fallback_mode = True

        def run(self, reference_data = None, current_data = None):
            pro_log("Running drift analysis fallback mode")
            return self

        def show(self):
            return "<Evidently fallback mode - drift monitoring disabled>"

        def save_html(self, filename):
            os.makedirs(os.path.dirname(filename), exist_ok = True)
            with open(filename, 'w', encoding = 'utf - 8') as f:
                f.write("""
<!DOCTYPE html>
<html>
<head><title>Drift Report - Fallback Mode</title></head>
<body>
<h1>🔧 Evidently Fallback Mode</h1>
<p>Drift analysis is running in fallback mode.</p>
<p>For full drift monitoring, install evidently: pip install evidently =  = 0.4.30</p>
</body>
</html>
                """)
            pro_log(f"Drift report fallback saved to {filename}")

    class ValueDrift:
        """Production fallback for evidently ValueDrift"""
        def __init__(self, column_name = None):
            self.column_name = column_name
            self._fallback_mode = True

except Exception as e:
    EVIDENTLY_ERROR = str(e)
    pro_log(f"❌ Evidently critical error: {e} - Creating minimal fallbacks")

    # Minimal fallback for critical errors
    class Report:
        def __init__(self, metrics = None):
            self._fallback_mode = True
        def run(self, reference_data = None, current_data = None):
            return self
        def show(self):
            return "Evidently minimal fallback"
        def save_html(self, filename):
            os.makedirs(os.path.dirname(filename), exist_ok = True)
            with open(filename, 'w', encoding = 'utf - 8') as f:
                f.write("<html><body><h1>Drift Analysis Disabled</h1></body></html>")

    class ValueDrift:
        def __init__(self, column_name = None):
            self.column_name = column_name

    pro_log("⚠️ Evidently not available - drift monitoring disabled")

# Additional dependencies with fallbacks
try:
    PANDERA_AVAILABLE = True
except ImportError:
    pa = None
    Column = None
    DataFrameSchema = None
    PANDERA_AVAILABLE = False
    pro_log("[Warning] pandera not installed. Schema validation will be skipped.")

try:
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

# Import project modules

# Rich console imports

# Schema definition
if PANDERA_AVAILABLE and DataFrameSchema is not None and Column is not None:
    schema = DataFrameSchema({
        "Open": Column(float, nullable = False), 
        "High": Column(float, nullable = False), 
        "Low": Column(float, nullable = False), 
        "Close": Column(float, nullable = False), 
        "Volume": Column(float, nullable = False, checks = pa.Check.ge(0)), 
    })
else:
    schema = None

def run_drift_monitor(ref_df: pd.DataFrame, new_df: pd.DataFrame, out_html: str = "output_default/drift_report.html") -> None:
    """Production - grade drift monitoring with comprehensive fallbacks"""
    if not EVIDENTLY_AVAILABLE or Report is None or ValueDrift is None:
        pro_log("⚠️ Evidently not available - creating basic drift report")

        # Create basic drift analysis
        os.makedirs(os.path.dirname(out_html), exist_ok = True)
        try:
            # Basic statistical comparison
            drift_summary = []

            for col in ref_df.select_dtypes(include = [float, int]).columns:
                if col in new_df.columns:
                    ref_mean = ref_df[col].mean()
                    new_mean = new_df[col].mean()
                    drift_pct = abs((new_mean - ref_mean) / ref_mean * 100) if ref_mean != 0 else 0
                    drift_summary.append({
                        'column': col, 
                        'ref_mean': ref_mean, 
                        'new_mean': new_mean, 
                        'drift_percent': drift_pct, 
                        'status': 'WARNING' if drift_pct > 10 else 'OK'
                    })

            # Generate HTML report
            with open(out_html, 'w', encoding = 'utf - 8') as f:
                f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>Basic Drift Report</title>
    <style>
        body {{ font - family: Arial, sans - serif; margin: 20px; }}
        .warning {{ color: orange; }}
        .ok {{ color: green; }}
        table {{ border - collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text - align: left; }}
    </style>
</head>
<body>
    <h1>📊 Basic Drift Analysis Report</h1>
    <p>Generated: {pd.Timestamp.now()}</p>
    <table>
        <tr><th>Column</th><th>Ref Mean</th><th>New Mean</th><th>Drift %</th><th>Status</th></tr>
                """)

                for item in drift_summary:
                    status_class = 'warning' if item['status'] == 'WARNING' else 'ok'
                    f.write(f"""
        <tr>
            <td>{item['column']}</td>
            <td>{item['ref_mean']:.4f}</td>
            <td>{item['new_mean']:.4f}</td>
            <td>{item['drift_percent']:.2f}%</td>
            <td class = "{status_class}">{item['status']}</td>
        </tr>
                    """)

                f.write("""
    </table>
    <p><em>Note: Install evidently for advanced drift analysis</em></p>
</body>
</html>
                """)

            pro_log(f"✅ Basic drift report saved to {out_html}")

        except Exception as e:
            pro_log(f"❌ Error creating basic drift report: {e}")
        return

    # Full evidently drift monitoring
    try:
        report = Report(metrics = [ValueDrift()])
        report.run(reference_data = ref_df, current_data = new_df)
        os.makedirs(os.path.dirname(out_html), exist_ok = True)
        report.save_html(out_html)
        pro_log(f"✅ Evidently drift report saved to {out_html}")
    except Exception as e:
        pro_log(f"❌ Evidently drift monitoring failed: {e}")

def run_preprocess(config: Optional[Dict[str, Any]] = None, mode: str = "default") -> Optional[Dict[str, Any]]:
    """
    Production - grade preprocessing with comprehensive multi - mode support

    Args:
        config: Configuration dictionary
        mode: Processing mode - "default", "debug", "fast", "ultimate", "production"

    Returns:
        dict: Processing results and metrics, or None if failed
    """
    console = Console()
    trace_id = str(uuid.uuid4())
    set_log_context(trace_id = trace_id, pipeline_step = "preprocess")

    # Enhanced production - grade mode configuration
    mode_config = {
        "default": {
            "use_drift_monitor": True, 
            "schema_validation": True, 
            "resource_logging": True, 
            "performance_logging": True, 
            "export_summary": True, 
            "timeout_seconds": 300, 
            "retry_count": 1
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
            "retry_count": 2
        }, 
        "fast": {
            "use_drift_monitor": False, 
            "schema_validation": False, 
            "resource_logging": False, 
            "performance_logging": False, 
            "export_summary": False, 
            "timeout_seconds": 60, 
            "retry_count": 0
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
            "retry_count": 3
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
            "health_checks": True
        }
    }

    current_mode = mode_config.get(mode, mode_config["default"])
    start_time = time.time()

    console.print(Panel(f"🚀 Running Preprocessing in {mode.upper()} Mode", style = "bold green"))
    pro_log(f"🚀 Running preprocess in {mode.upper()} mode with trace_id: {trace_id}")

    if current_mode.get("verbose_logging") or current_mode.get("debug_outputs"):
        pro_log(f"Mode configuration: {current_mode}")

    # Production health checks
    if current_mode.get("health_checks"):
        pro_log("🔍 Running pre - processing health checks...")
        health_status = {}

        # Check Python version compatibility
        health_status["python_version"] = f"{sys.version_info.major}.{sys.version_info.minor}"

        # Check critical dependencies
        health_status["evidently_available"] = EVIDENTLY_AVAILABLE
        health_status["pandera_available"] = PANDERA_AVAILABLE
        health_status["dask_available"] = DASK_AVAILABLE

        # Check disk space (simplified)
        try:
            free_space = shutil.disk_usage('.').free / (1024**3)  # GB
            health_status["free_disk_gb"] = round(free_space, 2)
            if free_space < 1:
                pro_log("⚠️ Warning: Low disk space detected")
        except Exception as e:
            health_status["disk_check_error"] = str(e)

        pro_log(f"Health check results: {health_status}")

    # Initialize processing metrics
    processing_metrics = {
        "mode": mode, 
        "start_time": start_time, 
        "trace_id": trace_id, 
        "status": "running"
    }

    try:
        pro_log_json({"event": "start_preprocess", "mode": mode, "config": current_mode}, tag = "Preprocess", level = "INFO")
        validate_config_yaml("config.yaml")
        pro_log("[Preprocess] Loading and feature engineering...", tag = "Preprocess")

        with Progress(SpinnerColumn(), BarColumn(), TextColumn("[progress.description]{task.description}"), TimeElapsedColumn(), console = console) as progress:
            task = progress.add_task("[cyan]Preprocessing data...", total = 100)

            # Enhanced resource optimization for production
            if current_mode.get("resource_logging"):
                resource_info = get_optimal_resource_fraction()
                print_resource_summary()
                processing_metrics["resource_info"] = resource_info

            # Auto - enable Dask if available for performance
            use_dask = DASK_AVAILABLE
            if use_dask:
                pro_log("[Preprocess] Dask is available and will be used for processing.", tag = "Preprocess")
            else:
                pro_log("[Preprocess] Dask not found. Using pandas for processing.", tag = "Preprocess")

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

            multi_paths = [get_abs_path("XAUUSD_M1.csv"), get_abs_path("XAUUSD_M15.csv")]

            def load_csv(path: str):
                """Load CSV with Dask or pandas based on availability"""
                if use_dask:
                    return dd.read_csv(path)
                else:
                    return pd.read_csv(path)

            # Initialize df variable
            df = None

            # - - - Integration อัตโนมัติ: รองรับ CSV Timestamp พ.ศ. - -  - 
            progress.update(task, advance = 10, description = "[cyan]Checking config and data paths...")

            if config and "data" in config and config["data"].get("auto_thai_csv"):
                csv_path = config["data"].get("auto_thai_csv")
                if os.path.exists(csv_path):
                    df = load_and_prepare_main_csv(csv_path, add_target = True)
                    df.columns = [c.lower() for c in df.columns]  # Normalize columns
                    pro_log(f"[Preprocess] Loaded and prepared Thai CSV: {df.shape}", tag = "Preprocess")
                    progress.update(task, advance = 40, description = "[green]Loaded Thai CSV")
                else:
                    pro_log(f"[Preprocess] Thai CSV not found: {csv_path}", level = "error", tag = "Preprocess")
                    progress.update(task, completed = 100, description = "[red]Thai CSV not found")
                    return None

            elif config and "data" in config and (config["data"].get("multi") or config["data"].get("path") == "all"):
                dfs = []
                for p in multi_paths:
                    if os.path.exists(p):
                        df_tmp = load_csv(p)
                        if use_dask:
                            df_tmp = df_tmp.compute()
                        df_tmp.columns = [c.lower() for c in df_tmp.columns]  # Normalize columns
                        df_tmp['__src_timeframe'] = os.path.basename(p)
                        dfs.append(df_tmp)
                        progress.update(task, advance = 20, description = f"[cyan]Loaded {os.path.basename(p)}")
                    else:
                        pro_log(f"[Preprocess] Data file not found: {p}", level = "error", tag = "Preprocess")

                if not dfs:
                    pro_log(f"[Preprocess] No data files found for multi timeframe", level = "error", tag = "Preprocess")
                    progress.update(task, completed = 100, description = "[red]No data files found")
                    return None

                # Merge multiple timeframes
                time_keys = [c for c in dfs[0].columns if c.lower() in ["time", "date", "datetime", "timestamp"]]
                key = time_keys[0] if time_keys else dfs[0].columns[0]
                df = dfs[0]
                for df_next in dfs[1:]:
                    df = pd.merge(df, df_next, on = key, how = 'outer', suffixes = ("", "_tf2"))
                progress.update(task, advance = 20, description = "[green]Merged multi - timeframe data")

            else:
                # Single file processing
                data_path = get_abs_path(config["data"]["path"]) if config and "data" in config and "path" in config["data"] else get_abs_path("XAUUSD_M1.csv")
                if not os.path.exists(data_path):
                    pro_log(f"[Preprocess] Data file not found: {data_path}", level = "error", tag = "Preprocess")
                    return None

                # Backup data if enabled
                if current_mode.get("backup_enabled"):
                    backup_dir = os.path.abspath(os.path.join("output_default", "backup_preprocess"))
                    os.makedirs(backup_dir, exist_ok = True)
                    shutil.copy2(data_path, os.path.join(backup_dir, os.path.basename(data_path)))
                    pro_log(f"Data backed up to {backup_dir}")

                df = load_csv(data_path)
                if use_dask:
                    df = df.compute()
                df.columns = [c.lower() for c in df.columns]  # Normalize columns
                progress.update(task, advance = 30, description = "[green]Loaded data file")

            if df is None:
                pro_log("[Preprocess] No data loaded", level = "error", tag = "Preprocess")
                return None

            # - - - Normalize & map columns to standard names (เทพ) - -  - 
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

            df = df.rename(columns = col_map)

            # ลบคอลัมน์ตัวเล็กซ้ำซ้อนออก (เทพ)
            for col in ["close", "volume", "open", "high", "low"]:
                if col in df.columns and col.capitalize() in df.columns:
                    df = df.drop(columns = [col])

            # ตรวจสอบว่ามีคอลัมน์หลักครบ
            expected_cols = ["Open", "High", "Low", "Close", "Volume"]
            missing = [col for col in expected_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing} in input data. กรุณาตรวจสอบไฟล์ข้อมูลหรือ mapping column ให้ถูกต้อง")

            progress.update(task, advance = 20, description = "[cyan]Normalizing columns...")

            # Feature engineering (เทพ: เพิ่ม technical indicators)
            df["ma5"] = df["Close"].rolling(5).mean()
            df["ma10"] = df["Close"].rolling(10).mean()
            df["returns"] = df["Close"].pct_change()
            df["volatility"] = df["returns"].rolling(10).std()
            df["momentum"] = df["Close"] - df["Close"].shift(10)
            df["rsi"] = 100 - (100 / (1 + df["returns"].rolling(14).apply(lambda x: (x[x>0].sum() / abs(x[x<0].sum())) if abs(x[x<0].sum())>0 else 0)))
            df["macd"] = df["Close"].ewm(span = 12, adjust = False).mean() - df["Close"].ewm(span = 26, adjust = False).mean()

            progress.update(task, advance = 20, description = "[cyan]Adding technical indicators...")

            # ปรับ logic target: ขึ้น 1, ลง -1, sideway 0 (threshold 0.1%)
            threshold = 0.001
            future_return = (df["Close"].shift( - 1) - df["Close"]) / df["Close"]
            df["target"] = np.where(future_return > threshold, 1, np.where(future_return < -threshold, -1, 0))

            # Add domain - specific and lagged features
            df = add_domain_and_lagged_features(df)

            # Cap outliers in volume (99th percentile)
            if "Volume" in df.columns:
                vol_cap = df["Volume"].quantile(0.99)
                df["Volume"] = df["Volume"].clip(upper = vol_cap)

            # Check feature collinearity
            if current_mode.get("quality_checks"):
                check_feature_collinearity(df)

            progress.update(task, advance = 20, description = "[green]Feature engineering completed")

            # Schema validation
            if current_mode.get("schema_validation") and schema is not None:
                try:
                    schema.validate(df[expected_cols])
                    pro_log("✅ Schema validation passed")
                except Exception as e:
                    pro_log(f"⚠️ Schema validation failed: {e}")

            # Remove rows with NaN values
            initial_rows = len(df)
            df = df.dropna()
            final_rows = len(df)
            if initial_rows != final_rows:
                pro_log(f"Removed {initial_rows - final_rows} rows with NaN values")

            # Check for data leakage
            check_no_data_leak(df)

            # Final processing metrics
            end_time = time.time()
            processing_time = end_time - start_time

            processing_metrics.update({
                "status": "completed", 
                "end_time": end_time, 
                "processing_time_seconds": processing_time, 
                "rows_processed": len(df), 
                "columns_processed": len(df.columns), 
                "features_created": len([col for col in df.columns if col not in expected_cols])
            })

            progress.update(task, completed = 100, description = "[bold green]Preprocessing completed!")

            # Export summary if enabled
            if current_mode.get("export_summary"):
                summary_path = "output_default/preprocess_summary.json"
                os.makedirs(os.path.dirname(summary_path), exist_ok = True)

                summary = {
                    "processing_metrics": processing_metrics, 
                    "data_shape": df.shape, 
                    "columns": list(df.columns), 
                    "target_distribution": df["target"].value_counts().to_dict() if "target" in df.columns else None
                }

                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent = 2, default = str)
                pro_log(f"✅ Processing summary exported to {summary_path}")

            # Save processed data
            output_path = "output_default/preprocessed_data.csv"
            os.makedirs(os.path.dirname(output_path), exist_ok = True)
            df.to_csv(output_path, index = False)
            pro_log(f"✅ Preprocessed data saved to {output_path}")

            console.print(Panel(f"✅ Preprocessing completed successfully in {processing_time:.2f} seconds", style = "bold green"))

            return {
                "data": df, 
                "metrics": processing_metrics, 
                "status": "success"
            }

    except Exception as e:
        error_time = time.time()
        processing_metrics.update({
            "status": "failed", 
            "error_time": error_time, 
            "error": str(e), 
            "traceback": traceback.format_exc()
        })

        console.print(Panel(f"❌ Preprocessing failed: {str(e)}", style = "bold red"))
        pro_log(f"❌ Preprocessing failed: {str(e)}", level = "error")

        if current_mode.get("verbose_logging") or current_mode.get("debug_outputs"):
            pro_log(f"Full traceback: {traceback.format_exc()}")

        return {
            "data": None, 
            "metrics": processing_metrics, 
            "status": "failed", 
            "error": str(e)
        }

# Compatibility wrapper for existing code
def run_preprocess_legacy(config = None):
    """Legacy compatibility wrapper"""
    result = run_preprocess(config, mode = "default")
    return result["data"] if result and result["status"] == "success" else None

# Main execution
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "default"
    result = run_preprocess(mode = mode)

    if result and result["status"] == "success":
        print(f"✅ Preprocessing completed successfully")
        print(f"Data shape: {result['data'].shape}")
        print(f"Processing time: {result['metrics']['processing_time_seconds']:.2f} seconds")
    else:
        print(f"❌ Preprocessing failed")
        if result and "error" in result:
            print(f"Error: {result['error']}")
        sys.exit(1)