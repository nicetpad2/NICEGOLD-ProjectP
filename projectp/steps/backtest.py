

# Step: Backtest
    from backtest_engine import validate_config_yaml
from projectp.model_guard import check_no_data_leak
from projectp.oms_mm.interface import OMSMMEngine
from projectp.pro_log import pro_log
from projectp.utils_data import load_main_training_data
from projectp.utils_data_csv import load_and_prepare_main_csv
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
        from sklearn.metrics import auc, roc_curve
from src.simulation import run_backtest_simulation_v34
from src.utils.data_utils import convert_thai_datetime
from src.utils.log_utils import export_log_to, pro_log_json, set_log_context
from src.utils.resource_auto import (
from tqdm import tqdm
from utils import prepare_csv_auto
    import asyncio
import hashlib
    import json
    import logging
        import matplotlib.pyplot as plt
import os
import pandas as pd
    import psutil
        import pynvml
    import queue
import shutil
import sys
import threading
    import time
import uuid
            import yaml
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

# Import from the root - level backtest_engine.py with error handling
try:
except ImportError as e:

    logger = logging.getLogger(__name__)
    logger.warning(f"Could not import validate_config_yaml: {e}")

    # Create a comprehensive fallback function
    def validate_config_yaml(config_path = "config.yaml"):
        """Fallback validation function when backtest_engine is not available"""
        logger.info(f"Using fallback validation for config: {config_path}")

        # Basic validation - check if file exists
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}")
            return False

        # Try to load as YAML
        try:

            with open(config_path, "r", encoding = "utf - 8") as f:
                config = yaml.safe_load(f)
            logger.info(f"Config file {config_path} loaded successfully")
            return True
        except Exception as yaml_error:
            logger.error(f"Error loading config {config_path}: {yaml_error}")
            return False
        """Fallback function for config validation"""
        return True


# - - - OMS/MM Production - level Integration Example - -  - 
    get_optimal_resource_fraction, 
    print_resource_summary, 
)


def default_fee_model(symbol, qty, price, side):
    return 0.0002 * qty * price  # 2 bps


risk_config = {
    "max_position": {"XAUUSD": 10, "EURUSD": 20}, 
    "max_loss": 0.2, 
    "max_drawdown": 0.15, 
}


console = Console()


def on_event(event_type, event):
    order = event.get("order")
    if order:
        o = order.to_dict() if hasattr(order, "to_dict") else dict(order)
        table = Table(
            title = f"OMSMM {event_type.upper()}", 
            box = box.ROUNDED, 
            show_header = True, 
            header_style = "bold magenta", 
        )
        for k in [
            "id", 
            "symbol", 
            "qty", 
            "order_type", 
            "price", 
            "status", 
            "filled_qty", 
            "timestamp", 
            "user_id", 
            "tag", 
        ]:
            table.add_row(k, str(o.get(k, " - ")))
        if "reject_reason" in o and o["reject_reason"]:
            table.add_row("reject_reason", o["reject_reason"])
        if "qty" in event:
            table.add_row("fill_qty", str(event["qty"]))
        console.print(table)
    else:
        console.print(f"[yellow][OMSMM] {event_type}: {event}")


# สร้าง engine production - ready (ใช้ใน backtest/simulation loop ได้ทันที)
omsmm_engine = OMSMMEngine(
    initial_capital = 10000, risk_config = risk_config, fee_model = default_fee_model
)
omsmm_engine.on_event = on_event

# ตัวอย่างการส่ง batch order (สามารถใช้ใน simulation loop/backtest logic ได้)
# orders = [
#     {'symbol': 'XAUUSD', 'qty': 2, 'order_type': 'MARKET', 'price': 2400}, 
#     {'symbol': 'EURUSD', 'qty': 5, 'order_type': 'LIMIT', 'price': 1.08}, 
# ]
# order_ids = omsmm_engine.batch_send_orders(orders)
# for oid in order_ids:
#     omsmm_engine.fill_order(oid)
# print(omsmm_engine.get_portfolio_stats())
# print(omsmm_engine.get_risk_status())
# print(omsmm_engine.serialize_state())
# - - - End OMS/MM Integration Example - -  - 


def run_backtest(config = None):

    console = Console()
    trace_id = str(uuid.uuid4())
    set_log_context(trace_id = trace_id, pipeline_step = "backtest")
    pro_log_json({"event": "start_backtest"}, tag = "Backtest", level = "INFO")
    validate_config_yaml("config.yaml")
    pro_log("[Backtest] Running backtest step...", tag = "Backtest")

    # - - - FIX: Load predictions.csv which contains features, pred_proba, and label - -  - 
    data_path = os.path.join("output_default", "predictions.csv")
    if not os.path.exists(data_path):
        pro_log(
            f"[Backtest] Predictions data not found: {data_path}. Please run predict step first.", 
            level = "error", 
            tag = "Backtest", 
        )
        return None

    df = pd.read_csv(data_path)
    df.columns = [c.lower() for c in df.columns]
    pro_log(f"[Backtest] Loaded predictions data shape: {df.shape}", tag = "Backtest")

    # - - - FIX: Ensure required columns exist - -  - 
    required_cols = ["label", "pred_proba"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        pro_log(
            f"[Backtest] ERROR: Predictions data missing required columns: {missing_cols}", 
            level = "error", 
            tag = "Backtest", 
        )
        raise ValueError(f"Predictions data missing required columns: {missing_cols}")

    # Guard: No data leak (เทพ)
    if len(df) > 1:
        n = len(df)
        split = int(n * 0.8)
        check_no_data_leak(df.iloc[:split], df.iloc[split:])
    # - - - เทพ: Log resource usage, assert target, auto - diagnosis - -  - 

    pro_log(
        f"[Backtest] RAM: {psutil.virtual_memory().percent:.1f}% used, CPU: {psutil.cpu_percent()}%", 
        tag = "Backtest", 
    )
    if "label" not in df.columns:
        pro_log(
            f"[Backtest] ERROR: No 'label' column found in predictions data!", 
            level = "error", 
            tag = "Backtest", 
        )
        raise ValueError("No 'label' column found in predictions data!")
    if df.isnull().sum().sum() > 0:
        pro_log(
            f"[Backtest] WARNING: Missing values detected in predictions data!", 
            level = "warn", 
            tag = "Backtest", 
        )  # - - - เทพ: Resource Allocation 80% - -  - 
    ram_gb, gpu_gb = get_optimal_resource_fraction(ram_fraction = 0.8, gpu_fraction = 0.8)
    print_resource_summary()
    gpu_display = f"{gpu_gb:.2f} GB" if gpu_gb is not None else "N/A"
    console.print(
        Panel(
            f"[bold green]Allocated RAM: {ram_gb:.2f} GB | GPU: {gpu_display} (80%)", 
            title = "[green]Resource Allocation", 
            border_style = "green", 
        )
    )
    # เรียกใช้ simulation loop จริง (เทพ)
    # ตัวอย่าง mapping argument (ควรปรับตาม config จริง):
    label = config.get("label", "default") if config else "default"
    # - - - รองรับทุนตั้งต้น 100 ดอลลาร์ (default) - -  - 
    initial_capital = config.get("initial_capital", 100) if config else 100
    # TODO: mapping argument อื่น ๆ จาก config ตามที่ simulation ต้องการ
    result = run_backtest_simulation_v34(
        df, 
        label, 
        initial_capital, 
        # ...ใส่ argument อื่น ๆ ตามต้องการ...
    )
    # สมมติ result เป็น DataFrame หรือ dict ที่ export ได้
    out_path = os.path.join("output_default", "backtest_result.csv")
    if os.path.exists(out_path):
        shutil.copy2(out_path, os.path.join(backup_dir, "backtest_result_backup.csv"))
    if hasattr(result, "to_csv"):
        result.to_csv(out_path, index = False)
    elif (
        isinstance(result, dict) and "df" in result and hasattr(result["df"], "to_csv")
    ):
        result["df"].to_csv(out_path, index = False)
    pro_log(
        f"[Backtest] Saved backtest result to {out_path}", 
        level = "success", 
        tag = "Backtest", 
    )
    # - - - เทพ: Export summary, plot, and diagnostics - -  - 
    try:

        if "target" in df.columns and "pred_proba" in df.columns:
            fpr, tpr, _ = roc_curve(df["target"], df["pred_proba"])
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, label = f"ROC curve (AUC = {roc_auc:.2f})")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            plt.savefig(os.path.join("output_default", "backtest_roc_curve.png"))
            plt.close()
            pro_log(f"[Backtest] ROC curve exported.", tag = "Backtest")
    except Exception as e:
        pro_log(f"[Backtest] ROC curve export error: {e}", level = "warn", tag = "Backtest")
    # - - - เทพ: Export resource log (RAM/CPU/GPU) - -  - 

    resource_info = {
        "ram_percent": psutil.virtual_memory().percent, 
        "ram_used_gb": psutil.virtual_memory().used / 1e9, 
        "ram_total_gb": psutil.virtual_memory().total / 1e9, 
        "cpu_percent": psutil.cpu_percent(), 
    }
    # - - - Fix: GPU resource info robust type handling and type cast for None - -  - 
    try:

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(h)

        def _to_float(val):
            try:
                return float(val)
            except Exception:
                return 0.0

        resource_info["gpu_used_gb"] = (
            _to_float(getattr(gpu_mem, "used", 0)) / 1e9
            if hasattr(gpu_mem, "used")
            else 0.0
        )
        resource_info["gpu_total_gb"] = (
            _to_float(getattr(gpu_mem, "total", 0)) / 1e9
            if hasattr(gpu_mem, "total")
            else 0.0
        )
    except Exception:
        resource_info["gpu_used_gb"] = 0.0
        resource_info["gpu_total_gb"] = 0.0
    resource_log_path = os.path.join("output_default", "backtest_resource_log.json")
    with open(resource_log_path, "w", encoding = "utf - 8") as f:
        json.dump(resource_info, f, indent = 2)
    pro_log(f"[Backtest] Resource log exported to {resource_log_path}", tag = "Backtest")

    # - - - เทพ: Export summary metrics - -  - 
    try:
        summary = {}
        if isinstance(result, pd.DataFrame):
            df_result = result
        elif isinstance(result, dict) and "df" in result:
            df_result = result["df"]
        else:
            df_result = None
        if df_result is not None:
            for col in df_result.select_dtypes(include = [float, int]).columns:
                summary[col] = {
                    "mean": float(df_result[col].mean()), 
                    "std": float(df_result[col].std()), 
                    "min": float(df_result[col].min()), 
                    "max": float(df_result[col].max()), 
                }
            summary_path = os.path.join(
                "output_default", "backtest_summary_metrics.json"
            )
            with open(summary_path, "w", encoding = "utf - 8") as f:
                json.dump(summary, f, indent = 2)
            pro_log(
                f"[Backtest] Summary metrics exported to {summary_path}", tag = "Backtest"
            )
    except Exception as e:
        pro_log(
            f"[Backtest] Summary metrics export error: {e}", 
            level = "warn", 
            tag = "Backtest", 
        )

    # - - - เทพ: Export equity curve/PNL plot - -  - 
    try:
        if df_result is not None:
            for col in ["equity", "pnl", "balance"]:
                if col in df_result.columns:

                    plt.figure()
                    plt.plot(df_result[col])
                    plt.title(f"{col.capitalize()} Curve")
                    plt.xlabel("Step")
                    plt.ylabel(col.capitalize())
                    plt.savefig(
                        os.path.join("output_default", f"backtest_{col}_curve.png")
                    )
                    plt.close()
                    pro_log(
                        f"[Backtest] {col.capitalize()} curve exported.", tag = "Backtest"
                    )
    except Exception as e:
        pro_log(
            f"[Backtest] Equity/PNL curve export error: {e}", 
            level = "warn", 
            tag = "Backtest", 
        )

    # - - - เทพ: Export diagnostics log - -  - 
    diagnostics_path = os.path.join("output_default", "backtest_diagnostics.log")
    with open(diagnostics_path, "a", encoding = "utf - 8") as f:
        f.write(f"[DIAG] Backtest run at: {pd.Timestamp.now()}\n")
        f.write(f"Resource: {resource_info}\n")
        if "summary" in locals():
            f.write(f"Summary: {summary}\n")
        f.write(f"Output: {out_path}\n")
    pro_log(
        f"[Backtest] Diagnostics log exported to {diagnostics_path}", tag = "Backtest"
    )

    # - - - เทพ: Assert output file/column completeness - -  - 
    try:
        assert os.path.exists(out_path), f"Backtest output file missing: {out_path}"
        df_out = pd.read_csv(out_path)
        required_cols = ["Time", "Open", "High", "Low", "Close", "Volume"]
        missing_cols = [c for c in required_cols if c not in df_out.columns]
        if missing_cols:
            raise ValueError(f"Backtest output missing columns: {missing_cols}")
        pro_log(f"[Backtest] Output file and columns validated.", tag = "Backtest")
    except Exception as e:
        pro_log(
            f"[Backtest] Output validation error: {e}", level = "error", tag = "Backtest"
        )

    # - - - เทพ: Export timing/performance log - -  - 

    end_time = time.time()
    perf_log_path = os.path.join("output_default", "backtest_perf.log")
    with open(perf_log_path, "a", encoding = "utf - 8") as f:
        f.write(
            f"Backtest finished at {pd.Timestamp.now()} | Duration: {end_time - start_time:.2f} sec\n"
        )
    pro_log(f"[Backtest] Performance log exported to {perf_log_path}", tag = "Backtest")

    # - - - เทพ: Usability hints - -  - 
    pro_log(
        f"[Backtest] ผลลัพธ์หลัก: {out_path}\nดู resource log ที่ {resource_log_path}\nดู summary metrics ที่ {summary_path if 'summary_path' in locals() else ' - '}\nดู diagnostics log ที่ {diagnostics_path}", 
        tag = "Backtest", 
    )
    pro_log_json(
        {"event": "end_backtest", "result": str(out_path)}, 
        tag = "Backtest", 
        level = "SUCCESS", 
    )
    export_log_to(f"logs/backtest_{trace_id}.jsonl")
    return out_path

    # - - - OMS/MM Deep Integration Example (production, async, distributed ready) - -  - 
    # 1. ใช้ omsmm_engine ใน simulation loop (order, fill, risk, event)
    # 2. รองรับ async: event queue, await, callback (ใช้ asyncio, queue, หรือ thread/process pool ได้)
    # 3. รองรับ distributed: export/import state, checkpoint, multi - worker

    event_queue = queue.Queue()  # หรือใช้ asyncio.Queue() สำหรับ async จริง

    def async_event_handler(event_type, event):
        event_queue.put((event_type, event))

    omsmm_engine.on_event = async_event_handler

    # ตัวอย่าง simulation loop (sync/async)
    for i, row in df.iterrows():
        # - - - ตัดสินใจส่ง order ตามกลยุทธ์ - -  - 
        if row["target"] == 1:
            oid = omsmm_engine.send_order(
                symbol = "XAUUSD", qty = 1, order_type = "MARKET", price = row["Close"]
            )
            if oid:
                omsmm_engine.fill_order(oid)
        # - - - async: process event queue (เช่นใน worker/consumer) - -  - 
        while not event_queue.empty():
            event_type, event = event_queue.get()
            # สามารถส่ง event นี้ไปยัง distributed worker, logger, หรือ monitoring system ได้
            # ตัวอย่าง: print(f'[ASYNC EVENT] {event_type}:', event)
        # - - - distributed: checkpoint/export state ทุก N step - -  - 
        if i % 1000 == 0:
            state = omsmm_engine.serialize_state()
            # ตัวอย่าง: save state to disk, send to remote, หรือ broadcast ไปยัง worker อื่น
            # with open(f'output_default/omsmm_checkpoint_{i}.json', 'w') as f:
            #     import json; json.dump(state, f)
    # - - - End OMS/MM Deep Integration Example - -  - 


# - - - OMS/MM + Execution Engine Integration Example (production, async, distributed) - -  - 
# สมมติคุณมี execution engine จริง (REST, WebSocket, FIX, exchange, broker API)
# สามารถเชื่อม OMSMMEngine กับ execution engine ได้ดังนี้:

# สมมติเป็น execution queue (event - driven, async, distributed - ready)
execution_queue = queue.Queue()
execution_response_queue = queue.Queue()


# ตัวอย่าง async execution worker (thread/process/asyncio)
def execution_worker():
    while True:
        try:
            order_req = execution_queue.get(timeout = 1)
        except Exception:
            continue
        # ส่ง order ไป execution engine จริง (REST/WS/FIX)
        # ตัวอย่าง: response = requests.post(...)
        # หรือ ws.send(...), หรือส่งไป exchange/broker API จริง
        # จำลอง response:
        response = {
            "order_id": order_req["order_id"], 
            "status": "FILLED", 
            "fill_qty": order_req["qty"], 
            "price": order_req["price"], 
        }
        time.sleep(0.1)  # simulate latency
        execution_response_queue.put(response)


# Start execution worker (production: ใช้ multi - thread/process/asyncio/distributed ได้)
threading.Thread(target = execution_worker, daemon = True).start()

# - - - End OMS/MM + Execution Engine Integration Example - -  - 


# - - - OMS/MM + Execution Engine Deep Integration (async/distributed, production) - -  - 
def run_oms_execution_integration():
    """
    Run OMS/MM + Execution Engine integration (moved from module level to function to prevent blocking imports)
    """
    # โหลดข้อมูล df ให้ถูกต้องก่อนใช้งาน
    try:
        df, _ = load_main_training_data()
    except Exception as e:
        pro_log(
            f"[Backtest] Error loading main training data: {e}", 
            level = "error", 
            tag = "Backtest", 
        )
        df = None

    if df is not None:
        for i, row in df.iterrows():
            # - - - ตัดสินใจส่ง order ตามกลยุทธ์ - -  - 
            if row["target"] == 1:
                oid = omsmm_engine.send_order(
                    symbol = "XAUUSD", qty = 1, order_type = "MARKET", price = row["Close"]
                )
                if oid:
                    # ส่ง order ไป execution engine จริง (async/distributed)
                    execution_queue.put(
                        {
                            "order_id": oid, 
                            "symbol": "XAUUSD", 
                            "qty": 1, 
                            "price": row["Close"], 
                        }
                    )
            # - - - รับ execution response (async, event - driven, distributed - ready) - -  - 
            while not execution_response_queue.empty():
                exec_resp = execution_response_queue.get()
                # ตรวจสอบสถานะ/ราคา/qty จริงจาก execution engine
                if exec_resp["status"] == "FILLED":
                    omsmm_engine.fill_order(
                        exec_resp["order_id"], fill_qty = exec_resp["fill_qty"]
                    )
                # รองรับ error/retry, logging, distributed broadcast ได้ที่นี่


# - - - End OMS/MM + Execution Engine Deep Integration - -  - 


def load_main_training_data(
    config = None, default_path = "output_default/preprocessed.csv"
):
    """
    Load, validate, and log main training data file (production - ready, multi - format, versioning - ready)
    Args:
        config: dict, may contain 'train_data_path'
        default_path: fallback path
    Returns:
        df: pd.DataFrame
        info: dict (path, shape, columns, hash, summary)
    """
    # import pandas as pd  # ลบออก เพราะ import แล้วที่ global scope

    path = (
        config.get("train_data_path")
        if config and "train_data_path" in config
        else default_path
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training data file not found: {path}")
    ext = os.path.splitext(path)[ - 1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext == ".feather":
        df = pd.read_feather(path)
    else:
        raise ValueError(f"Unsupported training data file format: {ext}")
    # Validate schema/columns
    required_cols = ["target"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in training data!")
    # Log/Export summary
    file_hash = hashlib.md5(open(path, "rb").read()).hexdigest()
    info = {
        "path": path, 
        "shape": df.shape, 
        "columns": list(df.columns), 
        "hash": file_hash, 
        "head": df.head(2).to_dict(), 
    }
    # จุด hook: data versioning, data catalog, lineage, logging
    # ตัวอย่าง: export log
    with open("output_default/train_data_info.json", "w") as f:

        json.dump(info, f, indent = 2)
    return df, info


def load_and_prepare_main_csv(
    path, add_target = False, target_func = None, rename_timestamp = True
):
    """
    Load CSV with Thai Buddhist Timestamp, convert to datetime, add target column if needed, and rename column for pipeline compatibility.
    Args:
        path: str, path to CSV
        add_target: bool, whether to add a 'target' column
        target_func: callable, function to generate target (df) -> Series
        rename_timestamp: bool, rename 'Timestamp' to 'timestamp'
    Returns:
        df: pd.DataFrame (ready for pipeline)
    """
    df = prepare_csv_auto(path)
    # แปลง Timestamp พ.ศ. เป็น datetime ค.ศ. (inplace)
    df = convert_thai_datetime(df, "Timestamp")
    if rename_timestamp and "Timestamp" in df.columns:
        df = df.rename(columns = {"Timestamp": "timestamp"})
    # เติม target อัตโนมัติถ้าต้องการ
    if add_target and "target" not in df.columns:
        if target_func:
            df["target"] = target_func(df)
        else:
            # Default: next close > close = 1 else 0
            df["target"] = (df["Close"].shift( - 1) > df["Close"]).astype(int)
    return df


# ตัวอย่างการใช้งาน (เทพสุด)
# df = load_and_prepare_main_csv('data/raw/your_data_file.csv', add_target = True)
# print(df.dtypes)
# print(df.head())