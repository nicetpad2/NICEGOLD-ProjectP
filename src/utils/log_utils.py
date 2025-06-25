from datetime import datetime
from typing import Any, Optional, Dict
import json
import logging
    import numpy as np
import os
import sys
"""
Modern logging utilities for terminal and file (เทพระดับ enterprise)
- Rich color, structured, JSON, and key - value support
- Context - aware: user, session, run_id, etc.
"""

# - - - Context management - -  - 
LOG_CONTEXT: Dict[str, Any] = {}

def set_log_context(**kwargs):
    """Set global context for all logs (user, session, run_id, etc.)"""
    LOG_CONTEXT.update(kwargs)

def clear_log_context():
    LOG_CONTEXT.clear()

# - - - Logging setup - -  - 
class ContextFilter(logging.Filter):
    def filter(self, record):
        for k, v in LOG_CONTEXT.items():
            setattr(record, k, v)
        return True

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "ts": datetime.fromtimestamp(record.created).isoformat(), 
            "level": record.levelname, 
            "tag": getattr(record, "tag", " - "), 
            "ctx": {k: v for k, v in LOG_CONTEXT.items()}, 
            "msg": record.getMessage(), 
        }
        if hasattr(record, "json_data"):
            log_record.update(record.json_data)
        return json.dumps(log_record, ensure_ascii = False, default = str)

logger = logging.getLogger("pro_logger")
logger.setLevel(logging.INFO)
logger.handlers.clear()

# Console handler (colorized if possible)
console_handler = logging.StreamHandler(sys.stdout)
console_fmt = logging.Formatter(
    "%(asctime)s | %(levelname) - 8s | %(tag)s | %(ctx)s | %(message)s", 
    datefmt = "%Y - %m - %d %H:%M:%S"
)
console_handler.setFormatter(console_fmt)
console_handler.addFilter(ContextFilter())
logger.addHandler(console_handler)

# File handler (JSON log)
log_dir = "logs"
os.makedirs(log_dir, exist_ok = True)
log_path = os.path.join(log_dir, f"system_{datetime.now().strftime('%Y%m%d')}.log")
file_handler = logging.FileHandler(log_path, encoding = "utf - 8")
file_handler.setFormatter(JsonFormatter())
file_handler.addFilter(ContextFilter())
logger.addHandler(file_handler)

# - - - Main log function - -  - 
def pro_log(msg: str, tag: Optional[str] = None, level: str = "INFO", **kwargs):
    """
    Log a message with rich formatting, context, and optional structured data.
    Args:
        msg (str): Main message
        tag (str): Context tag (e.g. 'Backtest', 'SIM')
        level (str): Log level ('INFO', 'SUCCESS', 'ERROR', ...)
        **kwargs: Extra data (will be shown as JSON if present)
    """
    ctx = {**LOG_CONTEXT}
    extra = {"tag": tag or " - ", "ctx": json.dumps(ctx, ensure_ascii = False, default = str)}
    if kwargs:
        msg = f"{msg} | {json.dumps(kwargs, ensure_ascii = False, default = str)}"
    logger.log(getattr(logging, level.upper(), logging.INFO), msg, extra = extra)

# - - - JSON log for machine analytics - -  - 
def pro_log_json(data: dict, tag: Optional[str] = None, level: str = "INFO"):
    """
    Log a structured JSON object (for machine parsing/analytics).
    """
    extra = {"tag": tag or " - ", "ctx": json.dumps(LOG_CONTEXT, ensure_ascii = False, default = str), "json_data": data}
    logger.log(getattr(logging, level.upper(), logging.INFO), "", extra = extra)

# - - - Export log utility - -  - 
def export_log_to(path: str = "logs/exported_log.jsonl", level: str = "INFO"):
    """
    Export all logs at given level or above to a JSONL file (for analytics/BI).
    """
    log_path = f"logs/system_{datetime.now().strftime('%Y%m%d')}.log"
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return
    with open(log_path, "r", encoding = "utf - 8") as fin, open(path, "w", encoding = "utf - 8") as fout:
        for line in fin:
            try:
                rec = json.loads(line)
                if rec.get("level", "INFO") >= level.upper():
                    fout.write(json.dumps(rec, ensure_ascii = False) + "\n")
            except Exception:
                continue
    print(f"Exported log to {path}")

def safe_fmt(val, fmt = "{:.3f}"):
    """Format ค่าเลข/metric/summary ให้ปลอดภัยจาก None, NaN, type error"""
    if val is None:
        return "None"
    try:
        if isinstance(val, float) and np.isnan(val):
            return "NaN"
        return fmt.format(val)
    except Exception:
        return str(val)