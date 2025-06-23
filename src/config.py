#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NICEGOLD-ProjectP Configuration Module
Production-ready configuration with ASCII-only logging for cross-platform compatibility.
"""

import atexit
import importlib
import logging
import os
import pathlib
import subprocess
import sys
import time
import traceback
import warnings as _warnings

# Suppress joblib/loky physical core warnings
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

_warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores*",
    category=UserWarning,
)
_warnings.filterwarnings(
    "ignore", message="joblib.externals.loky.backend.context.*", category=UserWarning
)

# Default constants tested by tests/test_config_defaults.py
MIN_SIGNAL_SCORE_ENTRY = 0.3
M15_TREND_RSI_UP = 60
M15_TREND_RSI_DOWN = 40
FORCED_ENTRY_MIN_GAIN_Z_ABS = 0.5
FORCED_ENTRY_ALLOWED_REGIMES = [
    "Normal",
    "Breakout",
    "StrongTrend",
    "Reversal",
    "Pullback",
    "InsideBar",
    "Choppy",
]
ENABLE_SOFT_COOLDOWN = True
ADAPTIVE_SIGNAL_SCORE_QUANTILE = 0.4
REENTRY_MIN_PROBA_THRESH = 0.40
OMS_ENABLED = True
OMS_DEFAULT = True
PAPER_MODE = False
POST_TRADE_COOLDOWN_BARS = 2

# Additional strategy constants imported by logic.py
USE_MACD_SIGNALS = True
USE_RSI_SIGNALS = True
ENABLE_PARTIAL_TP = True
PARTIAL_TP_LEVELS = [{"r_multiple": 0.8, "close_pct": 0.5}]
PARTIAL_TP_MOVE_SL_TO_ENTRY = True
ENABLE_KILL_SWITCH = True
KILL_SWITCH_MAX_DD_THRESHOLD = 0.15
KILL_SWITCH_CONSECUTIVE_LOSSES_THRESHOLD = 5
RECOVERY_MODE_CONSECUTIVE_LOSSES = 4
min_equity_threshold_pct = 0.70
M15_TREND_ALLOWED = True
ENABLE_SPIKE_GUARD = True

# Default hyperparameters used in training
LEARNING_RATE = 0.01
DEPTH = 6
L2_LEAF_REG = None

# Core imports
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from joblib import dump as joblib_dump
from joblib import load

# Read version from VERSION file
VERSION_FILE = os.path.join(os.path.dirname(__file__), "..", "VERSION")
with open(VERSION_FILE, "r", encoding="utf-8") as vf:
    __version__ = vf.read().strip()

# Register module as 'config' for reload compatibility
if "src.config" not in sys.modules:
    sys.modules["src.config"] = sys.modules[__name__]

# Auto-installation configuration
AUTO_INSTALL_LIBS = False  # If True, attempt to auto-install missing libraries

# Output and models directories
OUTPUT_DIR = Path(__file__).parent.parent / "output_default"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = OUTPUT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Production configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://nicegold:password@localhost:5432/nicegold"
)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = os.getenv("DEBUG", "true").lower() in ("true", "1", "yes")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Performance settings
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
MAX_CONNECTIONS = int(os.getenv("MAX_CONNECTIONS", "100"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))

# Trading limits
MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", "10000"))
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "5000"))
EMERGENCY_STOP_DRAWDOWN = float(os.getenv("EMERGENCY_STOP_DRAWDOWN", "0.15"))

# Data configuration
BASE_DIR = pathlib.Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SYMBOL = os.getenv("SYMBOL", "XAUUSD")
TIMEFRAME = os.getenv("TIMEFRAME", "M1")

# Initialize hyperparameters to prevent missing attribute warnings
for _attr in [
    "subsample",
    "colsample_bylevel",
    "bagging_temperature",
    "random_strength",
    "seed",
]:
    if _attr not in globals():
        globals()[_attr] = None

# Core ML imports
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


# Enhanced logging setup
def setup_enhanced_logging():
    """Setup enhanced logging with better formatting and duplicate prevention"""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    class ColoredFormatter(logging.Formatter):
        """Custom formatter with colors for different log levels"""

        COLORS = {
            "DEBUG": "\\033[36m",
            "INFO": "\\033[32m",
            "WARNING": "\\033[33m",
            "ERROR": "\\033[31m",
            "CRITICAL": "\\033[35m",
            "RESET": "\\033[0m",
        }

        def format(self, record):
            log_color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
            record.levelname = f"{log_color}{record.levelname:<8}{self.COLORS['RESET']}"
            return super().format(record)

    console_handler = logging.StreamHandler()
    console_formatter = ColoredFormatter(
        "%(asctime)s | %(levelname)s | %(name)-15s | %(message)s", datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(logs_dir / "nicegold.log")
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-15s | %(funcName)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return logging.getLogger("NiceGold")


# Initialize enhanced logging
logger = setup_enhanced_logging()
logger.setLevel(logging.INFO)

# Load runtime settings
try:
    from src.utils import get_env_float, load_settings, log_settings

    SETTINGS = load_settings()
    log_settings(SETTINGS, logger)
except ImportError:
    logger.warning("Utils module not available, using default settings")
    SETTINGS = {}

# Logging configuration
BASE_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
LOG_DATE = datetime.now().strftime("%Y-%m-%d")
FOLD_ID = os.getenv("FOLD_ID", "fold0")
LOG_DIR = os.path.join(BASE_LOG_DIR, LOG_DATE, FOLD_ID)
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILENAME = os.path.join(LOG_DIR, f"gold_ai_v{__version__}_qa.log")
DEFAULT_TRADE_LOG_MIN_ROWS = int(os.getenv("TRADE_LOG_MIN_ROWS", 10))

# Setup logger configuration
if os.environ.get("PYTEST_CURRENT_TEST"):
    os.environ["COMPACT_LOG"] = "1"

_compact_log = os.environ.get("COMPACT_LOG", "0") == "1"
_log_level_name = (
    "WARNING" if _compact_log else os.environ.get("LOG_LEVEL", "INFO").upper()
)
_log_level = getattr(logging, _log_level_name, logging.INFO)

logger.setLevel(_log_level)
formatter = logging.Formatter(
    "[%(asctime)s][%(levelname)s][%(process)d][%(filename)s:%(lineno)d] - %(message)s"
)

fh = logging.FileHandler(LOG_FILENAME, mode="w", encoding="utf-8")
fh.setFormatter(formatter)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(formatter)

for h in logger.handlers:
    try:
        h.close()
    except Exception:
        pass

logger.handlers.clear()
logger.addHandler(fh)
logger.addHandler(sh)
logger.propagate = True
atexit.register(logging.shutdown)

root_logger = logging.getLogger()
root_logger.setLevel(_log_level)

logger.info(f"--- (Start) Gold AI v{__version__} ---")
logger.info("--- Loading libraries and checking dependencies ---")


# Library version logging helper
def log_library_version(library_name, library_object=None, version=None):
    """Logs the version of the imported library."""
    try:
        version = version or getattr(library_object, "__version__", "N/A")
        logger.info(f"   (Info) Using {library_name} version: {version}")
    except Exception as e:
        logger.warning(f"   (Warning) Could not retrieve {library_name} version: {e}")


# Log versions of core libraries
log_library_version("Pandas", pd)
log_library_version("NumPy", np)
log_library_version("Scikit-learn", sklearn)


# Library installation and checks
def install_library_if_needed(library_name, import_name=None, install_name=None):
    """Install a library if it's not available and AUTO_INSTALL_LIBS is True"""
    import_name = import_name or library_name.lower()
    install_name = install_name or library_name.lower()

    try:
        lib = __import__(import_name)
        log_library_version(library_name, lib)
        return lib
    except ImportError:
        if AUTO_INSTALL_LIBS:
            logger.info(f"   Installing {library_name} library...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", install_name, "-q"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                lib = __import__(import_name)
                logger.info(f"   (Success) {library_name} installed successfully.")
                log_library_version(library_name, lib)
                return lib
            except Exception as e:
                logger.error(f"   (Error) Cannot install {library_name}: {e}")
                return None
        else:
            logger.warning(
                f"Library '{library_name}' not installed and AUTO_INSTALL_LIBS=False"
            )
            return None


# Install/check optional libraries
tqdm = install_library_if_needed("tqdm")
optuna = install_library_if_needed("Optuna", "optuna")

# TA library
TA_VERSION = "N/A"
try:
    import ta

    TA_VERSION = getattr(ta, "__version__", "N/A")
    if TA_VERSION == "N/A":
        try:
            from importlib.metadata import version as _v

            TA_VERSION = _v("ta")
        except Exception:
            TA_VERSION = "N/A"
    log_library_version("TA", ta, version=TA_VERSION)
except ImportError:
    ta = install_library_if_needed("TA", "ta")
    if ta:
        TA_VERSION = getattr(ta, "__version__", "N/A")

# CatBoost library
try:
    import catboost
    from catboost import CatBoostClassifier, Pool

    logger.info(f"   (Info) CatBoost version: {catboost.__version__}")
    try:
        from catboost.utils import get_gpu_device_count

        gpu_count = get_gpu_device_count()
        logger.info(f"   (Info) GPU count for CatBoost: {gpu_count}")
    except Exception as e:
        logger.warning(f"   (Warning) Cannot check CatBoost GPU count: {e}")
except ImportError:
    catboost = install_library_if_needed("CatBoost", "catboost")
    if catboost:
        try:
            from catboost import CatBoostClassifier, Pool
            from catboost.utils import get_gpu_device_count
        except ImportError:
            CatBoostClassifier = None
            Pool = None
    else:
        CatBoostClassifier = None
        Pool = None

# XGBoost (disabled)
XGBClassifier = None
logger.debug("XGBoost is not used in this version.")

# psutil library
psutil = install_library_if_needed("psutil")

# SHAP library
SHAP_INSTALLED = False
SHAP_AVAILABLE = False
try:
    import shap

    SHAP_INSTALLED = True
    SHAP_AVAILABLE = True
    log_library_version("SHAP", shap)
except ImportError:
    shap = install_library_if_needed("SHAP", "shap")
    if shap:
        SHAP_INSTALLED = True
        SHAP_AVAILABLE = True


def install_shap():
    """Install the shap library if not already available."""
    global SHAP_INSTALLED, SHAP_AVAILABLE, shap
    if SHAP_INSTALLED:
        return
    shap = install_library_if_needed("SHAP", "shap")
    if shap:
        SHAP_INSTALLED = True
        SHAP_AVAILABLE = True


# GPUtil library (optional)
try:
    import GPUtil

    logger.debug("GPUtil library already installed.")
except Exception as e:
    GPUtil = install_library_if_needed("GPUtil", "GPUtil", "gputil")
    if not GPUtil:
        logger.debug(f"GPUtil import error: {e}")


# Colab/Drive setup
def is_colab():
    """Return True if running within Google Colab."""
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"):
        try:
            import google.colab  # noqa: F401

            return True
        except Exception:
            return False
    try:
        from IPython import get_ipython

        ip = get_ipython()
        return (
            bool(ip)
            and getattr(ip, "kernel", None) is not None
            and "google.colab" in str(ip.__class__)
        )
    except Exception:
        return False


# File base configuration
FILE_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FILE_BASE_OVERRIDE = os.getenv("FILE_BASE_OVERRIDE")
DRIVE_BASE_PATH = "/content/drive/MyDrive/Phiradon168"

if FILE_BASE_OVERRIDE and os.path.isdir(FILE_BASE_OVERRIDE):
    FILE_BASE = FILE_BASE_OVERRIDE
elif os.path.isdir(DRIVE_BASE_PATH):
    FILE_BASE = DRIVE_BASE_PATH
elif is_colab():
    try:
        from google.colab import drive

        logger.info("(Info) Running on Google Colab - mounting Google Drive...")
        drive.mount("/content/drive", force_remount=True)
        logger.info("(Success) Mount Google Drive successful")
        FILE_BASE = os.getcwd()
    except Exception as e:
        logger.error(f"(Error) Failed to mount Google Drive: {e}")
        FILE_BASE = os.getcwd()
else:
    logger.info("(Info) Not Colab - using local paths for logs and data storage")

DEFAULT_CSV_PATH_M1 = os.path.join(FILE_BASE, "XAUUSD_M1.csv")
DEFAULT_CSV_PATH_M15 = os.path.join(FILE_BASE, "XAUUSD_M15.csv")
DEFAULT_LOG_DIR = BASE_LOG_DIR

# GPU acceleration setup
USE_GPU_ACCELERATION = os.getenv("USE_GPU_ACCELERATION", "True").lower() in (
    "true",
    "1",
    "yes",
)
cudf = None
cuml = None
cuStandardScaler = None
pynvml = None
nvml_handle = None


def setup_gpu_acceleration():
    """Enhanced GPU setup with better error handling and automatic fallback"""
    global USE_GPU_ACCELERATION, pynvml, nvml_handle

    logger.info("Checking GPU availability...")

    try:
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = (
                torch.cuda.get_device_properties(0).total_memory / 1024**3
            )  # GB

            logger.info(f"GPU Found: {gpu_name}")
            logger.info(f"   GPU Memory: {gpu_memory:.1f} GB")
            logger.info(f"   GPU Count: {device_count}")

            # Test GPU with a simple operation
            try:
                test_tensor = torch.rand(100, 100).cuda()
                result = torch.matmul(test_tensor, test_tensor)
                del test_tensor, result
                torch.cuda.empty_cache()
                logger.info("   GPU Test: Successfully ran matrix multiplication")

                # Setup pynvml for monitoring
                try:
                    import pynvml

                    pynvml.nvmlInit()
                    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    logger.info("   GPU Monitoring: pynvml initialized successfully")
                except Exception as e_nvml:
                    logger.warning(
                        f"   GPU Monitoring: pynvml not available - {e_nvml}"
                    )
                    pynvml = None
                    nvml_handle = None

                USE_GPU_ACCELERATION = True
                logger.info("GPU Acceleration: ENABLED")
                return True

            except Exception as e_test:
                logger.error(f"   GPU Test Failed: {e_test}")
                USE_GPU_ACCELERATION = False

        else:
            logger.info("   CUDA not available or no GPU detected")
            USE_GPU_ACCELERATION = False

    except ImportError as e_import:
        logger.warning(f"PyTorch not available: {e_import}")
        USE_GPU_ACCELERATION = False
    except Exception as e_general:
        logger.error(f"GPU setup failed: {e_general}")
        USE_GPU_ACCELERATION = False

    logger.info(
        f"Final GPU Status: {'ENABLED' if USE_GPU_ACCELERATION else 'DISABLED'}"
    )


# Initialize GPU acceleration
setup_gpu_acceleration()

logger.info("Configuration loaded successfully")
