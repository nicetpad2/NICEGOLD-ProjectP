import argparse
import logging
import logging.config
import os
import subprocess
import sys

import pandas as pd
import yaml

from src import csv_validator
from src.data_loader import auto_convert_gold_csv
from src.real_data_loader import RealDataLoader, load_real_data
from src.state_manager import StateManager


# [Patch v6.8.17] CSV to Parquet helper for preprocess stage
def auto_convert_csv_to_parquet(source_path: str, dest_folder) -> None:
    """Convert CSV file to Parquet in ``dest_folder`` with safe fallback."""
    from pathlib import Path

    dest_folder = Path(dest_folder)
    dest_folder.mkdir(parents=True, exist_ok=True)

    if not source_path or not os.path.exists(source_path):
        logger.warning("[AutoConvert] Source CSV not found: %s", source_path)
        return

    try:
        df = pd.read_csv(source_path)
    except Exception as exc:  # pragma: no cover - unexpected read error
        logger.error("[AutoConvert] Failed reading %s: %s", source_path, exc)
        return

    dest_file = dest_folder / (Path(source_path).stem + ".parquet")
    try:
        df.to_parquet(dest_file)
        logger.info("[AutoConvert] Saved Parquet to %s", dest_file)
    except Exception as exc:
        logger.warning(
            "[AutoConvert] Could not save Parquet (%s). Saving CSV fallback", exc
        )
        df.to_csv(dest_file.with_suffix(".csv"), index=False)


from src.utils.errors import PipelineError
from src.utils.hardware import has_gpu
from src.utils.model_utils import get_latest_model_and_threshold
from src.utils.pipeline_config import DEFAULT_CONFIG_FILE, PipelineConfig, load_config

logger = logging.getLogger(__name__)


def setup_logging(level: str) -> None:
    """Configure logging from YAML file."""
    with open("config/logger_config.yaml", "r") as f:
        log_cfg = yaml.safe_load(f.read())
    if level:
        log_cfg["root"]["level"] = level.upper()
    logging.config.dictConfig(log_cfg)


# [Patch v5.8.2] CLI pipeline orchestrator


def parse_args(args=None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Pipeline controller")
    parser.add_argument(
        "--mode",
        choices=[
            "preprocess",
            "sweep",
            "threshold",
            "backtest",
            "report",
            "all",
            "full_pipeline",
        ],
        default="all",
        help="Stage of the pipeline to run",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_FILE,
        help="Path to pipeline configuration file",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging level override (e.g., DEBUG)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (use limited rows for fast run)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        help="Limit number of rows loaded from data (overrides debug default)",
    )
    parser.add_argument("--profile", action="store_true", help="Profile backtest stage")
    parser.add_argument(
        "--output-file",
        default="backtest_profile.prof",
        help="Profiling output file",
    )
    parser.add_argument(
        "--live-loop",
        type=int,
        default=0,
        help="Run live trading loop after pipeline (number of iterations)",
    )
    return parser.parse_args(args)


def run_preprocess(config: PipelineConfig, runner=subprocess.run) -> None:
    """Run data preprocessing stage using real data from datacsv folder."""
    logger.info("[Stage] preprocess - Using real data from datacsv")

    from pathlib import Path

    # Initialize real data loader
    try:
        real_loader = RealDataLoader()
        data_info = real_loader.get_data_info()
        logger.info("Real data info: %s", data_info)

        # Load real data to validate it's accessible
        logger.info("Validating real data access...")
        test_df = real_loader.load_m1_data(
            limit_rows=1000
        )  # Load sample for validation
        logger.info("✅ Real data validation successful: %d rows loaded", len(test_df))

    except Exception as exc:
        logger.error("❌ Failed to access real data from datacsv: %s", exc)
        raise PipelineError("Real data access failed") from exc

    parquet_output_dir_str = getattr(config, "parquet_dir", None)
    if not parquet_output_dir_str:
        base_data_dir = getattr(config, "data_dir", "./data")
        parquet_output_dir = Path(base_data_dir) / "parquet_cache"
        logger.warning(
            "[AutoConvert] 'data.parquet_dir' not set in config. Defaulting to: %s",
            parquet_output_dir,
        )
    else:
        parquet_output_dir = Path(parquet_output_dir_str)

    # Use real data paths from datacsv
    m1_real_path = "datacsv/XAUUSD_M1.csv"
    m15_real_path = "datacsv/XAUUSD_M15.csv"

    logger.info("Converting real M1 data to parquet...")
    auto_convert_csv_to_parquet(
        source_path=m1_real_path, dest_folder=parquet_output_dir
    )

    logger.info("Converting real M15 data to parquet...")
    auto_convert_csv_to_parquet(
        source_path=m15_real_path, dest_folder=parquet_output_dir
    )

    # Validate real M1 data
    if os.path.exists(m1_real_path):
        try:
            csv_validator.validate_and_convert_csv(m1_real_path)
            logger.info("✅ Real M1 data validation successful")
        except FileNotFoundError as exc:
            logger.error("[Validation] Real M1 CSV file not found: %s", exc)
        except ValueError as exc:
            logger.error("[Validation] Real M1 CSV validation error: %s", exc)
        except Exception as exc:
            logger.error("[Validation] Real M1 CSV validation failed: %s", exc)
    else:
        logger.error("[Validation] Real M1 CSV file not found: %s", m1_real_path)
        raise PipelineError(f"Real M1 data file not found: {m1_real_path}")

    # Clean real data
    fill_method = getattr(config, "cleaning_fill_method", "drop")
    try:
        logger.info("Cleaning real M1 data...")
        runner(
            [
                os.environ.get("PYTHON", "python"),
                "src/data_cleaner.py",
                m1_real_path,
                "--fill",
                fill_method,
            ],
            check=True,
        )
        logger.info("✅ Real data preprocessing completed successfully")
    except subprocess.CalledProcessError as exc:
        logger.error("❌ Real data preprocessing failed", exc_info=True)
        raise PipelineError("Real data preprocess stage failed") from exc


def run_sweep(config: PipelineConfig, runner=subprocess.run) -> None:
    """Run hyperparameter sweep stage."""
    logger.info("[Stage] sweep")
    try:
        runner(
            [os.environ.get("PYTHON", "python"), "tuning/hyperparameter_sweep.py"],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.error("Sweep failed", exc_info=True)
        raise PipelineError("sweep stage failed") from exc


def run_threshold(config: PipelineConfig, runner=subprocess.run) -> None:
    """Run threshold optimization stage."""
    logger.info("[Stage] threshold")
    try:
        runner(
            [os.environ.get("PYTHON", "python"), "threshold_optimization.py"],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.error("Threshold optimization failed", exc_info=True)
        raise PipelineError("threshold stage failed") from exc


def run_backtest_pipeline(features_df, price_df, model_path, threshold) -> None:
    """[Patch v5.9.12] Execute simple backtest pipeline."""
    logger.info("Running backtest with model=%s threshold=%s", model_path, threshold)
    try:
        from src.main import run_pipeline_stage

        run_pipeline_stage("backtest")
    except Exception:
        logger.exception("Internal backtest error")
        raise


def run_backtest(config: PipelineConfig, pipeline_func=run_backtest_pipeline) -> None:
    """Run backtest stage."""
    logger.info("[Stage] backtest")
    from config import config as cfg

    try:
        # เดิม: from ProjectP import load_trade_log
        # ย้ายฟังก์ชัน load_trade_log ไปไว้ในไฟล์ utils หรือไฟล์ใหม่ แล้ว import จากตรงนั้นแทน
        from utils import load_trade_log
    except Exception:  # pragma: no cover - fallback for tests

        def load_trade_log(*_a, **_kw):
            return pd.DataFrame()

    trade_log_file = getattr(cfg, "TRADE_LOG_PATH", None)
    try:
        trade_df = load_trade_log(
            trade_log_file,
            min_rows=getattr(cfg, "MIN_TRADE_ROWS", 10),
        )
    except FileNotFoundError as exc:
        logger.error("Trade log file not found: %s", exc)
        trade_df = pd.DataFrame(columns=["timestamp", "price", "signal"])
        logger.info("Initialized empty trade_df for pipeline execution.")
    except ValueError as exc:
        logger.error("Invalid trade log format: %s", exc)
        trade_df = pd.DataFrame(columns=["timestamp", "price", "signal"])
        logger.info("Initialized empty trade_df for pipeline execution.")
    except Exception as exc:
        logger.error("Failed loading trade log: %s", exc)
        trade_df = pd.DataFrame(columns=["timestamp", "price", "signal"])
        logger.info("Initialized empty trade_df for pipeline execution.")
    else:
        logger.debug("Loaded trade log with %d rows", len(trade_df))
    model_dir = config.model_dir
    model_path, threshold = get_latest_model_and_threshold(
        model_dir, config.threshold_file
    )
    try:
        pipeline_func(pd.DataFrame(), pd.DataFrame(), model_path, threshold)
    except Exception as exc:
        logger.error("Backtest failed", exc_info=True)
        raise PipelineError("backtest stage failed") from exc


def run_report(config: PipelineConfig) -> None:
    """Generate report stage."""
    logger.info("[Stage] report")
    try:
        from src.main import run_pipeline_stage

        run_pipeline_stage("report")
    except Exception as exc:
        logger.error("Report failed", exc_info=True)
        raise PipelineError("report stage failed") from exc


from src.pipeline_manager import PipelineManager


def run_all(config: PipelineConfig) -> None:
    """Run all pipeline stages sequentially."""
    logger.info("[Stage] all")
    run_preprocess(config)
    run_sweep(config)
    run_threshold(config)
    run_backtest(config)
    run_report(config)
    qa_path = os.path.join(config.model_dir, ".qa_pipeline.log")
    with open(qa_path, "a", encoding="utf-8") as fh:
        fh.write("qa completed\n")
    logger.info("[Stage] all completed")


def main(args=None) -> int:
    """Entry point for command-line execution."""
    parsed = parse_args(args)
    config = load_config(parsed.config)
    setup_logging(parsed.log_level or config.log_level)
    state_manager = StateManager(state_file_path="output/system_state.json")

    DEBUG_DEFAULT_ROWS = 2000
    if parsed.rows is not None:
        max_rows = parsed.rows
    elif parsed.debug:
        max_rows = DEBUG_DEFAULT_ROWS
    else:
        max_rows = None

    if max_rows:
        print(f"--- [DEBUG MODE] ใช้ข้อมูลจริงจาก datacsv แต่จำกัด max_rows={max_rows} ---")
        logger.info(
            "🔧 Debug mode: Using real data from datacsv with row limit=%d", max_rows
        )
        # Set global variable for real data loader to use row limit
        import os

        os.environ["NICEGOLD_ROW_LIMIT"] = str(max_rows)
    else:
        print("--- [FULL PIPELINE] ใช้ข้อมูลจริงจาก datacsv ทั้งหมด ---")
        logger.info("🚀 Full pipeline: Using complete real data from datacsv")
        # Remove any row limit
        import os

        if "NICEGOLD_ROW_LIMIT" in os.environ:
            del os.environ["NICEGOLD_ROW_LIMIT"]

    if has_gpu():
        logger.info("GPU detected")
    else:
        logger.info("GPU not available, running on CPU")

    stage = parsed.mode
    if stage == "full_pipeline":
        stage = "all"
    logger.debug("Selected stage: %s", stage)

    try:
        if parsed.profile and stage == "backtest":
            import profile_backtest

            profile_backtest.run_profile(
                lambda: run_backtest(config), parsed.output_file
            )
            return 0
        if stage == "preprocess":
            run_preprocess(config)
        elif stage == "sweep":
            run_sweep(config)
        elif stage == "threshold":
            run_threshold(config)
        elif stage == "backtest":
            run_backtest(config)
        elif stage == "report":
            run_report(config)
        else:
            run_all(config)

        if parsed.live_loop > 0:
            import src.main as src_main

            src_main.run_live_trading_loop(parsed.live_loop)
    except PipelineError as exc:
        logger.error("Pipeline error: %s", exc, exc_info=True)
        return 1
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc, exc_info=True)
        return 1
    except ValueError as exc:
        logger.error("Invalid value: %s", exc, exc_info=True)
        return 1
    except KeyboardInterrupt:
        logger.warning(
            "\u0e1c\u0e39\u0e49\u0e43\u0e0a\u0e49\u0e22\u0e01\u0e40\u0e25\u0e34\u0e01\u0e01\u0e32\u0e23\u0e17\u0e33\u0e07\u0e32\u0e19"
        )
        return 1
    except Exception as exc:  # pragma: no cover - unexpected errors
        logger.error("Unexpected error: %s", exc, exc_info=True)
        return 1
    finally:
        state_manager.save_state()
        logger.info("Main script finished. Final state saved.")
    return 0


from projectp.cli import main_cli

if __name__ == "__main__":
    # main.py ถูกปิดการใช้งาน entry point เพื่อให้รันผ่าน ProjectP.py เท่านั้น
    # หากต้องการใช้งาน ให้รัน ProjectP.py

    raise SystemExit("กรุณารันผ่าน ProjectP.py เท่านั้น (main.py ถูกปิดการใช้งาน)")
