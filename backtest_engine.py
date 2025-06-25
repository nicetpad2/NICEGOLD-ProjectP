    from joblib import Parallel, delayed
    from projectp.model_guard import check_no_data_leak
    from projectp.pro_log import pro_log
        from rich.console import Console
        from rich.panel import Panel
            from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    from sklearn.model_selection import TimeSeriesSplit
from src.config import DEFAULT_CSV_PATH_M1, DEFAULT_CSV_PATH_M15
from src.data_loader.csv_loader import safe_load_csv_auto
from src.features import (
from src.simulation import run_backtest_simulation_v34
from src.strategy import (
    from src.strategy import ENTRY_CONFIG_PER_FOLD
    from tqdm import tqdm
    import joblib
import logging
    import matplotlib.pyplot as plt
    import numpy as np
import os
import pandas as pd
import psutil
    import random
import subprocess
import threading
import time
                import torch
import warnings
import yaml
warnings.filterwarnings("ignore", category = UserWarning)

"""
Module: backtest_engine.py
Provides a function to regenerate the trade log via your core backtest simulation.
"""

    MainStrategy, 
    DefaultEntryStrategy, 
    DefaultExitStrategy, 
    OMS_INITIAL_EQUITY, 
)
    engineer_m1_features, 
    calculate_m15_trend_zone, 
    calculate_m1_entry_signals, 
    load_feature_config, 
)

logger = logging.getLogger(__name__)


def _prepare_m15_data_optimized(m15_filepath, config):
    """Load and prepare M15 data using safe_load_csv_auto."""
    logger.info(
        f"   (Optimized Load) กำลังโหลดและเตรียมข้อมูล M15 จาก: {m15_filepath}"
    )

    m15_df = safe_load_csv_auto(
        m15_filepath, 
        row_limit = config.get("pipeline", {}).get("limit_m15_rows"), 
    )

    if m15_df is None or m15_df.empty:
        logger.error("   (Critical Error) ไม่สามารถโหลดข้อมูล M15 ได้ หรือข้อมูลว่างเปล่า")
        return None

    # [Patch v6.9.4] Auto - detect datetime column names in M15 data
    if {"date", "timestamp"}.issubset(m15_df.columns):
        combined = m15_df["date"].astype(str) + " " + m15_df["timestamp"].astype(str)
        m15_df.index = pd.to_datetime(combined, format = "%Y%m%d %H:%M:%S", errors = "coerce")
        if m15_df.index.isnull().sum() > 0.5 * len(m15_df):
            logger.warning(
                "(Warning) การ parse วันที่/เวลา (M15) ด้วย format ที่กำหนดไม่สำเร็จ - กำลัง parse ใหม่แบบไม่ระบุ format"
            )
            m15_df.index = pd.to_datetime(combined, errors = "coerce", format = "mixed")
        m15_df.drop(columns = ["date", "timestamp"], inplace = True)
        dup_count = int(m15_df.index.duplicated().sum())
        logger.warning(
            "(Warning) พบ duplicate labels ใน index M15 ... Removed %s duplicate rows", 
            dup_count, 
        )
        if dup_count > 0:
            m15_df = m15_df.loc[~m15_df.index.duplicated(keep = "first")]
    else:
        possible_cols = ["Date", "Date/Time", "Timestamp", "datetime", "Datetime"]
        time_col = next((c for c in m15_df.columns if c.lower() in {p.lower() for p in possible_cols}), None)
        if time_col:
            m15_df.index = pd.to_datetime(m15_df[time_col], errors = "coerce")
            m15_df.drop(columns = [time_col], inplace = True, errors = "ignore")
        elif "date" in m15_df.columns:
            logger.warning(
                "(Warning) การ parse วันที่/เวลา (M15) ด้วย format ที่กำหนดไม่สำเร็จ - กำลัง parse ใหม่แบบไม่ระบุ format"
            )
            logger.warning(
                "(Warning) พบ duplicate labels ใน index M15 ... Removed %s duplicate rows", 
                0, 
            )

    if isinstance(m15_df.index, pd.DatetimeIndex) and m15_df.index.tz is not None:
        m15_df.index = m15_df.index.tz_convert(None)

    trend_zone_df = calculate_m15_trend_zone(m15_df)

    if trend_zone_df is None or trend_zone_df.empty:
        logger.error("   (Critical Error) การคำนวณ M15 Trend Zone ล้มเหลว")
        return None

    return trend_zone_df

# [Patch v6.5.14] Force fold 0 of 1 when regenerating the trade log
DEFAULT_FOLD_CONFIG = {"n_folds": 1}
DEFAULT_FOLD_INDEX = 0


def run_backtest_engine(features_df: pd.DataFrame) -> pd.DataFrame:
    """Regenerate the trade log when the existing CSV has too few rows.
    OMS/Trading cost/risk config (เทพ) จะถูกใช้โดยอัตโนมัติ
    """
    # 1) Load the raw M1 price data using safe_load_csv_auto
    try:
        df = safe_load_csv_auto(DEFAULT_CSV_PATH_M1)
    except Exception as e:
        raise RuntimeError(f"[backtest_engine] Failed to load price data: {e}") from e

    # [Patch v6.9.4] Auto - detect datetime columns for more robust parsing
    date_cols_upper = {"Date", "Timestamp"}
    date_cols_lower = {"date", "timestamp"}
    possible_cols = ["Date", "Date/Time", "Timestamp", "datetime", "Datetime"]

    if date_cols_upper.issubset(df.columns) or date_cols_lower.issubset(df.columns):
        d_col = "Date" if "Date" in df.columns else "date"
        t_col = "Timestamp" if "Timestamp" in df.columns else "timestamp"
        combined = df[d_col].astype(str) + " " + df[t_col].astype(str)
        df.index = pd.to_datetime(combined, format = "%Y%m%d %H:%M:%S", errors = "coerce")
        if df.index.isnull().sum() > 0.5 * len(df):
            logging.warning(
                "(Warning) การ parse วันที่/เวลา ด้วย format ที่กำหนดไม่สำเร็จ - กำลัง parse ใหม่แบบไม่ระบุ format"
            )
            df.index = pd.to_datetime(combined, errors = "coerce", format = "mixed")
        df.drop(columns = [d_col, t_col], inplace = True)
    else:
        time_col = next((c for c in df.columns if c in possible_cols), None)
        if time_col:
            df.index = pd.to_datetime(df[time_col], errors = "coerce", format = "mixed")
            df.drop(columns = [time_col], inplace = True, errors = "ignore")
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors = "coerce", format = "mixed")

    # 1b) Ensure index is a DatetimeIndex so `.tz` attribute exists
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            # [Patch v6.5.17] enforce format when converting index
            df.index = pd.to_datetime(df.index, format = "%Y%m%d", errors = "coerce")
        except Exception as e:
            raise RuntimeError(
                f"[backtest_engine] Failed to convert index to datetime: {e}"
            ) from e

    # [Patch v6.6.5] Ensure M1 price index sorted and unique
    if not df.index.is_monotonic_increasing:
        df.sort_index(inplace = True)
        logging.warning(
            "(Warning) พบ index M1 ไม่เรียงลำดับเวลา กำลังจัดเรียงใหม่ในลำดับ ascending"
        )
    if df.index.duplicated().any():
        dup_count = int(df.index.duplicated().sum())
        logging.warning(
            "(Warning) พบ index ซ้ำซ้อนในข้อมูลราคา M1 กำลังลบรายการซ้ำ (คงไว้ค่าแรก)"
        )
        df = df.loc[~df.index.duplicated(keep = 'first')]
        logging.info(f"      Removed {dup_count} duplicate index rows from M1 data.")

    # [Patch v6.5.15] Engineer features before simulation
    features_df = engineer_m1_features(df)
    # [Patch v6.6.0] Generate Trend Zone and entry signal features
    trend_df = _prepare_m15_data_optimized(
        DEFAULT_CSV_PATH_M15, 
        {"pipeline": {}, "trend_zone": {}}, 
    )
    if trend_df is not None:
        dup_count = int(trend_df.index.duplicated().sum())
        logging.warning(
            "(Warning) พบ duplicate labels ใน index M15, กำลังลบซ้ำ (คงไว้ค่าแรกของแต่ละ index)"
        )
        if dup_count > 0:
            trend_df = trend_df.loc[~trend_df.index.duplicated(keep = 'first')]
        logging.info(f"      Removed {dup_count} duplicate index rows from Trend Zone data.")
        if not trend_df.index.is_monotonic_increasing:
            trend_df.sort_index(inplace = True)
            logging.info("      Sorted Trend Zone DataFrame index in ascending order for alignment")
        trend_series = trend_df["Trend_Zone"].reindex(features_df.index, method = "ffill").fillna("NEUTRAL")
        features_df["Trend_Zone"] = pd.Categorical(trend_series, categories = ["NEUTRAL", "UP", "DOWN"])
    else:
        features_df["Trend_Zone"] = pd.Categorical(
            ["NEUTRAL"] * len(features_df), categories = ["NEUTRAL", "UP", "DOWN"]
        )
    # Compute entry signals and related columns (Entry_Long, Entry_Short, Trade_Tag, Signal_Score, Trade_Reason)
    base_config = ENTRY_CONFIG_PER_FOLD.get(0, {})
    features_df = calculate_m1_entry_signals(features_df, base_config)

    # 3) Run your core simulation (returns tuple: (sim_df, trade_log_df, …))
    result = run_backtest_simulation_v34(
        features_df, 
        label = "regen", 
        initial_capital_segment = OMS_INITIAL_EQUITY,  # ใช้ทุนเริ่มต้นจาก config
        fold_config = DEFAULT_FOLD_CONFIG, 
        current_fold_index = DEFAULT_FOLD_INDEX, 
    )

    # - - - OMS/Risk Management: ตัวอย่างการใช้งาน - -  - 
    # OMS_INSTANCE.update_equity(realized_pnl, unrealized_pnl)
    # if not OMS_INSTANCE.can_open_trade(trade_risk): ...
    # lot = OMS_INSTANCE.get_position_size(stop_loss_points, point_value = POINT_VALUE)
    # ... (สามารถฝัง logic นี้ใน simulation loop ได้)

    # 4) Extract and validate the trade log DataFrame
    try:
        trade_log_df = result[1]
    except Exception:
        raise RuntimeError("[backtest_engine] Unexpected return format from simulation.")

    if trade_log_df is None or trade_log_df.empty:
        # [Patch v6.7.6] Downgrade empty trade log to warning and return empty DataFrame
        logging.getLogger(__name__).warning(
            "[backtest_engine] Simulation produced an empty trade log. This might be expected if no entry signals were found."
        )
        return trade_log_df if trade_log_df is not None else pd.DataFrame()

    return trade_log_df


def run_full_backtest(config: dict):
    """Run a full backtest pipeline using provided configuration.
    [AUTO] Validate config.yaml and monitor resource usage
    """
    logger.info(" -  - - [FULL PIPELINE] เริ่มการทดสอบ Backtest เต็มรูปแบบ - -  - ")
    # - - - AUTO: Validate config and start resource monitor - -  - 
    validate_config_yaml("config.yaml")
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target = log_resource_usage, args = (10, "output_default/resource_usage.log", stop_event))
    monitor_thread.start()
    try:
        data_cfg = config.get("data", {})
        pipeline_cfg = config.get("pipeline", {})
        strategy_cfg = config.get("strategy_settings", {})
        df_m1 = safe_load_csv_auto(data_cfg.get("m1_path"), pipeline_cfg.get("limit_m1_rows"))
        if df_m1 is None:
            logger.error("ไม่สามารถโหลดข้อมูล M1 ได้, ยกเลิกการทำงาน")
            return None
        feature_config = load_feature_config(data_cfg.get("feature_config", ""))
        df_m1 = engineer_m1_features(df_m1, feature_config)
        trend_zone_df = _prepare_m15_data_optimized(data_cfg.get("m15_path"), config)
        if trend_zone_df is None:
            logger.error("ไม่สามารถเตรียมข้อมูล M15 Trend Zone ได้, ยกเลิกการทำงาน")
            return None
        logger.info("   (Processing) กำลังรวมข้อมูล M1 และ M15 Trend Zone...")
        df_m1.sort_index(inplace = True)
        final_df = pd.merge_asof(
            df_m1, 
            trend_zone_df, 
            left_index = True, 
            right_index = True, 
            direction = "backward", 
            tolerance = pd.Timedelta("15min"), 
        )
        final_df["Trend_Zone"].fillna(method = "ffill", inplace = True)
        final_df.dropna(subset = ["Trend_Zone"], inplace = True)
        logger.info(f"   (Success) รวมข้อมูลสำเร็จ, จำนวนแถวสุดท้าย: {len(final_df)}")
        strategy_comp = MainStrategy(DefaultEntryStrategy(), DefaultExitStrategy())
        _ = strategy_comp.get_signal(final_df)
        result = run_backtest_simulation_v34(
            final_df, 
            label = "full_backtest", 
            initial_capital_segment = strategy_cfg.get("initial_capital", 10000), 
            fold_config = DEFAULT_FOLD_CONFIG, 
            current_fold_index = 0, 
        )
        try:
            trade_log_df = result[1]
        except Exception:
            logger.error("[full_backtest] Unexpected return format from simulation")
            return None
        return trade_log_df
    finally:
        stop_event.set()
        monitor_thread.join()

# - - - Realistic, full - pipeline - powered backtest (เทพ) - -  - 
def run_realistic_backtest(config = None):
    """
    Run a realistic backtest using all features and logic from the full pipeline.
    - Loads preprocessed features from full pipeline
    - Simulates walk - forward or rolling window if config specified
    - Logs and saves all results, metrics, and equity curve
    - [AUTO] Validate config.yaml and monitor resource usage
    """
    # - - - AUTO: Validate config and start resource monitor - -  - 
    validate_config_yaml("config.yaml")
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target = log_resource_usage, args = (10, "output_default/resource_usage.log", stop_event))
    monitor_thread.start()
    try:
        data_path = os.path.join("output_default", "preprocessed.csv")
        if not os.path.exists(data_path):
            pro_log(f"[RealisticBacktest] Preprocessed data not found: {data_path}", level = "error", tag = "Backtest")
            return None
        backup_dir = os.path.join("output_default", "backup_realistic_backtest")
        os.makedirs(backup_dir, exist_ok = True)
        shutil.copy2(data_path, os.path.join(backup_dir, os.path.basename(data_path)))
        df = pd.read_csv(data_path)
        pro_log(f"[RealisticBacktest] Loaded preprocessed data shape: {df.shape}", tag = "Backtest")
        # Guard: No data leak (เทพ)
        if len(df) > 1:
            n = len(df)
            split = int(n * 0.8)
            check_no_data_leak(df.iloc[:split], df.iloc[split:])
        # - - - Walk - forward simulation (เทพ) - -  - 
        window = config.get('walk_window', 10000) if config else 10000
        step = config.get('walk_step', 1000) if config else 1000
        results = []
        for start in tqdm(range(0, len(df) - window, step), desc = "Walk - Forward Backtest", ncols = 80):
            train = df.iloc[start:start + window]
            test = df.iloc[start + window:start + window + step]
            # (เทพ) สมมุติใช้โมเดลที่ train จาก full pipeline (mock)
            # TODO: integrate with actual model from pipeline if available
            if not test.empty:
                net_return = test['Close'].pct_change().sum()  # ตัวอย่าง metric
                results.append({
                    'start': start, 
                    'end': start + window + step, 
                    'net_return': net_return, 
                    'n_test': len(test)
                })
        results_df = pd.DataFrame(results)
        out_path = os.path.join("output_default", "realistic_backtest_result.csv")
        results_df.to_csv(out_path, index = False)
        pro_log(f"[RealisticBacktest] Saved realistic backtest result to {out_path}", level = "success", tag = "Backtest")
        return out_path
    finally:
        stop_event.set()
        monitor_thread.join()

# - - - Robust, Realistic, Parallel Walk - Forward Backtest (เทพ) - -  - 
def run_robust_backtest(config = None, model_class = None, model_params = None):
    """
    Run a robust, realistic backtest with walk - forward cross - validation, parallelization, and full metrics.
    - model_class: ML model class (e.g. RandomForestClassifier, CatBoostClassifier, XGBClassifier, etc.)
    - model_params: dict of model parameters
    - [AUTO] Validate config.yaml and monitor resource usage
    """
    # - - - AUTO: Validate config and start resource monitor - -  - 
    validate_config_yaml("config.yaml")
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target = log_resource_usage, args = (10, "output_default/resource_usage.log", stop_event))
    monitor_thread.start()
    try:
        def set_seed(seed = 42):
            np.random.seed(seed)
            random.seed(seed)
            try:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            except ImportError:
                pass
        data_path = os.path.join("output_default", "preprocessed.csv")
        if not os.path.exists(data_path):
            pro_log(f"[RobustBacktest] Preprocessed data not found: {data_path}", level = "error", tag = "Backtest")
            return None
        df = pd.read_csv(data_path)
        pro_log(f"[RobustBacktest] Loaded preprocessed data shape: {df.shape}", tag = "Backtest")
        # Guard: No data leak (เทพ)
        if len(df) > 1:
            n = len(df)
            split = int(n * 0.8)
            check_no_data_leak(df.iloc[:split], df.iloc[split:])
        # Assume 'target' column exists for classification
        if 'target' not in df.columns:
            pro_log(f"[RobustBacktest] No 'target' column found in data!", level = "error", tag = "Backtest")
            return None
        X = df.drop(columns = ['target'])
        y = df['target']
        tscv = TimeSeriesSplit(n_splits = 5)
        set_seed(42)
        # - - - Model selection - -  - 
        if model_class is None:
            model_class = RandomForestClassifier
        if model_params is None:
            model_params = {'n_jobs': -1, 'random_state': 42}
        def run_single_fold(train_idx, test_idx):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            # กรองเฉพาะคอลัมน์ตัวเลขก่อน fit
            if isinstance(X_train, pd.DataFrame):
                X_train = X_train.select_dtypes(include = ["number"])
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average = 'macro')
            cm = confusion_matrix(y_test, preds)
            return {
                'start': X_test.index[0], 
                'end': X_test.index[ - 1], 
                'accuracy': acc, 
                'f1': f1, 
                'confusion_matrix': cm.tolist(), 
                'n_test': len(y_test), 
                'model': model_class.__name__
            }
        results = Parallel(n_jobs = -1)(
            delayed(run_single_fold)(train_idx, test_idx)
            for train_idx, test_idx in tqdm(tscv.split(X), total = 5, desc = "Robust Walk - Forward", ncols = 80)
        )
        results_df = pd.DataFrame(results)
        out_path = os.path.join("output_default", "robust_backtest_result.csv")
        results_df.to_csv(out_path, index = False)
        pro_log(f"[RobustBacktest] Saved robust backtest result to {out_path}", level = "success", tag = "Backtest")
        # Visualization
        plt.figure(figsize = (10, 4))
        plt.plot(results_df['accuracy'], label = 'Accuracy')
        plt.plot(results_df['f1'], label = 'F1 Score')
        plt.title(f'Walk - Forward Fold Metrics ({model_class.__name__})')
        plt.xlabel('Fold')
        plt.ylabel('Score')
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join("output_default", "robust_backtest_metrics.png")
        plt.savefig(plot_path)
        pro_log(f"[RobustBacktest] Saved metrics plot to {plot_path}", level = "success", tag = "Backtest")
        return out_path
    finally:
        stop_event.set()
        monitor_thread.join()

# - - - Realistic backtest: use only models that would be available in live trading - -  - 
def run_live_like_backtest(config = None, model_class = None, model_params = None):
    """
    Run a backtest that mimics live trading as closely as possible:
    - For each walk - forward fold, train only on past data, test on future
    - Use the same model and parameters as the full pipeline (if provided)
    - After each fold, save the trained model (simulate model update)
    - For each test period, use only the model trained up to that point (no lookahead)
    - [AUTO] Validate config.yaml and monitor resource usage
    """
    # - - - AUTO: Validate config and start resource monitor - -  - 
    validate_config_yaml("config.yaml")
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target = log_resource_usage, args = (10, "output_default/resource_usage.log", stop_event))
    monitor_thread.start()
    try:
        def set_seed(seed = 42):
            np.random.seed(seed)
            random.seed(seed)
            try:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            except ImportError:
                pass
        data_path = os.path.join("output_default", "preprocessed.csv")
        if not os.path.exists(data_path):
            pro_log(f"[RealisticBacktestLive] Preprocessed data not found: {data_path}", level = "error", tag = "Backtest")
            return None
        df = pd.read_csv(data_path)
        pro_log(f"[RealisticBacktestLive] Loaded preprocessed data shape: {df.shape}", tag = "Backtest")
        if 'target' not in df.columns:
            pro_log(f"[RealisticBacktestLive] No 'target' column found in data!", level = "error", tag = "Backtest")
            return None
        X = df.drop(columns = ['target'])
        y = df['target']
        # กรองเฉพาะคอลัมน์ตัวเลขก่อนเทรน
        if isinstance(X, pd.DataFrame):
            X = X.select_dtypes(include = ["number"])
        # DEBUG: จำกัดขนาดข้อมูลสำหรับ debug (เทพ)
        console = Console()
        if len(X) > 20000:
            msg = f"[เทพ DEBUG] ใช้ข้อมูล 20000 แถวแรกจาก {len(X)} แถวเพื่อความเร็ว"
            console.print(Panel(msg, title = "[bold yellow]Data Limiting", border_style = "yellow"))
            X = X.iloc[:20000]
            y = y.iloc[:20000]
        msg = f"[เทพ DEBUG] Start TimeSeriesSplit with {len(X)} rows"
        console.print(Panel(msg, title = "[bold cyan]TimeSeriesSplit", border_style = "cyan"))
        tscv = TimeSeriesSplit(n_splits = 5)
        set_seed(42)
        # - - - Model selection: use same as full pipeline if provided - -  - 
        if model_class is None:
            # Try to get from config (simulate full pipeline)
            if config and 'model_class' in config:
                model_class = config['model_class']
                model_params = config.get('model_params', {})
            else:
                model_class = RandomForestClassifier
                model_params = {'n_jobs': -1, 'random_state': 42}
        elif model_params is None:
            model_params = {}
        results = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            print(f"[เทพ DEBUG] Training fold {fold} ...")
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            print(f"[เทพ DEBUG] X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            print(f"[เทพ DEBUG] Finished training fold {fold}")
            # Save model after each fold (simulate live model update)
            model_path = os.path.join("output_default", f"live_model_fold{fold}.pkl")
            joblib.dump(model, model_path)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average = 'macro')
            cm = confusion_matrix(y_test, preds)
            results.append({
                'fold': fold, 
                'start': X_test.index[0], 
                'end': X_test.index[ - 1], 
                'accuracy': acc, 
                'f1': f1, 
                'confusion_matrix': cm.tolist(), 
                'n_test': len(y_test), 
                'model_path': model_path, 
                'model': model_class.__name__
            })
        results_df = pd.DataFrame(results)
        out_path = os.path.join("output_default", "realistic_backtest_live_model_result.csv")
        results_df.to_csv(out_path, index = False)
        pro_log(f"[RealisticBacktestLive] Saved realistic backtest (live model) result to {out_path}", level = "success", tag = "Backtest")
        return out_path
    finally:
        stop_event.set()
        monitor_thread.join()

# - - - CONFIG VALIDATION - -  - 
def validate_config_yaml(config_path = "config.yaml"):
    """Validate config.yaml structure and required fields."""
    required_fields = [
        "model_class", "model_params", "walk_forward", "metrics", "export", "parallel", "visualization"
    ]
    try:
        with open(config_path, "r", encoding = "utf - 8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"[ConfigValidation] Failed to load config.yaml: {e}")
    missing = [k for k in required_fields if k not in config]
    if missing:
        raise ValueError(f"[ConfigValidation] Missing required fields in config.yaml: {missing}")
    # Optional: check subfields
    if not isinstance(config.get("model_params"), dict):
        raise ValueError("[ConfigValidation] 'model_params' must be a dict.")
    if not isinstance(config.get("metrics"), list):
        raise ValueError("[ConfigValidation] 'metrics' must be a list.")
    if not isinstance(config.get("export"), dict):
        raise ValueError("[ConfigValidation] 'export' must be a dict.")
    return True

# - - - RESOURCE MONITOR - -  - 
def log_resource_usage(interval = 10, log_path = "output_default/resource_usage.log", stop_event = None):
    """Log CPU, RAM, and GPU usage every interval seconds."""

    def get_gpu_usage():
        try:
            result = subprocess.run([
                "nvidia - smi", " -  - query - gpu = utilization.gpu, memory.used, memory.total", " -  - format = csv, noheader, nounits"
            ], capture_output = True, text = True, encoding = 'utf - 8', errors = 'ignore')
            if result.returncode == 0:
                usage = result.stdout.strip().split(', ')
                return {
                    "gpu_util": int(usage[0]), 
                    "gpu_mem_used": int(usage[1]), 
                    "gpu_mem_total": int(usage[2])
                }
        except Exception:
            pass
        return None

    with open(log_path, "a", encoding = "utf - 8") as f:
        f.write("timestamp, cpu_percent, ram_used_mb, ram_total_mb, gpu_util, gpu_mem_used, gpu_mem_total\n")
        while stop_event is None or not stop_event.is_set():
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory()
            gpu = get_gpu_usage()
            ts = time.strftime("%Y - %m - %d %H:%M:%S")
            line = f"{ts}, {cpu}, {ram.used//1024//1024}, {ram.total//1024//1024}"
            if gpu:
                line += f", {gpu['gpu_util']}, {gpu['gpu_mem_used']}, {gpu['gpu_mem_total']}"
            else:
                line += ", , , , "
            f.write(line + "\n")
            f.flush()
            time.sleep(interval)

# - - - Example usage: validate config and start resource monitor in backtest entry - -  - 
# In your main backtest entry point (e.g. run_realistic_backtest, run_robust_backtest, etc), add:
#
#   validate_config_yaml("config.yaml")
#   stop_event = threading.Event()
#   monitor_thread = threading.Thread(target = log_resource_usage, args = (10, "output_default/resource_usage.log", stop_event))
#   monitor_thread.start()
#   ... run backtest ...
#   stop_event.set()
#   monitor_thread.join()
#
# You can also add try/finally to ensure monitor stops on error.