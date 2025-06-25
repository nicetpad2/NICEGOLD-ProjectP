from itertools import product
from src.config import (
from src.config import OUTPUT_DIR as CFG_OUTPUT_DIR
from src.cooldown_utils import (
from src.features import reset_indicator_caches
from src.log_analysis import summarize_block_reasons
from src.utils import get_env_float, load_json_with_comments
from src.utils.leakage import assert_no_overlap
from src.utils.model_utils import predict_with_time_check
from src.utils.sessions import get_session_tag
from tqdm import tqdm
from typing import Dict, List, Tuple
    import itertools
import json
import logging
import math
import numpy as np
import os
import pandas as pd
import time
"""Main strategy logic, ML, training, entry/exit, risk management, etc."""

# TODO: ย้ายฟังก์ชันหลักเกี่ยวกับกลยุทธ์, ML, train, backtest, entry/exit, risk management มาที่นี่


# = =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
# = = = Model Training Function (moved from strategy.py) = =  = 
# = =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 


    ENABLE_KILL_SWITCH, 
    ENABLE_PARTIAL_TP, 
    ENABLE_SPIKE_GUARD, 
    KILL_SWITCH_CONSECUTIVE_LOSSES_THRESHOLD, 
    KILL_SWITCH_MAX_DD_THRESHOLD, 
    M15_TREND_ALLOWED, 
    MIN_SIGNAL_SCORE_ENTRY, 
)
    PARTIAL_TP_LEVELS, 
    PARTIAL_TP_MOVE_SL_TO_ENTRY, 
    RECOVERY_MODE_CONSECUTIVE_LOSSES, 
    USE_MACD_SIGNALS, 
    USE_RSI_SIGNALS, 
    min_equity_threshold_pct, 
)
    CooldownState, 
    enter_cooldown, 
    is_soft_cooldown_triggered, 
    should_enter_cooldown, 
    should_warn_drawdown, 
    should_warn_losses, 
    step_soft_cooldown, 
    update_drawdown, 
    update_losses, 
)


def train_and_export_meta_model(
    trade_log_path = "trade_log_v32_walkforward.csv", 
    m1_data_path = "final_data_m1_v32_walkforward.csv", 
    output_dir = None, 
    model_purpose = "main", 
    trade_log_df_override = None, 
    model_type_to_train = "catboost", 
    link_model_as_default = "catboost", 
    enable_dynamic_feature_selection = True, 
    feature_selection_method = "shap", 
    shap_importance_threshold = 0.01, 
    permutation_importance_threshold = 0.001, 
    prelim_model_params = None, 
    enable_optuna_tuning = True, 
    optuna_n_trials = 50, 
    optuna_cv_splits = 5, 
    optuna_metric = "AUC", 
    optuna_direction = "maximize", 
    drift_observer = None, 
    catboost_gpu_ram_part = 0.95, 
    optuna_n_jobs = -1, 
    sample_size = None, 
    features_to_drop_before_train = None, 
    early_stopping_rounds = 200, 
    enable_threshold_tuning = False, 
    fold_index = None, 
):
    """
    Trains and exports a Meta Classifier (L1) model for a specific purpose
    (main, spike, cluster) using trade log data and M1 features. Includes options
    for dynamic feature selection and hyperparameter optimization (Optuna).
    (v4.8.8 Patch 2: Fixed UnboundLocalError, Pool usage)
    """
    # ...function body moved from strategy.py...
    pass


def run_hyperparameter_sweep(base_params: dict, grid: dict, train_func):
    """รันการค้นหา Hyperparameter แบบ grid search และพิมพ์ผลลัพธ์ทันที"""

    keys = list(grid.keys())
    values = list(grid.values())
    combinations = list(itertools.product(*values))
    output_dir = base_params.get("output_dir")
    if output_dir:
        os.makedirs(output_dir, exist_ok = True)
    results = []
    for idx, combo in enumerate(combinations, start = 1):
        params = base_params.copy()
        for k, v in zip(keys, combo):
            params[k] = v
        logging.info("เริ่มพารามิเตอร์ run %s: %s", idx, params)
        model_path, feature_list = train_func(**params)
        result_entry = {
            "params": params, 
            "model_path": model_path, 
            "features": feature_list, 
        }
        logging.info("Run %s: %s", idx, result_entry)
        results.append(result_entry)
    return results


def spike_guard_london(row: pd.Series, session: str, consecutive_losses: int) -> bool:
    """Spike guard filter for London session with debug reasons."""

    if not ENABLE_SPIKE_GUARD:
        logging.debug("      (Spike Guard) Disabled via config.")
        return True
    if not isinstance(session, str) or "London" not in session:
        logging.debug("      (Spike Guard) Not London session - skipping.")
        return True

    spike_score_val = pd.to_numeric(
        getattr(row, "spike_score", np.nan), errors = "coerce"
    )
    if pd.notna(spike_score_val) and spike_score_val > 0.85:
        logging.debug(
            "      (Spike Guard Filtered) Reason: London Session & High Spike Score (%.2f > 0.85)", 
            spike_score_val, 
        )
        return False

    adx_val = pd.to_numeric(getattr(row, "ADX", np.nan), errors = "coerce")
    wick_ratio_val = pd.to_numeric(getattr(row, "Wick_Ratio", np.nan), errors = "coerce")
    vol_index_val = pd.to_numeric(
        getattr(row, "Volatility_Index", np.nan), errors = "coerce"
    )
    candle_body_val = pd.to_numeric(
        getattr(row, "Candle_Body", np.nan), errors = "coerce"
    )
    candle_range_val = pd.to_numeric(
        getattr(row, "Candle_Range", np.nan), errors = "coerce"
    )
    gain_val = pd.to_numeric(getattr(row, "Gain", np.nan), errors = "coerce")
    atr_val = pd.to_numeric(getattr(row, "ATR_14", np.nan), errors = "coerce")

    if any(
        pd.isna(v)
        for v in [
            adx_val, 
            wick_ratio_val, 
            vol_index_val, 
            candle_body_val, 
            candle_range_val, 
            gain_val, 
            atr_val, 
        ]
    ):
        logging.debug("      (Spike Guard) Missing values - skip filter.")
        return True

    safe_candle_range_val = max(candle_range_val, 1e - 9)

    if adx_val < 20 and wick_ratio_val > 0.7 and vol_index_val < 0.8:
        logging.debug(
            "      (Spike Guard Filtered) Reason: Low ADX(%.1f), High Wick(%.2f), Low Vol(%.2f)", 
            adx_val, 
            wick_ratio_val, 
            vol_index_val, 
        )
        return False

    try:
        body_ratio = candle_body_val / safe_candle_range_val
        if body_ratio < 0.07:
            logging.debug(
                "      (Spike Guard Filtered) Reason: Low Body Ratio(%.3f)", body_ratio
            )
            return False
    except ZeroDivisionError:
        logging.warning("      (Spike Guard) ZeroDivisionError calculating body_ratio.")
        return False

    if gain_val > 3 and atr_val > 4 and (candle_body_val / safe_candle_range_val) > 0.3:
        logging.debug(
            "      (Spike Guard Allowed) Reason: Strong directional move override."
        )
        return True

    logging.debug("      (Spike Guard) Passed all checks.")
    return True


def is_mtf_trend_confirmed(m15_trend: str | None, side: str) -> bool:
    """Validate entry direction using M15 trend zone."""

    trend = str(m15_trend).upper() if isinstance(m15_trend, str) else "NEUTRAL"
    if side == "BUY" and trend not in M15_TREND_ALLOWED:
        return False
    if side == "SELL" and trend not in ["DOWN", "NEUTRAL"]:
        return False
    return True


def passes_volatility_filter(vol_index: float, min_ratio: float = 1.0) -> bool:
    """Return True if Volatility_Index >= min_ratio."""
    vol_val = pd.to_numeric(vol_index, errors = "coerce")
    if pd.isna(vol_val):
        return False
    return vol_val >= min_ratio


def is_entry_allowed(
    row: pd.Series, 
    session: str, 
    consecutive_losses: int, 
    side: str, 
    m15_trend: str | None = None, 
    signal_score_threshold: float | None = None, 
) -> Tuple[bool, str]:
    """Checks if entry is allowed based on filters with debug logging."""

    if signal_score_threshold is None:
        signal_score_threshold = MIN_SIGNAL_SCORE_ENTRY

    if not spike_guard_london(row, session, consecutive_losses):
        logging.debug("      Entry blocked by Spike Guard.")
        return False, "SPIKE_GUARD_LONDON"

    if not is_mtf_trend_confirmed(m15_trend, side):
        logging.debug("      Entry blocked by M15 Trend filter.")
        return False, f"M15_TREND_{str(m15_trend).upper()}"

    vol_index_val = pd.to_numeric(
        getattr(row, "Volatility_Index", np.nan), errors = "coerce"
    )
    if not passes_volatility_filter(vol_index_val):
        logging.debug("      Entry blocked by Low Volatility (%s)", vol_index_val)
        return False, f"LOW_VOLATILITY({vol_index_val})"

    signal_score = pd.to_numeric(getattr(row, "Signal_Score", np.nan), errors = "coerce")
    if pd.isna(signal_score):
        logging.debug("      Entry blocked: Invalid Signal Score (NaN)")
        return False, "INVALID_SIGNAL_SCORE (NaN)"
    if abs(signal_score) < signal_score_threshold:
        logging.debug(
            "      Entry blocked: Low Signal Score %.2f < %.2f", 
            signal_score, 
            signal_score_threshold, 
        )
        return False, f"LOW_SIGNAL_SCORE ({signal_score:.2f}<{signal_score_threshold})"

    logging.debug("      Entry allowed by filters.")
    return True, "ALLOWED"


def adjust_sl_tp_oms(
    entry_price: float, 
    sl_price: float, 
    tp_price: float, 
    atr: float, 
    side: str, 
    margin_pips: float, 
    max_pips: float, 
) -> tuple[float, float]:
    """Validate SL/TP distance and auto - adjust if outside allowed range."""
    if any(pd.isna(v) for v in [entry_price, sl_price, tp_price, atr]):
        return sl_price, tp_price

    sl_dist = abs(entry_price - sl_price) * 10.0
    tp_dist = abs(tp_price - entry_price) * 10.0

    if sl_dist < margin_pips:
        adj = atr if pd.notna(atr) and atr > 1e - 9 else margin_pips / 10.0
        sl_price = entry_price - adj if side == "BUY" else entry_price + adj
        logging.info("[OMS_Guardian] Adjust SL to margin level: %.5f", sl_price)

    if sl_dist > max_pips:
        sl_price = entry_price - atr if side == "BUY" else entry_price + atr
        logging.info("[OMS_Guardian] SL distance too wide. Adjusted to %.5f", sl_price)

    if tp_dist > max_pips:
        tp_price = entry_price + atr if side == "BUY" else entry_price - atr
        logging.info("[OMS_Guardian] TP distance too wide. Adjusted to %.5f", tp_price)

    return sl_price, tp_price


def update_breakeven_half_tp(
    order: Dict, 
    current_high: float, 
    current_low: float, 
    now: pd.Timestamp, 
    entry_buffer: float = 0.0001, 
) -> tuple[Dict, bool]:
    """Move SL to breakeven when price moves halfway to TP1."""
    if order.get("be_triggered", False):
        return order, False

    side = order.get("side")
    entry = pd.to_numeric(order.get("entry_price"), errors = "coerce")
    tp1 = pd.to_numeric(order.get("tp1_price"), errors = "coerce")
    sl = pd.to_numeric(order.get("sl_price"), errors = "coerce")

    if any(pd.isna(v) for v in [side, entry, tp1, sl]):
        return order, False

    trigger = (
        entry + 0.5 * (tp1 - entry) if side == "BUY" else entry - 0.5 * (entry - tp1)
    )
    hit = (side == "BUY" and current_high >= trigger) or (
        side == "SELL" and current_low <= trigger
    )

    if hit:
        new_sl = entry + entry_buffer if side == "BUY" else entry - entry_buffer
        if not math.isclose(new_sl, sl, rel_tol = 1e - 9, abs_tol = 1e - 9):
            order["sl_price"] = new_sl
            order["be_triggered"] = True
            order["be_triggered_time"] = now
            logging.info("Move to Breakeven at price %.5f", new_sl)
            return order, True

    return order, False


def run_backtest_simulation_v34(
    df_m1_segment_pd, 
    label, 
    initial_capital_segment, 
    side = "BUY", 
    fund_profile = None, 
    fold_config = None, 
    available_models = None, 
    model_switcher_func = None, 
    pattern_label_map = None, 
    meta_min_proba_thresh_override = None, 
    current_fold_index = None, 
    enable_partial_tp = ENABLE_PARTIAL_TP, 
    partial_tp_levels = PARTIAL_TP_LEVELS, 
    partial_tp_move_sl_to_entry = PARTIAL_TP_MOVE_SL_TO_ENTRY, 
    enable_kill_switch = ENABLE_KILL_SWITCH, 
    kill_switch_max_dd_threshold = KILL_SWITCH_MAX_DD_THRESHOLD, 
    kill_switch_consecutive_losses_config = KILL_SWITCH_CONSECUTIVE_LOSSES_THRESHOLD, 
    recovery_mode_consecutive_losses_config = RECOVERY_MODE_CONSECUTIVE_LOSSES, 
    min_equity_threshold_pct = min_equity_threshold_pct, 
    initial_kill_switch_state = False, 
    initial_consecutive_losses = 0, 
):
    """
    Runs the core backtesting simulation loop for a single fold, side, and fund profile.
    (v4.8.8 Patch 26.5.1: Unified error handling, logging, and exit logic fixes)
    """
    # ...existing function body from src/strategy.py lines 1842 - 2042...
    pass