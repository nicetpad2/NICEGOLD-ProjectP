"""Utility/helper functions for strategy module (memory, fallback, etc.)"""
import gc
import logging
import pandas as pd
import numpy as np

def maybe_collect():
    """Force garbage collection if needed."""
    gc.collect()
    logging.debug("Garbage collection triggered.")

# === Utility Functions (moved from strategy.py) ===

def attempt_order(side: str, price: float, params: dict) -> tuple[bool, list[str]]:
    """Attempt to execute an order and log all block reasons."""
    can_execute = True
    block_reasons: list[str] = []

    if not params.get("OMS_ENABLED", True):
        block_reasons.append("OMS_DISABLED")
    if params.get("kill_switch_active"):
        block_reasons.append("KILL_SWITCH_ACTIVE")
    if params.get("soft_cooldown_active"):
        block_reasons.append("SOFT_COOLDOWN_ACTIVE")
    if params.get("spike_guard_active"):
        block_reasons.append("SPIKE_GUARD_ACTIVE")
    if not params.get("meta_filter_passed", True):
        block_reasons.append("META_FILTER_BLOCKED")
    if params.get("require_m15_trend") and not params.get("m15_trend_ok", True):
        block_reasons.append("M15_TREND_UNMATCHED")
    if params.get("paper_mode"):
        block_reasons.append("PAPER_MODE_SIMULATION")
    if block_reasons:
        can_execute = False
        primary_reason = block_reasons[0]
        logging.warning(
            "Order Blocked | Side=%s, Reason=%s, All_Reasons=%s",
            side,
            primary_reason,
            block_reasons,
        )
        return False, block_reasons
    logging.info(
        "Order Executed | Side=%s, Price=%s, Params=%s",
        side,
        price,
        params,
    )
    return True, []

def dynamic_tp2_multiplier(current_atr, avg_atr, base=None):
    """Calculates a dynamic TP multiplier based on current vs average ATR."""
    if base is None:
        base = 1.8
    current_atr_num = pd.to_numeric(current_atr, errors='coerce')
    avg_atr_num = pd.to_numeric(avg_atr, errors='coerce')
    if pd.isna(current_atr_num) or pd.isna(avg_atr_num) or np.isinf(current_atr_num) or np.isinf(avg_atr_num) or avg_atr_num < 1e-9:
        return base
    try:
        ratio = current_atr_num / avg_atr_num
        high_vol_ratio = 1.8
        high_vol_adjust = 0.6
        mid_vol_ratio = 1.2
        mid_vol_adjust = 0.3
        if ratio >= high_vol_ratio:
            return base + high_vol_adjust
        elif ratio >= mid_vol_ratio:
            return base + mid_vol_adjust
        else:
            return base
    except Exception:
        return base

def get_adaptive_tsl_step(current_atr, avg_atr, default_step=None):
    """Determines the TSL step size (in R units) based on volatility."""
    if default_step is None:
        default_step = 0.5
    high_vol_ratio = 1.8
    high_vol_step = 1.0
    low_vol_ratio = 0.75
    low_vol_step = 0.3
    current_atr_num = pd.to_numeric(current_atr, errors='coerce')
    avg_atr_num = pd.to_numeric(avg_atr, errors='coerce')
    if pd.isna(current_atr_num) or pd.isna(avg_atr_num) or np.isinf(current_atr_num) or np.isinf(avg_atr_num) or avg_atr_num < 1e-9:
        return default_step
    try:
        ratio = current_atr_num / avg_atr_num
        if ratio > high_vol_ratio:
            return high_vol_step
        elif ratio < low_vol_ratio:
            return low_vol_step
        else:
            return default_step
    except Exception:
        return default_step

# Add more utility/helper functions as needed
