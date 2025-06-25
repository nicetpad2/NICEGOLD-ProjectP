# backtest_engine.py
from collections import defaultdict
from src.data_loader.utils import safe_set_datetime
from src.evaluation import find_best_threshold
from src.features import rsi, macd, is_volume_spike, detect_macd_divergence
from src.utils.gc_utils import maybe_collect
    from tqdm import tqdm
from typing import Dict, List, Tuple, Union, Optional
import logging
import math
import numpy as np
import os
import pandas as pd
import random
import time
import traceback
"""
ฟังก์ชันเกี่ยวกับ backtesting, simulation, calculate_metrics, process_active_orders, attempt_order, dynamic_tp2_multiplier ฯลฯ
"""

try:
except ImportError:
    tqdm = None

# นำเข้า Helper functions

# อ่านเวอร์ชันจากไฟล์ VERSION
VERSION_FILE = os.path.join(os.path.dirname(__file__), '..', 'VERSION')
with open(VERSION_FILE, 'r', encoding = 'utf - 8') as vf:
    __version__ = vf.read().strip()

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_SESSION_TIMES_UTC = {"Asia": (22, 8), "London": (7, 16), "NY": (13, 21)}
DEFAULT_BASE_TP_MULTIPLIER = 1.8
DEFAULT_ADAPTIVE_TSL_HIGH_VOL_RATIO = 1.8
DEFAULT_ADAPTIVE_TSL_LOW_VOL_RATIO = 0.75
DEFAULT_ADAPTIVE_TSL_DEFAULT_STEP_R = 0.5
DEFAULT_ADAPTIVE_TSL_HIGH_VOL_STEP_R = 1.0
DEFAULT_ADAPTIVE_TSL_LOW_VOL_STEP_R = 0.3
DEFAULT_ADAPTIVE_TSL_START_ATR_MULT = 1.5
DEFAULT_MIN_SIGNAL_SCORE_ENTRY = 0.6
DEFAULT_MIN_LOT_SIZE = 0.01
DEFAULT_MAX_LOT_SIZE = 5.0
DEFAULT_RISK_PER_TRADE = 0.01
DEFAULT_META_FILTER_THRESHOLD = 0.5
DEFAULT_META_FILTER_RELAXED_THRESHOLD = 0.4

# - - - Backtesting Engine Functions - -  - 

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
    enable_partial_tp = True,  # ใช้ค่า default แทน global
    partial_tp_levels = None, 
    partial_tp_move_sl_to_entry = True, 
    enable_kill_switch = True, 
    kill_switch_max_dd_threshold = 0.15, 
    kill_switch_consecutive_losses_config = 5, 
    recovery_mode_consecutive_losses_config = 4, 
    min_equity_threshold_pct = 0.70, 
    initial_kill_switch_state = False, 
    initial_consecutive_losses = 0, 
):
    """
    Runs the core backtesting simulation loop for a single fold, side, and fund profile.
    (v4.8.8 Patch 26.5.1: Unified error handling, logging, and exit logic fixes)

    ย้ายมาจาก strategy.py เพื่อแยกการทำงานของ backtesting engine ออกมาเป็นโมดูลแยก
    """
    # Import necessary variables and functions that were globals in strategy.py
    # TODO: ต้องนำเข้าตัวแปรและฟังก์ชันที่จำเป็นจาก strategy.py

    # สำหรับตอนนี้ให้ return placeholder values
    logging.warning(f"run_backtest_simulation_v34 ยังไม่ได้ implement เต็มรูปแบบ - returning placeholder values for {label}")

    # Return placeholder values with same structure as original
    empty_df = df_m1_segment_pd.copy() if isinstance(df_m1_segment_pd, pd.DataFrame) else pd.DataFrame()
    empty_trade_log = pd.DataFrame()
    empty_equity_history = {pd.Timestamp.now(): initial_capital_segment}
    empty_run_summary = {
        "error_in_loop": False, 
        "total_commission": 0.0, 
        "total_spread": 0.0, 
        "total_slippage": 0.0, 
        "orders_blocked_dd": 0, 
        "orders_blocked_cooldown": 0, 
        "orders_scaled_lot": 0, 
        "be_sl_triggered_count": 0, 
        "tsl_triggered_count": 0, 
        "orders_skipped_ml_l1": 0, 
        "orders_skipped_ml_l2": 0, 
        "reentry_trades_opened": 0, 
        "forced_entry_trades_opened": 0, 
        "meta_model_type_l1": "N/A", 
        "meta_model_type_l2": "N/A", 
        "threshold_l1_used": 0.25, 
        "threshold_l2_used": np.nan, 
        "kill_switch_activated": False, 
        "forced_entry_disabled_status": False, 
        "orders_blocked_new_v46": 0, 
        "drift_override_active": False, 
        "drift_override_reason": "", 
        "final_risk_mode": "normal", 
        "fund_profile": fund_profile, 
        "total_ib_lot_accumulator": 0.0, 
    }

    return (
        empty_df, 
        empty_trade_log, 
        initial_capital_segment, 
        empty_equity_history, 
        0.0,  # max_drawdown_pct
        empty_run_summary, 
        [],   # blocked_order_log
        "N/A",  # sim_model_type_l1
        "N/A",  # sim_model_type_l2
        initial_kill_switch_state, 
        initial_consecutive_losses, 
        0.0   # total_ib_lot_accumulator
    )


def calculate_metrics(trade_log_df, final_equity, equity_history_segment, initial_capital = None, label = "", model_type_l1 = "N/A", model_type_l2 = "N/A", run_summary = None, ib_lot_accumulator = 0.0):
    """
    คำนวณเมตริกประสิทธิภาพจาก trade log และข้อมูล equity
    ย้ายมาจาก strategy.py เพื่อแยกการทำงานของ backtesting engine
    """
    # สำหรับตอนนี้ให้ return เมตริกพื้นฐาน
    if initial_capital is None:
        initial_capital = 100.0  # default value

    metrics = {
        f"{label} Initial Capital (USD)": initial_capital, 
        f"{label} Final Equity (USD)": final_equity, 
        f"{label} Total Net Profit (USD)": final_equity - initial_capital, 
        f"{label} Return (%)": ((final_equity - initial_capital) / initial_capital) * 100.0 if initial_capital > 0 else 0.0, 
        f"{label} Total Trades (Full)": len(trade_log_df) if trade_log_df is not None and not trade_log_df.empty else 0, 
        f"{label} ML Model Used (L1)": model_type_l1, 
        f"{label} ML Model Used (L2)": model_type_l2, 
    }

    if trade_log_df is not None and not trade_log_df.empty:
        # คำนวณเมตริกเพิ่มเติม
        if "pnl_usd_net" in trade_log_df.columns:
            wins = trade_log_df[trade_log_df["pnl_usd_net"] > 0]
            losses = trade_log_df[trade_log_df["pnl_usd_net"] < 0]

            metrics[f"{label} Total Wins (Full)"] = len(wins)
            metrics[f"{label} Total Losses (Full)"] = len(losses)
            metrics[f"{label} Win Rate (Full) (%)"] = (len(wins) / len(trade_log_df)) * 100.0 if len(trade_log_df) > 0 else 0.0
            metrics[f"{label} Gross Profit (USD)"] = wins["pnl_usd_net"].sum() if len(wins) > 0 else 0.0
            metrics[f"{label} Gross Loss (USD)"] = losses["pnl_usd_net"].sum() if len(losses) > 0 else 0.0

    return metrics


def attempt_order(side: str, price: float, params: dict) -> tuple[bool, list[str]]:
    """
    พยายามส่งคำสั่งซื้อขายตามเงื่อนไขที่กำหนด
    """
    # Implementation placeholder
    return True, []


def dynamic_tp2_multiplier(current_atr, avg_atr, base = None):
    """
    คำนวณค่า dynamic_tp2_multiplier ตามความผันผวนของตลาด
    """
    if base is None:
        base = DEFAULT_BASE_TP_MULTIPLIER

    if pd.isna(current_atr) or pd.isna(avg_atr) or avg_atr <= 0:
        return base

    atr_ratio = current_atr / avg_atr

    if atr_ratio > DEFAULT_ADAPTIVE_TSL_HIGH_VOL_RATIO:
        return base * 0.8  # ลด TP ในตลาดผันผวนสูง
    elif atr_ratio < DEFAULT_ADAPTIVE_TSL_LOW_VOL_RATIO:
        return base * 1.2  # เพิ่ม TP ในตลาดผันผวนต่ำ
    else:
        return base


def get_adaptive_tsl_step(current_atr, avg_atr, default_step = None):
    """
    คำนวณ adaptive trailing stop loss step
    """
    if default_step is None:
        default_step = DEFAULT_ADAPTIVE_TSL_DEFAULT_STEP_R

    if pd.isna(current_atr) or pd.isna(avg_atr) or avg_atr <= 0:
        return default_step

    atr_ratio = current_atr / avg_atr

    if atr_ratio > DEFAULT_ADAPTIVE_TSL_HIGH_VOL_RATIO:
        return DEFAULT_ADAPTIVE_TSL_HIGH_VOL_STEP_R
    elif atr_ratio < DEFAULT_ADAPTIVE_TSL_LOW_VOL_RATIO:
        return DEFAULT_ADAPTIVE_TSL_LOW_VOL_STEP_R
    else:
        return default_step


def get_dynamic_signal_score_entry(df, window = 1000, quantile = 0.7, min_val = 0.5, max_val = 3.0):
    """
    คำนวณ signal score entry แบบ dynamic
    """
    if 'Signal_Score' not in df.columns:
        return pd.Series(index = df.index, data = DEFAULT_MIN_SIGNAL_SCORE_ENTRY)

    signal_scores = pd.to_numeric(df['Signal_Score'], errors = 'coerce')
    rolling_quantile = signal_scores.rolling(window = window, min_periods = 1).quantile(quantile)

    return rolling_quantile.clip(min_val, max_val).fillna(DEFAULT_MIN_SIGNAL_SCORE_ENTRY)


def get_dynamic_signal_score_thresholds(series: pd.Series, window: int = 1000, quantile: float = 0.7, 
                                        min_val: float = 0.5, max_val: float = 3.0) -> np.ndarray:
    """
    คำนวณ signal score thresholds แบบเวกเตอร์
    """
    if series.empty:
        return np.array([DEFAULT_MIN_SIGNAL_SCORE_ENTRY])

    numeric_series = pd.to_numeric(series, errors = 'coerce')
    rolling_quantile = numeric_series.rolling(window = window, min_periods = 1).quantile(quantile)

    return rolling_quantile.clip(min_val, max_val).fillna(DEFAULT_MIN_SIGNAL_SCORE_ENTRY).to_numpy()


def update_tsl_only(order, current_high, current_low, current_atr, avg_atr, atr_multiplier = 1.5):
    """
    อัปเดต trailing stop loss เฉพาะ TSL
    """
    if not isinstance(order, dict):
        return order

    side = order.get("side", "BUY")
    entry_price = order.get("entry_price", 0)
    current_sl = order.get("sl_price", 0)

    if pd.isna(current_atr) or current_atr <= 0:
        return order

    tsl_distance = current_atr * atr_multiplier

    if side == "BUY":
        new_sl = current_high - tsl_distance
        if new_sl > current_sl:
            order["sl_price"] = new_sl
    elif side == "SELL":
        new_sl = current_low + tsl_distance
        if new_sl < current_sl:
            order["sl_price"] = new_sl

    return order


def update_trailing_tp2(order, atr, multiplier):
    """
    อัปเดต trailing TP2
    """
    if not isinstance(order, dict):
        return order

    if pd.isna(atr) or atr <= 0 or pd.isna(multiplier):
        return order

    side = order.get("side", "BUY")
    entry_price = order.get("entry_price", 0)

    # คำนวณ TP2 ใหม่ตาม multiplier
    tp2_distance = atr * multiplier

    if side == "BUY":
        new_tp2 = entry_price + tp2_distance
    else:
        new_tp2 = entry_price - tp2_distance

    order["tp_price"] = new_tp2
    return order


# Helper functions for backtesting
def _resolve_close_index(df, entry_idx, close_time):
    """
    หาตำแหน่ง index ที่ใกล้เคียงที่สุดสำหรับปิดออเดอร์
    """
    try:
        if entry_idx in df.index:
            return entry_idx

        # ถ้าไม่เจอ ให้หา index ที่ใกล้เคียงที่สุด
        if hasattr(df.index, 'get_loc'):
            try:
                loc = df.index.get_loc(close_time, method = 'nearest')
                return df.index[loc]
            except:
                pass

        # ถ้ายังไม่ได้ ให้ใช้ index แรกที่มี
        if not df.empty:
            return df.index[0]

        return None
    except:
        return None


def check_main_exit_conditions(order, row, current_bar_index, now):
    """
    ตรวจสอบเงื่อนไขการปิดออเดอร์หลัก (BE - SL, SL, TP, MaxBars)
    """
    order_closed = False
    exit_price = np.nan
    close_reason = "UNKNOWN"
    close_timestamp = now

    try:
        side = order.get("side", "BUY")
        sl_price = order.get("sl_price", np.nan)
        tp_price = order.get("tp_price", np.nan)
        entry_bar = order.get("entry_bar_count", 0)

        current_high = getattr(row, "High", np.nan)
        current_low = getattr(row, "Low", np.nan)
        current_close = getattr(row, "Close", np.nan)

        # ตรวจสอบ TP
        if pd.notna(tp_price):
            if side == "BUY" and current_high >= tp_price:
                order_closed = True
                exit_price = tp_price
                close_reason = "TP"
            elif side == "SELL" and current_low <= tp_price:
                order_closed = True
                exit_price = tp_price
                close_reason = "TP"

        # ตรวจสอบ SL
        if not order_closed and pd.notna(sl_price):
            if side == "BUY" and current_low <= sl_price:
                order_closed = True
                exit_price = sl_price
                close_reason = "SL"
            elif side == "SELL" and current_high >= sl_price:
                order_closed = True
                exit_price = sl_price
                close_reason = "SL"

        # ตรวจสอบ Max Bars (ถ้ามี)
        max_holding_bars = 1440  # default 24 ชั่วโมง
        if not order_closed and (current_bar_index - entry_bar) >= max_holding_bars:
            order_closed = True
            exit_price = current_close
            close_reason = "MAX_BARS"

    except Exception as e:
        logging.error(f"Error in check_main_exit_conditions: {e}")

    return order_closed, exit_price, close_reason, close_timestamp


# Helper functions สำหรับ position sizing และ risk management
def atr_position_size(equity, atr, risk_pct = 0.01, atr_mult = 2.8, pip_value = 1.0, min_lot = 0.01, max_lot = 5.0):
    """
    คำนวณขนาด position ตาม ATR
    """
    if pd.isna(atr) or atr <= 0 or equity <= 0:
        return min_lot, "INVALID_ATR_OR_EQUITY"

    risk_amount = equity * risk_pct
    sl_distance_points = atr * atr_mult * 10  # แปลงเป็น points

    if sl_distance_points <= 0:
        return min_lot, "INVALID_SL_DISTANCE"

    # คำนวณ lot size
    lot_size = risk_amount / (sl_distance_points * pip_value)
    lot_size = max(min_lot, min(max_lot, lot_size))
    lot_size = round(lot_size, 2)

    return lot_size, "OK"


def compute_dynamic_lot(base_lot, current_dd):
    """
    ปรับ lot size ตาม current drawdown
    """
    if pd.isna(current_dd) or current_dd < 0:
        return base_lot

    # ลด lot เมื่อ drawdown สูง
    if current_dd > 0.10:  # มากกว่า 10%
        return base_lot * 0.5
    elif current_dd > 0.05:  # มากกว่า 5%
        return base_lot * 0.75
    else:
        return base_lot


def adjust_lot_tp2_boost(trade_history, base_lot):
    """
    ปรับ lot size ตามประวัติการเทรด
    """
    if not trade_history or len(trade_history) < 3:
        return base_lot

    # ถ้าชนะ 3 รอบสุดท้าย ให้เพิ่ม lot
    recent_wins = sum(1 for result in trade_history[ - 3:] if "TP" in str(result))
    if recent_wins >= 3:
        return base_lot * 1.2

    return base_lot


def adjust_lot_recovery_mode(base_lot, consecutive_losses):
    """
    ปรับ lot size สำหรับ recovery mode
    """
    if consecutive_losses >= 4:  # Recovery mode
        recovery_multiplier = 1.5
        return base_lot * recovery_multiplier, "recovery"

    return base_lot, "normal"


def adjust_sl_tp_oms(entry_price, sl_price, tp_price, atr, side, margin_pips = 2, max_distance_pips = 200):
    """
    ปรับ SL/TP ตาม OMS requirements
    """
    # ตรวจสอบและปรับ SL/TP ให้อยู่ในระยะที่เหมาะสม
    min_distance = margin_pips / 10.0  # แปลงเป็น price
    max_distance = max_distance_pips / 10.0

    if side == "BUY":
        # ปรับ SL ให้ไม่ใกล้เกินไป
        if (entry_price - sl_price) < min_distance:
            sl_price = entry_price - min_distance
        # ปรับ TP ให้ไม่ไกลเกินไป
        if (tp_price - entry_price) > max_distance:
            tp_price = entry_price + max_distance
    else:  # SELL
        # ปรับ SL ให้ไม่ใกล้เกินไป
        if (sl_price - entry_price) < min_distance:
            sl_price = entry_price + min_distance
        # ปรับ TP ให้ไม่ไกลเกินไป
        if (entry_price - tp_price) > max_distance:
            tp_price = entry_price - max_distance

    return sl_price, tp_price


# Helper classes สำหรับ state management
class CooldownState:
    """
    จัดการ state ของ cooldown
    """
    def __init__(self):
        self.cooldown_bars_remaining = 0
        self.last_drawdown_warning = 0
        self.last_losses_warning = 0
        self.recent_losses = []


def enter_cooldown(cd_state, duration):
    """
    เข้าสู่โหมด cooldown
    """
    cd_state.cooldown_bars_remaining = duration
    return duration


def step_soft_cooldown(current_cooldown):
    """
    ลดค่า cooldown ลง 1 bar
    """
    return max(0, current_cooldown - 1)


def is_soft_cooldown_triggered(trade_pnls, lookback, loss_count, trade_sides, current_side):
    """
    ตรวจสอบว่าควรเข้า soft cooldown หรือไม่
    """
    if len(trade_pnls) < lookback:
        return False, 0

    recent_trades = trade_pnls[ - lookback:]
    recent_sides = trade_sides[ - lookback:] if len(trade_sides) >= lookback else trade_sides

    # นับการขาดทุนในช่วงที่ระบุ
    losses_count = sum(1 for pnl in recent_trades if pnl < 0)

    # นับการขาดทุนสำหรับ side เดียวกัน
    same_side_losses = sum(1 for i, pnl in enumerate(recent_trades)
                          if pnl < 0 and i < len(recent_sides) and recent_sides[i] == current_side)

    return losses_count >= loss_count, same_side_losses


def update_losses(cd_state, pnl):
    """
    อัปเดตรายการการขาดทุนล่าสุด
    """
    cd_state.recent_losses.append(pnl < 0)
    if len(cd_state.recent_losses) > 20:  # เก็บแค่ 20 รายการล่าสุด
        cd_state.recent_losses.pop(0)


def update_drawdown(cd_state, current_dd):
    """
    อัปเดต drawdown state
    """
    # อัปเดต state สำหรับ warning
    pass


def should_warn_drawdown(cd_state, threshold):
    """
    ตรวจสอบว่าควรแจ้งเตือน drawdown หรือไม่
    """
    return False  # Placeholder


def should_warn_losses(cd_state, threshold):
    """
    ตรวจสอบว่าควรแจ้งเตือนการขาดทุนติดต่อกันหรือไม่
    """
    return False  # Placeholder


# Entry/Exit condition helpers
def is_entry_allowed(row, session, consecutive_losses, side, trend, signal_score_threshold = 0.6):
    """
    ตรวจสอบว่าสามารถเปิดออเดอร์ได้หรือไม่
    """
    try:
        # ตรวจสอบ signal score
        signal_score = getattr(row, "Signal_Score", 0)
        if pd.isna(signal_score) or abs(signal_score) < signal_score_threshold:
            return False, "LOW_SIGNAL_SCORE"

        # ตรวจสอบ session
        if session not in ["London", "NY", "Asia"]:
            return False, "INVALID_SESSION"

        # ตรวจสอบการขาดทุนติดต่อกัน
        if consecutive_losses >= 5:
            return False, "HIGH_CONSECUTIVE_LOSSES"

        return True, "OK"

    except Exception as e:
        logging.error(f"Error in is_entry_allowed: {e}")
        return False, "ERROR"


def check_forced_trigger(bars_since_last, signal_score):
    """
    ตรวจสอบเงื่อนไข forced entry
    """
    forced_triggered = (
        bars_since_last >= 100 and  # รอนานพอ
        pd.notna(signal_score) and
        abs(signal_score) >= 2.0  # signal แรงพอ
    )

    return forced_triggered, {
        "bars_since_last": bars_since_last, 
        "score": signal_score
    }


def predict_with_time_check(model, X, current_time):
    """
    ทำนายด้วย model พร้อมตรวจสอบเวลา
    """
    try:
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)[0, 1]
        elif hasattr(model, 'predict'):
            return model.predict(X)[0]
        else:
            return 0.5  # default probability
    except Exception as e:
        logging.error(f"Error in predict_with_time_check: {e}")
        return 0.5


def _update_open_order_state(order, current_high, current_low, current_atr, avg_atr, now, 
                           be_r_thresh, sl_mult, tp_mult, be_count, tsl_count):
    """
    อัปเดต state ของออเดอร์ที่เปิดอยู่ (BE, TSL, TTP2)
    """
    be_triggered_this_bar = False
    tsl_updated_this_bar = False

    try:
        # Update breakeven
        if not order.get("be_triggered", False):
            # ตรวจสอบเงื่อนไข breakeven
            entry_price = order.get("entry_price", 0)
            side = order.get("side", "BUY")

            if side == "BUY" and current_high >= (entry_price + current_atr * be_r_thresh):
                order["be_triggered"] = True
                order["be_triggered_time"] = now
                order["sl_price"] = entry_price  # ย้าย SL ไป entry
                be_triggered_this_bar = True
                be_count += 1
            elif side == "SELL" and current_low <= (entry_price - current_atr * be_r_thresh):
                order["be_triggered"] = True
                order["be_triggered_time"] = now
                order["sl_price"] = entry_price  # ย้าย SL ไป entry
                be_triggered_this_bar = True
                be_count += 1

        # Update trailing stop loss
        if order.get("tsl_activated", False):
            order = update_tsl_only(order, current_high, current_low, current_atr, avg_atr)
            tsl_updated_this_bar = True
            tsl_count += 1

        # Update trailing TP2
        if order.get("use_trailing_for_tp2", False):
            multiplier = dynamic_tp2_multiplier(current_atr, avg_atr, tp_mult)
            order = update_trailing_tp2(order, current_atr, multiplier)

    except Exception as e:
        logging.error(f"Error in _update_open_order_state: {e}")

    return order, be_triggered_this_bar, tsl_updated_this_bar, be_count, tsl_count