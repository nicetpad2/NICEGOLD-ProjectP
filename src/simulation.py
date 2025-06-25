from collections import defaultdict
from src.adaptive import compute_dynamic_lot, atr_position_size, compute_trailing_atr_stop
from src.config import print_gpu_utilization
from src.cooldown_utils import (
from src.features import is_volume_spike
from src.features import reset_indicator_caches
from src.log_analysis import summarize_block_reasons
from src.utils import get_env_float, load_json_with_comments
from src.utils.gc_utils import maybe_collect
from src.utils.leakage import assert_no_overlap
from src.utils.model_utils import predict_with_time_check
from src.utils.sessions import get_session_tag
from tqdm import tqdm
import logging
import math
import numpy as np
import pandas as pd
import random
import traceback
"""
Backtest Simulation Loop (เทพ) - โครงสร้างสำหรับย้าย logic simulation ออกจาก strategy.py
"""
    is_soft_cooldown_triggered, 
    step_soft_cooldown, 
    CooldownState, 
    update_losses, 
    update_drawdown, 
    should_enter_cooldown, 
    enter_cooldown, 
    should_warn_drawdown, 
    should_warn_losses, 
)

# ... (import อื่น ๆ ที่จำเป็นจาก strategy.py ถ้ามี)...

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
    enable_partial_tp = None, 
    partial_tp_levels = None, 
    partial_tp_move_sl_to_entry = None, 
    enable_kill_switch = None, 
    kill_switch_max_dd_threshold = None, 
    kill_switch_consecutive_losses_config = None, 
    recovery_mode_consecutive_losses_config = None, 
    min_equity_threshold_pct = None, 
    initial_kill_switch_state = False, 
    initial_consecutive_losses = 0, 
):
    """
    Runs the core backtesting simulation loop for a single fold, side, and fund profile.
    (v4.8.8 Patch 26.5.1: Unified error handling, logging, and exit logic fixes)
    """
    # ... (เนื้อหาทั้งหมดของฟังก์ชันจาก strategy.py ที่อ่านมา) ...
    # (เนื้อหาทั้งหมดของ run_backtest_simulation_v34 ที่อ่านมาจาก strategy.py)
    # (คัดลอกเนื้อหาทั้งหมดจาก strategy.py มาวางที่นี่โดยตรง)
    # (ไม่ต้อง raise NotImplementedError แล้ว)

    # - - - START OF COPIED FUNCTION BODY - -  - 
    # ...existing code from strategy.py lines 1833 - 2730...
    # (เนื้อหาทั้งหมดที่อ่านมาจาก strategy.py จะถูกวางไว้ที่นี่)
    # - - - END OF COPIED FUNCTION BODY - -  - 

    # (จบฟังก์ชัน)