from pandas import Series
from pandas import Timestamp
from typing import Any
import logging
import math
import numpy as np
import pandas as pd
"""
Exit logic module (เทพ) - แยก logic exit ออกจาก strategy.py
"""

def check_main_exit_conditions(
    order: dict, row: Series, current_bar_index: int, now_timestamp: Timestamp
) -> tuple[bool, float, str, Timestamp]:
    """
    Checks exit conditions for an order in strict priority: BE - SL -> SL -> TP -> MaxBars.
    Uses tolerance for price checks.

    Args:
        order (dict): The active order dictionary.
        row (pd.Series): The current market data row (OHLC).
        current_bar_index (int): The index of the current bar.
        now_timestamp (pd.Timestamp): The timestamp of the current bar.

    Returns:
        tuple: (order_closed_this_bar, exit_price, close_reason, close_timestamp)
    """
    order_closed_this_bar = False
    exit_price_final = float('nan')
    close_reason_final = "UNKNOWN_EXIT"  # Default if no condition met
    close_timestamp_final = now_timestamp

    side = order.get("side")
    sl_price_order = float(order.get("sl_price", float('nan')))
    tp_price_order = float(order.get("tp_price", float('nan')))
    entry_price_order = float(order.get("entry_price", float('nan')))

    current_high = float(getattr(row, "High", float('nan')))
    current_low = float(getattr(row, "Low", float('nan')))
    current_close = float(getattr(row, "Close", float('nan')))
    be_triggered = order.get('be_triggered', False)
    entry_bar_count_order = order.get("entry_bar_count")
    entry_time_log = order.get('entry_time', 'N/A')  # For logging

    price_tolerance = 0.05

    sl_text = f"{sl_price_order:.5f}" if not math.isnan(sl_price_order) else "NaN"
    tp_text = f"{tp_price_order:.5f}" if not math.isnan(tp_price_order) else "NaN"
    logging.debug(
        f"            [Exit Check V2.1] Order {entry_time_log} "
        f"Side: {side}, SL: {sl_text}, TP: {tp_text}, BE: {be_triggered}"
    )
    logging.debug(f"            [Exit Check V2.1] Bar Prices: H = {current_high:.5f}, L = {current_low:.5f}, C = {current_close:.5f}")

    if be_triggered and not math.isnan(sl_price_order) and not math.isnan(entry_price_order) and math.isclose(sl_price_order, entry_price_order, abs_tol = price_tolerance):
        if side == 'BUY' and (current_low <= sl_price_order + price_tolerance):
            order_closed_this_bar = True; close_reason_final = 'BE - SL'; exit_price_final = sl_price_order
            logging.info(f"               [Patch B Check] BE - SL HIT (BUY). Order {entry_time_log}. Exit Price: {exit_price_final:.5f}")
        elif side == 'SELL' and (current_high >= sl_price_order - price_tolerance):
            order_closed_this_bar = True; close_reason_final = 'BE - SL'; exit_price_final = sl_price_order
            logging.info(f"               [Patch B Check] BE - SL HIT (SELL). Order {entry_time_log}. Exit Price: {exit_price_final:.5f}")
    # ... (ต่อยอด logic exit อื่น ๆ ตาม strategy.py)
    return order_closed_this_bar, exit_price_final, close_reason_final, close_timestamp_final