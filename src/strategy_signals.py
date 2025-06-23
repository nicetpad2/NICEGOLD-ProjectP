import pandas as pd
import numpy as np
from src.config import USE_MACD_SIGNALS, USE_RSI_SIGNALS
from src.signal_utils import (
    generate_open_signals as _generate_open_impl,
    generate_close_signals as _generate_close_impl,
    precompute_sl_array as _precompute_sl_impl,
    precompute_tp_array as _precompute_tp_impl,
)

def generate_open_signals(
    df: pd.DataFrame,
    use_macd: bool = USE_MACD_SIGNALS,
    use_rsi: bool = USE_RSI_SIGNALS,
    trend: str | None = None,
    ma_fast: int = 15,
    ma_slow: int = 50,
    volume_col: str = "Volume",
    vol_window: int = 10,
) -> np.ndarray:
    """สร้างสัญญาณเปิด order พร้อมตัวเลือกเปิด/ปิด MACD และ RSI"""
    return _generate_open_impl(
        df,
        use_macd=use_macd,
        use_rsi=use_rsi,
        trend=trend,
        ma_fast=ma_fast,
        ma_slow=ma_slow,
        volume_col=volume_col,
        vol_window=vol_window,
    )

def generate_close_signals(
    df: pd.DataFrame,
    use_macd: bool = USE_MACD_SIGNALS,
    use_rsi: bool = USE_RSI_SIGNALS,
) -> np.ndarray:
    """สร้างสัญญาณปิด order พร้อมตัวเลือกเปิด/ปิด MACD และ RSI"""
    return _generate_close_impl(df, use_macd=use_macd, use_rsi=use_rsi)

def precompute_sl_array(df: pd.DataFrame) -> np.ndarray:
    """คำนวณ Stop-Loss ล่วงหน้า"""
    return _precompute_sl_impl(df)

def precompute_tp_array(df: pd.DataFrame) -> np.ndarray:
    """คำนวณ Take-Profit ล่วงหน้า"""
    return _precompute_tp_impl(df)
