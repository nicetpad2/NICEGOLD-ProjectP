    from src.config import USE_MACD_SIGNALS, USE_RSI_SIGNALS
from src.features import macd, rsi, atr
import numpy as np
import pandas as pd
try:
except Exception:  # pragma: no cover - fallback for missing config
    USE_MACD_SIGNALS = True
    USE_RSI_SIGNALS = True


def generate_close_signals(
    df: pd.DataFrame, 
    use_macd: bool = USE_MACD_SIGNALS, 
    use_rsi: bool = USE_RSI_SIGNALS, 
) -> np.ndarray:
    """สร้างสัญญาณปิด order พร้อมตัวเลือกเปิด/ปิด MACD และ RSI"""
    close_mask = df["Close"] < df["Close"].shift(1)
    if use_macd:
        if "MACD_hist" not in df.columns:
            _, _, macd_hist = macd(df["Close"])
            df = df.copy()
            df["MACD_hist"] = macd_hist
        close_mask &= df["MACD_hist"] < 0
    if use_rsi:
        if "RSI" not in df.columns:
            df = df.copy()
            df["RSI"] = rsi(df["Close"])
        close_mask &= df["RSI"] < 50
    return close_mask.fillna(0).astype(np.int8).to_numpy()


def precompute_sl_array(df: pd.DataFrame) -> np.ndarray:
    """คำนวณ Stop - Loss ล่วงหน้าตาม ATR"""
    if "ATR_14" not in df.columns:
        df = atr(df, 14)

    atr_series = pd.to_numeric(df.get("ATR_14"), errors = "coerce")
    if atr_series.isna().all() and {"High", "Low"}.issubset(df.columns):
        # [Patch v6.8.12] Fallback ATR using High - Low when ta library returns NaN
        hl_range = pd.to_numeric(df["High"], errors = "coerce") - pd.to_numeric(df["Low"], errors = "coerce")
        atr_series = hl_range.ewm(alpha = 1/14, adjust = False, min_periods = 1).mean()

    sl = atr_series * 1.5
    return sl.fillna(0.0).to_numpy(dtype = np.float64)


def precompute_tp_array(df: pd.DataFrame) -> np.ndarray:
    """คำนวณ Take - Profit ล่วงหน้าตาม ATR"""
    if "ATR_14" not in df.columns:
        df = atr(df, 14)

    atr_series = pd.to_numeric(df.get("ATR_14"), errors = "coerce")
    if atr_series.isna().all() and {"High", "Low"}.issubset(df.columns):
        # [Patch v6.8.12] Fallback ATR using High - Low when ta library returns NaN
        hl_range = pd.to_numeric(df["High"], errors = "coerce") - pd.to_numeric(df["Low"], errors = "coerce")
        atr_series = hl_range.ewm(alpha = 1/14, adjust = False, min_periods = 1).mean()

    tp = atr_series * 3.0
    return tp.fillna(0.0).to_numpy(dtype = np.float64)