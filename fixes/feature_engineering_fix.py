
# Fixed Feature Engineering
import pandas as pd
import numpy as np

def create_high_predictive_features(df):
    """
    สร้างฟีเจอร์ที่มี predictive power สูง
    """
    features = df.copy()
    
    # 1. Multi-scale momentum (ที่สำคัญมาก)
    for period in [3, 5, 8, 13, 21]:
        features[f"momentum_{period}"] = (
            features["Close"] / features["Close"].shift(period) - 1
        )
        
    # 2. Volatility-adjusted price position
    for window in [10, 20]:
        rolling_mean = features["Close"].rolling(window).mean()
        rolling_std = features["Close"].rolling(window).std()
        features[f"price_zscore_{window}"] = (
            (features["Close"] - rolling_mean) / (rolling_std + 1e-8)
        )
        
    # 3. Market microstructure
    features["hl_ratio"] = (features["High"] - features["Low"]) / features["Close"]
    features["co_ratio"] = (features["Close"] - features["Open"]) / features["Close"]
    
    # 4. Trend consistency
    for window in [5, 10]:
        price_changes = features["Close"].diff()
        features[f"trend_consistency_{window}"] = (
            price_changes.rolling(window).apply(
                lambda x: (x > 0).sum() / len(x)
            )
        )
        
    # 5. Acceleration
    momentum_5 = features["Close"] / features["Close"].shift(5) - 1
    features["momentum_acceleration"] = momentum_5.diff()
    
    # 6. Cross-timeframe alignment
    sma_5 = features["Close"].rolling(5).mean()
    sma_20 = features["Close"].rolling(20).mean()
    features["sma_alignment"] = (sma_5 > sma_20).astype(int)
    
    # 7. Volatility regime
    vol_short = features["Close"].pct_change().rolling(5).std()
    vol_long = features["Close"].pct_change().rolling(20).std()
    features["vol_regime"] = vol_short / (vol_long + 1e-8)
    
    # 8. Support/Resistance proximity
    features["high_5"] = features["High"].rolling(5).max()
    features["low_5"] = features["Low"].rolling(5).min()
    features["resistance_distance"] = (features["high_5"] - features["Close"]) / features["Close"]
    features["support_distance"] = (features["Close"] - features["low_5"]) / features["Close"]
    
    return features

# Usage:
# enhanced_df = create_high_predictive_features(df)
