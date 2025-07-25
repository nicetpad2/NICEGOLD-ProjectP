
import os
import pandas as pd
import src.features as features
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_rsi_cache_reuse():
    series1 = pd.Series(range(20), dtype = 'float32')
    features.rsi(series1, period = 14)
    obj_first = features._rsi_cache.get(14)
    series2 = pd.Series(range(20, 40), dtype = 'float32')
    features.rsi(series2, period = 14)
    obj_second = features._rsi_cache.get(14)
    assert obj_first is obj_second


def test_atr_cache_reuse():
    df1 = pd.DataFrame({
        'High': pd.Series(range(1, 16), dtype = 'float32'), 
        'Low': pd.Series(range(0, 15), dtype = 'float32'), 
        'Close': pd.Series(range(0, 15), dtype = 'float32')
    })
    features.atr(df1, period = 14)
    obj_first = features._atr_cache.get(14)

    df2 = df1 + 1
    features.atr(df2, period = 14)
    obj_second = features._atr_cache.get(14)
    assert obj_first is obj_second


def test_reset_indicator_caches():
    series = pd.Series(range(20), dtype = 'float32')
    features.rsi(series, period = 14)
    assert 14 in features._rsi_cache
    features.reset_indicator_caches()
    assert features._rsi_cache == {}
    assert features._atr_cache == {}
    assert features._sma_cache == {}