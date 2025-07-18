
import logging
import os
import pandas as pd
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

# from src.strategy import get_dynamic_signal_score_entry, get_dynamic_signal_score_thresholds
# from src import trade_utils


# def test_dynamic_signal_score_entry_clamps_max():
#     df = pd.DataFrame({'Signal_Score': range(10)})
#     val = get_dynamic_signal_score_entry(df, window = 5, quantile = 0.8, min_val = 0.5, max_val = 3.0)
#     assert val == 3.0


# def test_dynamic_signal_score_entry_clamps_min():
#     df = pd.DataFrame({'Signal_Score': [0.1]*20})
#     val = get_dynamic_signal_score_entry(df, window = 10, quantile = 0.7, min_val = 0.5, max_val = 3.0)
#     assert val == 0.5


# def test_adaptive_threshold_logging(caplog):
#     df = pd.DataFrame({'Signal_Score': [1, 2, 3]})
#     with caplog.at_level(logging.INFO):
#         thresh = get_dynamic_signal_score_entry(df, window = 2, quantile = 0.5)
#         logging.info(f"[Adaptive] Current Signal_Score threshold: {thresh:.2f}")
#     assert any('[Adaptive] Current Signal_Score threshold' in m for m in caplog.messages)


# def test_vectorized_thresholds_match_single_calc():
#     series = pd.Series(range(10))
#     arr = get_dynamic_signal_score_thresholds(series, window = 5, quantile = 0.8, min_val = 0.5, max_val = 3.0)
#     for i in range(len(series)):
#         df_slice = pd.DataFrame({'Signal_Score': series.iloc[:i + 1]})
#         single_val = get_dynamic_signal_score_entry(df_slice, window = 5, quantile = 0.8, min_val = 0.5, max_val = 3.0)
#         assert arr[i] == single_val


# def test_dynamic_thresholds_atr_adjusts():
#     scores = pd.Series([0.5] * 10)
#     atr = pd.Series([1] * 5 + [2] * 5)
#     avg = pd.Series([1] * 10)
#     arr = trade_utils.get_dynamic_signal_score_thresholds_atr(scores, atr, avg, slope = 0.5)
#     assert arr[ - 1] > arr[0]


# def test_dynamic_thresholds_atr_clamps():
#     scores = pd.Series(range(5))
#     atr = pd.Series([10] * 5)
#     avg = pd.Series([1] * 5)
#     arr = trade_utils.get_dynamic_signal_score_thresholds_atr(scores, atr, avg, slope = 1.0, max_val = 3.0)
#     assert max(arr) <= 3.0