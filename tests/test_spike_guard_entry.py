

#                 "ADX": 10, 
#                 "ADX": 30, 
#                 "ADX": 30, 
#                 "ADX": np.nan, 
#                 "ATR_14": 1.0, 
#                 "ATR_14": 1.0, 
#                 "ATR_14": 1.0, 
#                 "ATR_14": 5.0, 
#                 "Candle_Body": 0.3, 
#                 "Candle_Body": 0.3, 
#                 "Candle_Body": 0.3, 
#                 "Candle_Body": 1.0, 
#                 "Candle_Range": 0.5, 
#                 "Candle_Range": 0.5, 
#                 "Candle_Range": 1.0, 
#                 "Candle_Range": 2.0, 
#                 "Gain": 1.0, 
#                 "Gain": 1.0, 
#                 "Gain": 1.0, 
#                 "Gain": 4.0, 
#                 "spike_score": 0.1, 
#                 "spike_score": 0.1, 
#                 "spike_score": 0.1, 
#                 "spike_score": 0.9, 
#                 "Volatility_Index": 0.7, 
#                 "Volatility_Index": 1.0, 
#                 "Volatility_Index": 1.0, 
#                 "Volatility_Index": 1.2, 
#                 "Wick_Ratio": 0.5, 
#                 "Wick_Ratio": 0.5, 
#                 "Wick_Ratio": 0.5, 
#                 "Wick_Ratio": 0.8, 
#             "London", 
#             "London", 
#             "London", 
#             "London", 
#             False, 
#             False, 
#             pd.Series({
#             pd.Series({
#             pd.Series({
#             pd.Series({
#             True, 
#             True, 
#             }), 
#             }), 
#             }), 
#             }), 
#         "ADX": 30, 
#         "ATR_14": 1.0, 
#         "BUY", 
#         "Candle_Body": 0.4, 
#         "Candle_Range": 0.8, 
#         "Gain": 1.0, 
#         "London", 
#         "Signal_Score": score, 
#         "spike_score": 0.0, 
#         "UP", 
#         "Volatility_Index": vol, 
#         "Wick_Ratio": 0.5, 
#         (
#         (
#         (
#         (
#         ("Asia", pd.Series({"spike_score": 0.9}), True), 
#         (make_row(0.5, 1.5), "LOW_VOLATILITY"), 
#         (make_row(1.2, 0.2), "LOW_SIGNAL_SCORE"), 
#         (make_row(1.2, 1.5), "ALLOWED"), 
#         (make_row(1.2, np.nan), "INVALID_SIGNAL_SCORE"), 
#         ), 
#         ), 
#         ), 
#         ), 
#         0, 
#         row, 
#         signal_score_threshold = 1.0, 
#     "row, expected_reason", 
#     "session, row, expected", 
#     )
#     [
#     [
#     ], 
#     ], 
#     allowed, reason = strategy.is_entry_allowed(
#     assert ("ALLOWED" in reason) == (expected_reason == "ALLOWED")
#     assert expected_reason.split("(")[0] in reason
#     assert strategy.spike_guard_london(row, session, 0) is expected
#     monkeypatch.setattr(strategy, "spike_guard_london", lambda *a, **k: True)
#     return pd.Series({
#     })
# )
# )
# @pytest.mark.parametrize(
# @pytest.mark.parametrize(
# def make_row(vol, score):
# def test_is_entry_allowed(monkeypatch, row, expected_reason):
# def test_spike_guard_london(session, row, expected):
# from src import strategy
# import numpy as np
# import os, sys
# import pandas as pd
# import pytest
# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, ROOT_DIR)