

#         "Close": [1.0, 1.1, 1.05, 1.2], 
#         "High": [1.1, 1.2, 1.1, 1.25], 
#         "Low": [0.9, 1.0, 1.0, 1.15], 
#     _run_oms_backtest_numba, 
#     assert count == 3
#     assert res[0] >= 0
#     close_sig = np.array([0, 1, 1], dtype = np.int8)
#     count = _run_oms_backtest_numba(prices, highs, lows, open_sig, close_sig, sl, tp)
#     df = pd.DataFrame({
#     folds = [(np.arange(0, 2), np.arange(2, 4))]
#     generate_close_signals, 
#     generate_open_signals, 
#     highs = np.array([1.1, 1.2, 1.3])
#     lows = np.array([0.9, 1.0, 1.1])
#     open_sig = np.array([1, 0, 0], dtype = np.int8)
#     precompute_sl_array, 
#     precompute_tp_array, 
#     prices = np.array([1.0, 1.1, 1.2])
#     res = run_simple_numba_backtest(df, folds)
#     run_simple_numba_backtest, 
#     sl = np.zeros(3, dtype = np.float64)
#     tp = np.zeros(3, dtype = np.float64)
#     })
# )
# def test_run_all_folds_with_threshold():
# def test_run_oms_backtest_numba_basic():
# from src.strategy import (
# import numpy as np
# import os, sys
# import pandas as pd
# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.insert(0, ROOT_DIR)