#
#
#
#
#
#             current_fold_index = 0, 
#             df, 
#             fold_config = {}, 
#             fund_profile = {'mm_mode': 'balanced', 'risk': 0.01}, 
#             initial_capital_segment = 1000.0, 
#             label = 'TEST', 
#             side = 'BUY', 
#         'Candle_Body', 'Candle_Range', 'Gain', 'cluster', 'spike_score', 'session'
#         'Pattern_Label', 'Entry_Long', 'Entry_Short', 'Trade_Tag', 'Signal_Score', 
#         'Trade_Reason', 'Volatility_Index', 'ADX', 'RSI', 'Wick_Ratio', 
#         'Trend_Zone', 'Gain_Z', 'MACD_hist', 'MACD_hist_smooth', 'Candle_Speed', 
#         )
#         df[col] = 0
#         strategy.run_backtest_simulation_v34(
#     ]
#     assert any('[QA][SUMMARY] Fold Finished' in m for m in caplog.messages)
#     df = simple_m1_df.copy()
#     df.index.name = 'Datetime'
#     df['ATR_14'] = 0.1
#     df['ATR_14_Rolling_Avg'] = 0.1
#     df['ATR_14_Shifted'] = 0.1
#     df['Entry_Long'] = 1
#     for col in required_extra:
#     required_extra = [
#     with caplog.at_level(logging.WARNING):
# def test_fold_summary_logging(simple_m1_df, caplog):
# from src import strategy
# import logging
# import os, sys
# import pandas as pd
# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, ROOT_DIR)