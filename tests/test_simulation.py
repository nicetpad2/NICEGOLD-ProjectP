

#                       'Candle_Range', 'Gain', 'cluster', 'spike_score', 'session']
#                       'Pattern_Label', 'Entry_Long', 'Entry_Short', 'Trade_Tag', 'Signal_Score', 
#                       'Trade_Reason', 'Volatility_Index', 'ADX', 'RSI', 'Wick_Ratio', 'Candle_Body', 
#             current_fold_index = 0, 
#             current_fold_index = 0, 
#             df, 
#             df, 
#             fold_config = {}, 
#             fold_config = {}, 
#             initial_capital_segment = 1000, 
#             initial_capital_segment = 1000, 
#             label = 'TEST', 
#             label = 'TEST', 
#             return [[0.6, 0.4]]
#         'main': {'model': DummyModel(), 'features': ['Signal_Score', missing_feature]}
#         'Pattern_Label', 'Entry_Long', 'Entry_Short', 'Trade_Tag', 
#         'Pattern_Label', 'Entry_Long', 'Entry_Short', 'Trade_Tag', 
#         'Signal_Score', 'Trade_Reason', 'Volatility_Index', 'ADX', 'RSI', 
#         'Signal_Score', 'Trade_Reason', 'Volatility_Index', 'ADX', 'RSI', 
#         'spike_score', 'session'
#         'spike_score', 'session'
#         'Trend_Zone', 'Gain_Z', 'MACD_hist', 'MACD_hist_smooth', 'Candle_Speed', 
#         'Trend_Zone', 'Gain_Z', 'MACD_hist', 'MACD_hist_smooth', 'Candle_Speed', 
#         'Wick_Ratio', 'Candle_Body', 'Candle_Range', 'Gain', 'cluster', 
#         'Wick_Ratio', 'Candle_Body', 'Candle_Range', 'Gain', 'cluster', 
#         )
#         )
#         available_models = available_models, 
#         current_fold_index = 0
#         current_fold_index = 0, 
#         def predict_proba(self, X):
#         df, 
#         df, 
#         df[col] = 0
#         df[col] = 0
#         df[col] = 0
#         fold_config = {}, 
#         fold_config = {}, 
#         fund_profile = {'mm_mode': 'balanced', 'risk': 0.01}, 
#         fund_profile = {'mm_mode': 'balanced', 'risk': 0.01}, 
#         initial_capital_segment = 1000, 
#         initial_capital_segment = 1000, 
#         label = 'TEST', 
#         label = 'TEST', 
#         model_switcher_func = lambda ctx, models: ('main', 1.0)
#         result = strategy.run_backtest_simulation_v34(
#         side = 'BUY', 
#         side = 'BUY', 
#         strategy.run_backtest_simulation_v34(
#     # omit ATR_14_Rolling_Avg to trigger early return
#     )
#     )
#     ]
#     ]
#     assert 'index is not DatetimeIndex' in caplog.text
#     assert run_summary['error_in_loop']
#     assert run_summary['error_in_loop']
#     assert run_summary['error_in_loop'] is False
#     available_models = {
#     class DummyModel:
#     df = simple_m1_df.copy()
#     df = simple_m1_df.copy()
#     df = simple_m1_df.drop(columns = ['High'])
#     df = simple_m1_df.reset_index(drop = True)
#     df['ATR_14'] = 0.1
#     df['ATR_14'] = 1.0
#     df['ATR_14'] = 1.0
#     df['ATR_14'] = 1.0
#     df['ATR_14_Rolling_Avg'] = 0.1
#     df['ATR_14_Rolling_Avg'] = 0.1
#     df['ATR_14_Shifted'] = 0.1
#     df['ATR_14_Shifted'] = 1.0
#     df['ATR_14_Shifted'] = 1.0
#     df['ATR_14_Shifted'] = 1.0
#     df['Entry_Long'] = 1
#     for col in required_extra:
#     for col in required_extra:
#     for col in required_extra:
#     missing_feature = 'dummy_feature'
#     monkeypatch.setattr(strategy, 'USE_META_CLASSIFIER', True, raising = False)
#     required_extra = [
#     required_extra = [
#     required_extra = ['Trend_Zone', 'Gain_Z', 'MACD_hist', 'MACD_hist_smooth', 'Candle_Speed', 
#     result = strategy.run_backtest_simulation_v34(
#     result = strategy.run_backtest_simulation_v34(
#     run_summary = result[5]
#     run_summary = result[5]
#     run_summary = result[5]
#     with caplog.at_level(logging.ERROR):
#     with pytest.raises(ValueError):
#     }
# def test_run_backtest_invalid_index(simple_m1_df, caplog):
# def test_run_backtest_simulation_missing_cols(simple_m1_df):
# def test_run_backtest_simulation_missing_price_cols(simple_m1_df):
# def test_run_backtest_simulation_ml_feature_check(simple_m1_df, monkeypatch):
# from src import strategy  # Disabled due to circular import issues
import logging
import pandas as pd
import pytest