
#
#
#
#
#
#
#
#
#
#         'ATR_14_Shifted': [0.1]*6, 
#         'Close': [1]*6, 
#         'High': [1]*6, 
#         'Low': [1]*6, 
#         'Open': [1]*6, 
#         df_m1_final = df, 
#         fund_profile = fund, 
#         n_walk_forward_splits = 2, 
#         output_dir = str(out_dir)
#         return (df.iloc[:0], pd.DataFrame(), 1000.0, {}, 0.0, {}, [], 'L1', 'L2', False, 0, 0.0)
#     )
#     assert trade_log.empty
#     def dummy_run(*args, **kwargs):
#     df = pd.DataFrame({
#     fund = {'name': 'DBG', 'mm_mode': 'static', 'risk': 1}
#     monkeypatch.setattr(strategy, 'run_backtest_simulation_v34', dummy_run)
#     os.environ.pop('DEBUG_FAKE_TRADE')
#     os.environ['DEBUG_FAKE_TRADE'] = '1'
#     out_dir = tmp_path / 'out'
#     out_dir.mkdir()
#     result = strategy.run_all_folds_with_threshold(
#     trade_log = result[3]
#     }, index = pd.date_range('2023 - 01 - 01', periods = 6, freq = 'min'))
# def test_fake_trade_generated(monkeypatch, tmp_path):
# from src import strategy
# import os
# import pandas as pd
# import sys
# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, ROOT_DIR)