

#             raise RuntimeError('fail')
#         'ATR_14_Shifted': [0.1, 0.1, 0.1, 0.1], 
#         'Close': [1, 1, 1, 1], 
#         'High': [1, 1, 1, 1], 
#         'Low': [1, 1, 1, 1], 
#         'Open': [1, 1, 1, 1], 
#         call['n'] += 1
#         df_m1_final = df, 
#         fund_profile = fund, 
#         if call['n'] == 1:
#         n_walk_forward_splits = 2, 
#         output_dir = str(out_dir)
#         return (df.iloc[:1], trade_log, 1000.0, {}, 0.0, {}, [], 'L1', 'L2', False, 0, 0.0)
#         return {}
#         trade_log = pd.DataFrame({'side': ['BUY'], 'exit_reason': ['TP']})
#     )
#     assert isinstance(metrics_buy, dict)
#     assert isinstance(metrics_sell, dict)
#     call = {'n': 0}
#     def dummy_metrics(*args, **kwargs):
#     def dummy_run(*args, **kwargs):
#     df = pd.DataFrame({
#     fund = {'name': 'TFund', 'mm_mode': 'static', 'risk': 1}
#     metrics_buy, metrics_sell = result[0], result[1]
#     monkeypatch.setattr(strategy, 'calculate_metrics', dummy_metrics)
#     monkeypatch.setattr(strategy, 'run_backtest_simulation_v34', dummy_run)
#     out_dir = tmp_path / 'out'
#     out_dir.mkdir()
#     result = strategy.run_all_folds_with_threshold(
#     }, index = pd.date_range('2023 - 01 - 01', periods = 4, freq = 'min'))
# def test_run_all_folds_metrics_error(monkeypatch, tmp_path):
# from src import strategy
# import os, sys
# import pandas as pd
# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.insert(0, ROOT_DIR)