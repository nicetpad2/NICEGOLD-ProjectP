#
#
#
#
#
#
#
#
#
#
#             'L1', 
#             'L2', 
#             0, 
#             0.0, 
#             0.0, 
#             1000.0, 
#             [], 
#             False, 
#             pd.DataFrame(), 
#             simple_m1_df.iloc[:0], 
#             {}, 
#             {}, 
#         )
#         df_m1_final = df, 
#         fund_profile = {'name': 'T', 'mm_mode': 'static', 'risk': 1}, 
#         n_walk_forward_splits = 2, 
#         output_dir = str(out_dir)
#         return (
#         {"reason": "ML_META_FILTER"}, 
#         {"reason": "ML_META_FILTER"}, 
#         {"reason": "SOFT_COOLDOWN"}, 
#     )
#     ]
#     assert isinstance(metrics_buy, dict)
#     assert result["ML_META_FILTER"] == 2
#     assert result["SOFT_COOLDOWN"] == 1
#     assert trade_log.empty
#     def dummy_run(*args, **kwargs):
#     df = simple_m1_df.copy()
#     df['ATR_14'] = 0.1
#     df['ATR_14_Rolling_Avg'] = 0.1
#     df['ATR_14_Shifted'] = 0.1
#     logs = [
#     metrics_buy, metrics_sell, df_final, trade_log = result[0], result[1], result[2], result[3]
#     monkeypatch.setattr(strategy, 'run_backtest_simulation_v34', dummy_run)
#     out_dir = tmp_path / 'out'
#     out_dir.mkdir()
#     result = log_analysis.summarize_block_reasons(logs)
#     result = strategy.run_all_folds_with_threshold(
# def test_run_all_folds_handles_no_trades(simple_m1_df, monkeypatch, tmp_path):
# def test_summarize_block_reasons():
# from src import strategy, log_analysis
# import os, sys
# import pandas as pd
# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, ROOT_DIR)