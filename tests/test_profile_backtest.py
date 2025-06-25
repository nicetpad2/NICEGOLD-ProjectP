

#                        'Open': [1, 2], 'High': [1, 2], 'Low': [1, 2], 'Close': [1, 2]})
#                      'Entry_Long', 'Entry_Short', 'Signal_Score', 
#                      'Trade_Tag', 'Trade_Reason']:
#             ' -  - output', str(out), ' -  - output - file', str(prof), ' -  - fund', 'SPIKE', 
#             ' -  - train', ' -  - train - output', str(train_dir)
#             'profile_backtest.py', str(m1), ' -  - rows', '2', ' -  - limit', '5', 
#             assert h.level == logging.WARNING
#         ' -  - output', str(out_txt), ' -  - output - file', str(out_prof)
#         ' -  - output - file', str(tmp_path / 'console.prof')
#         'argv', 
#         'Close': [1, 2, 3, 4]
#         'Close': [1, 2, 3]
#         'Close': [1, 2, 3]
#         'Close': [1, 2, 3]
#         'Close': [1, 2]
#         'Close': [1, 2]
#         'Close': [1, 2]
#         'Close': [1, 2]
#         'Close': [1, 2]
#         'Datetime': idx_m1, 
#         'Datetime': idx_m15, 
#         'Datetime': pd.date_range('2022 - 01 - 01', periods = 1200, freq = 'min', tz = 'UTC'), 
#         'Datetime': pd.date_range('2022 - 01 - 01', periods = 1500, freq = 'min', tz = 'UTC'), 
#         'Datetime': pd.date_range('2022 - 01 - 01', periods = 2, freq = 'min', tz = 'UTC'), 
#         'Datetime': pd.date_range('2022 - 01 - 01', periods = 2, freq = 'min', tz = 'UTC'), 
#         'Datetime': pd.date_range('2022 - 01 - 01', periods = 2, freq = 'min', tz = 'UTC'), 
#         'Datetime': pd.date_range('2022 - 01 - 01', periods = 2, freq = 'min', tz = 'UTC'), 
#         'Datetime': pd.date_range('2022 - 01 - 01', periods = 2, freq = 'min', tz = 'UTC'), 
#         'Datetime': pd.date_range('2022 - 01 - 01', periods = 3, freq = 'min', tz = 'UTC'), 
#         'Datetime': pd.date_range('2022 - 01 - 01', periods = 3, freq = 'min', tz = 'UTC'), 
#         'High': [1, 2, 3, 4], 
#         'High': [1, 2, 3], 
#         'High': [1, 2, 3], 
#         'High': [1, 2], 
#         'High': [1, 2], 
#         'High': [1, 2], 
#         'High': [1, 2], 
#         'High': [1, 2], 
#         'Low': [1, 2, 3, 4], 
#         'Low': [1, 2, 3], 
#         'Low': [1, 2, 3], 
#         'Low': [1, 2], 
#         'Low': [1, 2], 
#         'Low': [1, 2], 
#         'Low': [1, 2], 
#         'Low': [1, 2], 
#         'Open': [1, 2, 3, 4], 
#         'Open': [1, 2, 3], 
#         'Open': [1, 2, 3], 
#         'Open': [1, 2, 3], 
#         'Open': [1, 2], 
#         'Open': [1, 2], 
#         'Open': [1, 2], 
#         'Open': [1, 2], 
#         'Open': [1, 2], 
#         'Open': [1, 2], 'High': [1, 2], 'Low': [1, 2], 'Close': [1, 2]
#         'Open': [1]*1200, 'High': [1]*1200, 'Low': [1]*1200, 'Close': [1]*1200
#         'Open': [1]*1500, 'High': [1]*1500, 'Low': [1]*1500, 'Close': [1]*1500
#         'profile_backtest.py', str(m1), ' -  - rows', '2', ' -  - console_level', 'WARNING', 
#         'profile_backtest.py', str(m1), ' -  - rows', '2', ' -  - limit', '5', 
#         'profile_backtest.py', str(m1), ' -  - rows', '2', ' -  - output - profile - dir', str(out_dir)
#         'profile_backtest.py', str(m1), ' -  - rows', '500', ' -  - debug'
#         [
#         ], 
#         assert required in captured_cols['cols']
#         called['out'] = out
#         called['out'] = out
#         captured['fund'] = kwargs.get('fund_profile')
#         captured['index'] = df.index
#         captured['rows'] = len(df)
#         captured['rows'] = len(df)
#         captured_cols['cols'] = df.columns.tolist()
#         captured_df['df'] = df
#         if isinstance(h, logging.StreamHandler):
#         return {}
#         return {}
#         sys, 
#     )
#     ])
#     ])
#     ])
#     ])
#     assert "Missing required columns" in caplog.text
#     assert 'ncalls' in text
#     assert 'Trend_Zone' in captured_df['df'].columns
#     assert called['out'] == str(out_dir)
#     assert called['out'] == str(train_dir)
#     assert captured['fund']['mm_mode'] == 'high_freq'
#     assert captured['index'].name == 'Datetime'
#     assert captured['rows'] == 1000
#     assert captured['rows'] == 500
#     assert isinstance(captured['index'], pd.DatetimeIndex)
#     assert len(prof_files) == 1
#     assert out.is_file()
#     assert out_prof.is_file()
#     assert out_txt.is_file()
#     assert prof.is_file()
#     called = {}
#     called = {}
#     caplog.set_level(logging.ERROR)
#     captured = {}
#     captured = {}
#     captured = {}
#     captured = {}
#     captured_cols = {}
#     captured_df = {}
#     csv_path = tmp_path / 'fund_M1.csv'
#     csv_path = tmp_path / 'mini_M1.csv'
#     csv_path = tmp_path / 'missing.csv'
#     csv_path = tmp_path / 'numeric_M1.csv'
#     csv_path = tmp_path / 'train_M1.csv'
#     def dummy_run(df, *a, **k):
#     def dummy_run(df, *a, **k):
#     def dummy_run(df, *args, **kwargs):
#     def dummy_run_backtest(df, *args, **kwargs):
#     def dummy_run_backtest(df, *args, **kwargs):
#     def dummy_run_backtest(df, *args, **kwargs):
#     def dummy_train(out, *a, **k):
#     def dummy_train(out, *args, **kwargs):
#     df = pd.DataFrame({
#     df = pd.DataFrame({
#     df = pd.DataFrame({
#     df = pd.DataFrame({
#     df = pd.DataFrame({
#     df = pd.DataFrame({
#     df = pd.DataFrame({
#     df = pd.DataFrame({
#     df = pd.DataFrame({
#     df = pd.DataFrame({
#     df = pd.DataFrame({'Datetime': pd.date_range('2022 - 01 - 01', periods = 2, freq = 'min', tz = 'UTC'), 
#     df.to_csv(csv_path)
#     df.to_csv(csv_path, index = False)
#     df.to_csv(csv_path, index = False)
#     df.to_csv(csv_path, index = False)
#     df.to_csv(csv_path, index = False)
#     df.to_csv(m1, index = False)
#     df.to_csv(m1, index = False)
#     df.to_csv(m1, index = False)
#     df.to_csv(m1, index = False)
#     df.to_csv(m1, index = False)
#     df.to_csv(m1, index = False)
#     df_m1 = pd.DataFrame({
#     df_m1.to_csv(m1_path, index = False)
#     df_m15 = pd.DataFrame({
#     df_m15.to_csv(m15_path, index = False)
#     for h in logging.getLogger().handlers:
#     for required in ['ATR_14', 'Gain_Z', 'MACD_hist', 
#     idx_m1 = pd.date_range('2022 - 01 - 01', periods = 4, freq = '1min', tz = 'UTC')
#     idx_m15 = pd.date_range('2022 - 01 - 01', periods = 2, freq = '15min', tz = 'UTC')
#     m1 = tmp_path / 'debug_M1.csv'
#     m1 = tmp_path / 'debug_override_M1.csv'
#     m1 = tmp_path / 'mini_M1.csv'
#     m1 = tmp_path / 'mini_M1.csv'
#     m1 = tmp_path / 'mini_M1.csv'
#     m1 = tmp_path / 'mini_M1.csv'
#     m15_path = tmp_path / 'XAUUSD_M15.csv'
#     m1_path = tmp_path / 'XAUUSD_M1.csv'
#     monkeypatch.setattr(
#     monkeypatch.setattr('src.training.real_train_func', dummy_train)
#     monkeypatch.setattr('src.training.real_train_func', dummy_train)
#     monkeypatch.setattr(profile_backtest, 'run_backtest_simulation_v34', dummy_run)
#     monkeypatch.setattr(profile_backtest, 'run_backtest_simulation_v34', dummy_run)
#     monkeypatch.setattr(profile_backtest, 'run_backtest_simulation_v34', dummy_run)
#     monkeypatch.setattr(profile_backtest, 'run_backtest_simulation_v34', dummy_run_backtest)
#     monkeypatch.setattr(profile_backtest, 'run_backtest_simulation_v34', dummy_run_backtest)
#     monkeypatch.setattr(profile_backtest, 'run_backtest_simulation_v34', dummy_run_backtest)
#     monkeypatch.setattr(profile_backtest, 'run_backtest_simulation_v34', lambda *a, **k: None)
#     monkeypatch.setattr(profile_backtest, 'run_backtest_simulation_v34', lambda *a, **k: None)
#     monkeypatch.setattr(profile_backtest, 'run_backtest_simulation_v34', lambda *a, **k: None)
#     monkeypatch.setattr(profile_backtest, 'run_backtest_simulation_v34', lambda *a, **k: None)
#     monkeypatch.setattr(profile_backtest, 'run_backtest_simulation_v34', lambda *a, **k: None)
#     monkeypatch.setattr(sys, 'argv', [
#     monkeypatch.setattr(sys, 'argv', [
#     monkeypatch.setattr(sys, 'argv', [
#     monkeypatch.setattr(sys, 'argv', [
#     monkeypatch.setattr(sys, 'argv', ['profile_backtest.py', str(m1), ' -  - debug'])
#     out = tmp_path / 'stats.txt'
#     out_dir = tmp_path / 'models'
#     out_dir = tmp_path / 'profiles'
#     out_prof = tmp_path / 'run.prof'
#     out_txt = tmp_path / 'stats.txt'
#     prof = tmp_path / 'cli.prof'
#     prof_files = list(out_dir.glob('*.prof'))
#     profile_backtest.main_profile(str(csv_path), num_rows = 2)
#     profile_backtest.main_profile(str(csv_path), num_rows = 2)
#     profile_backtest.main_profile(str(csv_path), num_rows = 2, fund_profile_name = 'AGGRESSIVE')
#     profile_backtest.main_profile(str(csv_path), num_rows = 2, train = True, train_output = str(out_dir))
#     profile_backtest.main_profile(str(csv_path), num_rows = 3)
#     profile_backtest.main_profile(str(m1_path), num_rows = 4)
#     profile_backtest.profile_from_cli()
#     profile_backtest.profile_from_cli()
#     profile_backtest.profile_from_cli()
#     profile_backtest.profile_from_cli()
#     profile_backtest.profile_from_cli()
#     profile_backtest.profile_from_cli()
#     text = out_txt.read_text()
#     train_dir = tmp_path / 'models'
#     })
#     })
#     })
#     })
#     })
#     })
#     })
#     })
#     })
#     })
#     })
#     })
# def test_cli_console_level(monkeypatch, tmp_path):
# def test_cli_debug_uses_default_rows(monkeypatch, tmp_path):
# def test_cli_debug_with_rows_override(monkeypatch, tmp_path):
# def test_main_profile_custom_fund(monkeypatch, tmp_path):
# def test_main_profile_merges_m15(tmp_path, monkeypatch):
# def test_main_profile_missing_columns(tmp_path, caplog):
# def test_main_profile_numeric_index(tmp_path, monkeypatch):
# def test_main_profile_runs(tmp_path, monkeypatch):
# def test_main_profile_train_option(monkeypatch, tmp_path):
# def test_profile_cli_fund_and_train(monkeypatch, tmp_path):
# def test_profile_cli_output_dir(tmp_path, monkeypatch):
# def test_profile_cli_output_file(tmp_path, monkeypatch):
# import logging
# import os
# import pandas as pd
# import profile_backtest
# import sys
# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, ROOT_DIR)
# sys.path.insert(1, os.path.join(ROOT_DIR, 'src'))