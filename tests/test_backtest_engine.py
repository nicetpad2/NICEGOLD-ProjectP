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
#
#
#
#
#
#
#
#
#                          index = [pd.Timestamp('2024 - 01 - 02'), pd.Timestamp('2024 - 01 - 01')])
#             Entry_Long = 0, 
#             Entry_Short = 0, 
#             Signal_Score = 0.1, 
#             Trade_Reason = 'r'
#             Trade_Tag = 't', 
#         'Close': [1, 2]
#         'Close': [1]
#         'Close': [1], 
#         'Close': [1], 
#         'Date': ['2024 - 01 - 01', '2024 - 01 - 01'], 
#         'Date': ['2024 - 01 - 01'], 
#         'Date': ['2024 - 01 - 01'], 
#         'duplicate labels' in msg or 'duplicates based on' in msg
#         'High': [1], 
#         'High': [1], 
#         'Low': [1], 
#         'Low': [1], 
#         'Open': [1], 
#         'Open': [1], 
#         'Timestamp': ['00:00:00', '00:00:00'], 
#         'Timestamp': ['00:00:00'], 
#         'Timestamp': ['00:00:00'], 
#         )
#         assert col in flags.get('cols', [])
#         be.run_backtest_engine(pd.DataFrame())
#         calls['count'] += 1
#         captured['current_fold_index'] = kwargs.get('current_fold_index')
#         captured['fold_config'] = kwargs.get('fold_config')
#         captured['index_is_dt'] = isinstance(df.index, pd.DatetimeIndex)
#         captured['m15_index'] = df.index
#         captured['m1_index'] = df.index
#         captured['tz_attr'] = getattr(df.index, 'tz', None)
#         Entry_Long = 0, Entry_Short = 0, Trade_Tag = 't', Signal_Score = 0.0, Trade_Reason = 'r'))
#         flags['cols'] = df.columns.tolist()
#         flags['has_trend'] = 'Trend_Zone' in df.columns
#         flags['trend_called'] = True
#         for msg in caplog.messages
#         result = be.run_backtest_engine(pd.DataFrame())
#         result = be.run_backtest_engine(pd.DataFrame())
#         result = be.run_backtest_engine(pd.DataFrame())
#         result = be.run_backtest_engine(pd.DataFrame())
#         result = be.run_backtest_engine(pd.DataFrame())
#         result = be.run_backtest_engine(pd.DataFrame())
#         return df
#         return df
#         return df.assign(
#         return m1_df if path == be.DEFAULT_CSV_PATH_M1 else m15_df
#         return m1_df if path == be.DEFAULT_CSV_PATH_M1 else m15_df
#         return m1_df if path == be.DEFAULT_CSV_PATH_M1 else m15_df
#         return m1_df if path == be.DEFAULT_CSV_PATH_M1 else m15_df
#         return m1_df if path == be.DEFAULT_CSV_PATH_M1 else m15_df
#         return None, trade_df
#         return None, trade_df
#         return None, trade_df
#         return pd.DataFrame({'Trend_Zone': ['UP', 'DOWN']}, index = df.index)
#         return pd.DataFrame({'Trend_Zone': ['UP', 'DOWN']}, index = df.index)
#         return pd.DataFrame({'Trend_Zone': ['UP']}, index = df.index)
#         return pd.DataFrame({'Trend_Zone': ['UP']}, index = df.index)
#     """ควรรีเทิร์น DataFrame เมื่อ simulation ทำงานสำเร็จ"""
#     """ควรส่ง fold_config และ current_fold_index ให้ simulation"""
#     """ควรเรียก engineer_m1_features ก่อน simulation"""
#     """ควรแปลง index เป็น DatetimeIndex เพื่อให้มีคุณสมบัติ .tz"""
#     """หากรูปแบบวันที่ไม่ตรงควร fallback เป็น pd.to_datetime แบบอัตโนมัติ"""
#     """หากโหลดราคาล้มเหลวต้องยก RuntimeError"""
#     """เมื่อ trade log ว่างควรรีเทิร์น DataFrame ว่าง"""
#     )
#     assert 'tz_attr' in captured
#     assert any(
#     assert any('duplicate index rows' in msg for msg in caplog.messages)
#     assert any('duplicate labels' in msg for msg in caplog.messages)
#     assert any('index M1 ไม่เรียงลำดับเวลา' in msg for msg in caplog.messages)
#     assert any('index ซ้ำซ้อนในข้อมูลราคา M1' in msg for msg in caplog.messages)
#     assert any('parse วันที่/เวลา' in msg for msg in caplog.messages)
#     assert any('Sorted Trend Zone DataFrame index' in msg for msg in caplog.messages)
#     assert calls['count'] == 1
#     assert captured['current_fold_index'] == be.DEFAULT_FOLD_INDEX
#     assert captured['fold_config'] == be.DEFAULT_FOLD_CONFIG
#     assert captured['index_is_dt']
#     assert captured['m15_index'][0] == pd.Timestamp('2024 - 01 - 01 00:00:00')
#     assert captured['m1_index'][0] == pd.Timestamp('2024 - 01 - 01 00:00:00')
#     assert flags.get('has_trend')
#     assert flags.get('trend_called')
#     assert isinstance(captured.get('m15_index'), pd.DatetimeIndex)
#     assert isinstance(captured.get('m1_index'), pd.DatetimeIndex)
#     assert isinstance(result, pd.DataFrame)
#     assert result.empty
#     assert result.equals(trade_df)
#     assert result.equals(trade_df)
#     assert result.equals(trade_df)
#     assert result.equals(trade_df)
#     assert result.equals(trade_df)
#     assert result.equals(trade_df)
#     assert result.equals(trade_df)
#     assert result.equals(trade_df)
#     assert result.equals(trade_df)
#     assert result.equals(trade_df)
#     assert result.equals(trade_df)
#     calls = {'count': 0}
#     captured = {}
#     captured = {}
#     captured = {}
#     def fake_engineer(df, **k):
#     def fake_engineer(df, **k):
#     def fake_entry(df, cfg):
#     def fake_read_csv(path, *a, **k):
#     def fake_read_csv(path, *a, **k):
#     def fake_read_csv(path, *a, **k):
#     def fake_read_csv(path, *a, **k):
#     def fake_read_csv(path, *a, **k):
#     def fake_sim(df, **k):
#     def fake_simulation(df, **k):
#     def fake_simulation(df, **kwargs):
#     def fake_trend(df):
#     def fake_trend(df):
#     def fake_trend(df):
#     def fake_trend(df):
#     flags = {}
#     for col in ['Trend_Zone', 'Entry_Long', 'Entry_Short', 'Trade_Tag', 'Signal_Score', 'Trade_Reason']:
#     idx = [pd.Timestamp('2024 - 01 - 01'), pd.Timestamp('2024 - 01 - 01')]
#     m15_df = pd.DataFrame({
#     m15_df = pd.DataFrame({
#     m15_df = pd.DataFrame({'Close': [1, 2]}, index = [pd.Timestamp('2024 - 01 - 01'), pd.Timestamp('2024 - 01 - 01')])
#     m15_df = pd.DataFrame({'Close': [1, 2]}, index = [pd.Timestamp('2024 - 01 - 02'), pd.Timestamp('2024 - 01 - 01')])
#     m15_df = pd.DataFrame({'Close': [1]}, index = [pd.Timestamp('2024 - 01 - 01')])
#     m1_df = pd.DataFrame({
#     m1_df = pd.DataFrame({'Open': [1, 2], 'High': [1, 2], 'Low': [1, 2], 'Close': [1, 2]}, index = idx)
#     m1_df = pd.DataFrame({'Open': [1], 'High': [1], 'Low': [1], 'Close': [1]})
#     m1_df = pd.DataFrame({'Open': [1], 'High': [1], 'Low': [1], 'Close': [1]})
#     m1_df = pd.DataFrame({'Open': [1], 'High': [1], 'Low': [1], 'Close': [1]})
#     m1_df = pd.DataFrame({'Open': [1], 'High': [1], 'Low': [1], 'Close': [1]})
#     m1_df = pd.DataFrame({'Open': [1], 'High': [1], 'Low': [1], 'Close': [1]}, 
#     monkeypatch.setattr(be, 'calculate_m15_trend_zone', fake_trend)
#     monkeypatch.setattr(be, 'calculate_m15_trend_zone', fake_trend)
#     monkeypatch.setattr(be, 'calculate_m15_trend_zone', fake_trend)
#     monkeypatch.setattr(be, 'calculate_m15_trend_zone', fake_trend)
#     monkeypatch.setattr(be, 'calculate_m15_trend_zone', lambda df: pd.DataFrame({'Trend_Zone': ['UP', 'UP']}, index = df.index))
#     monkeypatch.setattr(be, 'calculate_m15_trend_zone', lambda df: pd.DataFrame({'Trend_Zone': ['UP']}, index = [pd.Timestamp('2024 - 01 - 01')]))
#     monkeypatch.setattr(be, 'calculate_m15_trend_zone', lambda df: pd.DataFrame({'Trend_Zone': ['UP']}, index = df.index))
#     monkeypatch.setattr(be, 'calculate_m1_entry_signals', fake_entry)
#     monkeypatch.setattr(be, 'calculate_m1_entry_signals', lambda df, cfg: df.assign(
#     monkeypatch.setattr(be, 'calculate_m1_entry_signals', lambda df, cfg: df.assign(Entry_Long = 0, Entry_Short = 0, Trade_Tag = 't', Signal_Score = 0.0, Trade_Reason = 'r'))
#     monkeypatch.setattr(be, 'calculate_m1_entry_signals', lambda df, cfg: df.assign(Entry_Long = 0, Entry_Short = 0, Trade_Tag = 't', Signal_Score = 0.0, Trade_Reason = 'r'))
#     monkeypatch.setattr(be, 'calculate_m1_entry_signals', lambda df, cfg: df.assign(Entry_Long = 0, Entry_Short = 0, Trade_Tag = 't', Signal_Score = 0.0, Trade_Reason = 'r'))
#     monkeypatch.setattr(be, 'calculate_m1_entry_signals', lambda df, cfg: df.assign(Entry_Long = 0, Entry_Short = 0, Trade_Tag = 't', Signal_Score = 0.0, Trade_Reason = 'r'))
#     monkeypatch.setattr(be, 'calculate_m1_entry_signals', lambda df, cfg: df.assign(Entry_Long = 0, Entry_Short = 0, Trade_Tag = 't', Signal_Score = 0.0, Trade_Reason = 'r'))
#     monkeypatch.setattr(be, 'engineer_m1_features', fake_engineer)
#     monkeypatch.setattr(be, 'engineer_m1_features', fake_engineer)
#     monkeypatch.setattr(be, 'engineer_m1_features', lambda df, **k: (_ for _ in ()).throw(AssertionError('should not be called')))
#     monkeypatch.setattr(be, 'engineer_m1_features', lambda df, **k: df)
#     monkeypatch.setattr(be, 'engineer_m1_features', lambda df, **k: df)
#     monkeypatch.setattr(be, 'engineer_m1_features', lambda df, **k: df)
#     monkeypatch.setattr(be, 'engineer_m1_features', lambda df, **k: df)
#     monkeypatch.setattr(be, 'engineer_m1_features', lambda df, **k: df)
#     monkeypatch.setattr(be, 'engineer_m1_features', lambda df, **k: df)
#     monkeypatch.setattr(be, 'engineer_m1_features', lambda df, **k: df)
#     monkeypatch.setattr(be, 'engineer_m1_features', lambda df, **k: df)
#     monkeypatch.setattr(be, 'engineer_m1_features', lambda df, **k: df)
#     monkeypatch.setattr(be, 'engineer_m1_features', lambda df, **k: df)
#     monkeypatch.setattr(be, 'run_backtest_simulation_v34', fake_sim)
#     monkeypatch.setattr(be, 'run_backtest_simulation_v34', fake_simulation)
#     monkeypatch.setattr(be, 'run_backtest_simulation_v34', fake_simulation)
#     monkeypatch.setattr(be, 'run_backtest_simulation_v34', lambda df, **k: (None, pd.DataFrame()))
#     monkeypatch.setattr(be, 'run_backtest_simulation_v34', lambda df, **k: (None, trade_df))
#     monkeypatch.setattr(be, 'run_backtest_simulation_v34', lambda df, **k: (None, trade_df))
#     monkeypatch.setattr(be, 'run_backtest_simulation_v34', lambda df, **k: (None, trade_df))
#     monkeypatch.setattr(be, 'run_backtest_simulation_v34', lambda df, **k: (None, trade_df))
#     monkeypatch.setattr(be, 'run_backtest_simulation_v34', lambda df, **k: (None, trade_df))
#     monkeypatch.setattr(be, 'run_backtest_simulation_v34', lambda df, **k: (None, trade_df))
#     monkeypatch.setattr(be, 'run_backtest_simulation_v34', lambda df, **k: (None, trade_df))
#     monkeypatch.setattr(be, 'run_backtest_simulation_v34', lambda df, **k: (None, trade_df))
#     monkeypatch.setattr(be, 'safe_load_csv_auto', fake_read_csv)
#     monkeypatch.setattr(be, 'safe_load_csv_auto', fake_read_csv)
#     monkeypatch.setattr(be, 'safe_load_csv_auto', fake_read_csv)
#     monkeypatch.setattr(be, 'safe_load_csv_auto', fake_read_csv)
#     monkeypatch.setattr(be, 'safe_load_csv_auto', fake_read_csv)
#     monkeypatch.setattr(be, 'safe_load_csv_auto', lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError('no')))
#     monkeypatch.setattr(be, 'safe_load_csv_auto', lambda *a, **k: m1_df)
#     monkeypatch.setattr(be, 'safe_load_csv_auto', lambda *a, **k: m1_df)
#     monkeypatch.setattr(be, 'safe_load_csv_auto', lambda *a, **k: price_df)
#     monkeypatch.setattr(be, 'safe_load_csv_auto', lambda *a, **k: price_df)
#     monkeypatch.setattr(be, 'safe_load_csv_auto', lambda *a, **k: price_df)
#     monkeypatch.setattr(be, 'safe_load_csv_auto', lambda *a, **k: price_df)
#     monkeypatch.setattr(be, 'safe_load_csv_auto', lambda *a, **k: price_df)
#     price_df = pd.DataFrame({
#     price_df = pd.DataFrame({'Open': [1], 'High': [1], 'Low': [1], 'Close': [1]})
#     price_df = pd.DataFrame({'Open': [1], 'High': [1], 'Low': [1], 'Close': [1]})
#     price_df = pd.DataFrame({'Open': [1], 'High': [1], 'Low': [1], 'Close': [1]})
#     price_df = pd.DataFrame({'Open': [1], 'High': [1], 'Low': [1], 'Close': [1]})
#     result = be.run_backtest_engine(pd.DataFrame())
#     result = be.run_backtest_engine(pd.DataFrame())
#     result = be.run_backtest_engine(pd.DataFrame())
#     result = be.run_backtest_engine(pd.DataFrame())
#     result = be.run_backtest_engine(pd.DataFrame())
#     result = be.run_backtest_engine(pd.DataFrame())
#     trade_df = pd.DataFrame({'pnl': [1.0]})
#     trade_df = pd.DataFrame({'pnl': [1.0]})
#     trade_df = pd.DataFrame({'pnl': [1.0]})
#     trade_df = pd.DataFrame({'pnl': [1.0]})
#     trade_df = pd.DataFrame({'pnl': [1.0]})
#     trade_df = pd.DataFrame({'pnl': [1.0]})
#     trade_df = pd.DataFrame({'pnl': [1.0]})
#     trade_df = pd.DataFrame({'pnl': [1.0]})
#     trade_df = pd.DataFrame({'pnl': [1.0]})
#     trade_df = pd.DataFrame({'pnl': [1.0]})
#     trade_df = pd.DataFrame({'pnl': [1.0]})
#     with caplog.at_level(logging.INFO):
#     with caplog.at_level(logging.INFO):
#     with caplog.at_level(logging.INFO):
#     with caplog.at_level(logging.INFO):
#     with caplog.at_level(logging.WARNING):
#     with caplog.at_level(logging.WARNING):
#     with pytest.raises(RuntimeError):
#     })
#     })
#     })
#     }, index = ['2024 - 01 - 01 00:00:00'])
# def test_run_backtest_engine_calls_feature_engineering(monkeypatch):
# def test_run_backtest_engine_dedup_m15_index(monkeypatch, caplog):
# def test_run_backtest_engine_drops_duplicate_m1_index(monkeypatch, caplog):
# def test_run_backtest_engine_drops_duplicate_trend_index(monkeypatch, caplog):
# def test_run_backtest_engine_empty_log(monkeypatch):
# def test_run_backtest_engine_fail_load(monkeypatch):
# def test_run_backtest_engine_generates_trend_and_signals(monkeypatch):
# def test_run_backtest_engine_index_conversion(monkeypatch):
# def test_run_backtest_engine_parse_datetime_fallback(monkeypatch, caplog):
# def test_run_backtest_engine_passes_fold_params(monkeypatch):
# def test_run_backtest_engine_sorts_m1_index(monkeypatch, caplog):
# def test_run_backtest_engine_sorts_trend_index(monkeypatch, caplog):
# def test_run_backtest_engine_success(monkeypatch):
# import backtest_engine as be
# import logging
# import pandas as pd
# import pytest