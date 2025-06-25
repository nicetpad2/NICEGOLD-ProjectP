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
#         config = importlib.import_module('src.config')
#         importlib.reload(config)
#     """Ensure files outside DATA_DIR are not deleted."""
#     # Inside DATA_DIR should be removed
#     assert 'outside DATA_DIR' in caplog.text
#     assert check_duplicates(df, subset = ['A', 'B']) == 1
#     assert check_nan_percent(df) == pytest.approx(0.5)
#     assert inspect_file_exists(str(file_path)) is True
#     assert inspect_file_exists(str(tmp_path / 'missing.txt')) is False
#     assert isinstance(res.index, pd.DatetimeIndex)
#     assert len(res) == 1
#     assert not inside_file.exists()
#     assert os.path.exists(path)
#     assert outside_file.exists()
#     assert pd.api.types.is_datetime64_ns_dtype(res['Date'])
#     assert pd.isna(ts)
#     assert res.index[0] == pd.Timestamp('2020 - 06 - 12 03:00:00')
#     assert res['A'].dtype == 'float32'
#     assert ts.year == 2024
#     caplog.set_level('WARNING')
#     clean_test_file(str(outside_file))
#     csv = tmp_path / 'f.csv'
#     df = pd.DataFrame({'A': [1, 1, 2], 'B': [1, 1, 3]})
#     df = pd.DataFrame({'A': [1, np.nan]})
#     df = pd.DataFrame({'A': [1, np.nan]})
#     df = pd.DataFrame({'A': [1]})
#     df = pd.DataFrame({'Date': ['2024 - 01 - 01']})
#     df = pd.DataFrame({'Timestamp': ['2024 - 01 - 01']})
#     df = pd.DataFrame({'Timestamp': ['2563 - 06 - 12 03:00:00']})
#     df_in = pd.DataFrame({'A': [1, 2]})
#     df_in.to_csv(csv, index = False)
#     df_out = read_csv_with_date_parse(str(csv))
#     dl_reload.clean_test_file(str(inside_file))
#     else:
#     file_path = tmp_path / 'a.txt'
#     file_path.write_text('x', encoding = 'utf - 8')
#     if 'src.config' not in sys.modules:
#     import importlib
#     import src.config as config
#     import src.data_loader as dl_reload
#     importlib.reload(dl_reload)
#     inside_file = tmp_path / 'y.txt'
#     inside_file.write_text('hi', encoding = 'utf - 8')
#     monkeypatch.setenv('DATA_DIR', str(tmp_path))
#     outside_file = tmp_path / 'x.txt'
#     outside_file.write_text('hi', encoding = 'utf - 8')
#     path = write_test_file(str(tmp_path / 'w.txt'))
#     pd.testing.assert_frame_equal(df_out, pd.read_csv(csv, parse_dates = True))
#     res = convert_thai_years(df.copy(), 'Date')
#     res = main.convert_to_float32(df)
#     res = main.drop_nan_rows(df)
#     res = prepare_datetime_index(df.copy())
#     res = prepare_datetime_index(df.copy())
#     ts = convert_thai_datetime('2567 - 01 - 01 00:00')
#     ts = convert_thai_datetime('notadate', errors = 'coerce')
# def test_check_duplicates_subset():
# def test_check_nan_percent_basic():
# def test_clean_test_file_guard(tmp_path, caplog, monkeypatch):
# def test_convert_thai_datetime_invalid():
# def test_convert_thai_datetime_valid():
# def test_convert_thai_years_parses():
# def test_convert_to_float32_dtype():
# def test_drop_nan_rows_drops_na():
# def test_inspect_file_exists_false(tmp_path):
# def test_inspect_file_exists_true(tmp_path):
# def test_prepare_datetime_index_buddhist_year():
# def test_prepare_datetime_index_sets_index():
# def test_read_csv_with_date_parse_valid(tmp_path):
# def test_write_test_file_creates(tmp_path):
# from src.data_loader import *
# import numpy as np
# import os
# import pandas as pd
# import pytest
# import src.main as main
# import sys
# import types
# sys.modules.setdefault("torch", types.SimpleNamespace())
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))