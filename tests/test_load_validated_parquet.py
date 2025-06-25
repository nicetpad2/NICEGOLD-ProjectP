#
# #
# #         'Close': [1.0]
# #         'Date': ['20240101'], 
# #         'High': [1.0], 
# #         'Low': [1.0], 
# #         'Open': [1.0], 
# #         'Timestamp': ['00:00:00'], 
# #         assert col in loaded.columns
# #         pd.testing.assert_series_equal(loaded[col].astype(str), df[col].astype(str), check_dtype = False)
# #     # Stub parquet functions to avoid pyarrow dependency
# #     df = pd.DataFrame({
# #     df.to_parquet(path)
# #     for col in df.columns:
# #     loaded = main.load_validated_csv(str(path), 'M1')
# #     monkeypatch.setattr(pd, 'read_parquet', lambda p: pd.read_csv(str(p).replace('.parquet', '.csv')))
# #     monkeypatch.setattr(pd.DataFrame, 'to_parquet', lambda self, p: self.to_csv(str(p).replace('.parquet', '.csv')))
# #     path = tmp_path / 'sample.parquet'
# #     })
# # def test_load_validated_csv_parquet(tmp_path, monkeypatch):
# import pandas as pd
# import src.main as main