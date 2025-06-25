

#         'ATR_14': np.ones(5)
#         'Close': np.linspace(1, 5, 5), 
#         'entry_time': pd.date_range('2023 - 01 - 01', periods = 5, freq = 'min'), 
#         'exit_reason': ['TP', 'SL', 'TP', 'SL', 'TP']
#         'High': np.linspace(1, 5, 5) + 0.1, 
#         'Low': np.linspace(1, 5, 5) - 0.1, 
#         'Open': np.linspace(1, 5, 5), 
#         enable_dynamic_feature_selection = False, 
#         enable_optuna_tuning = False, 
#         m1_data_path = str(m1_path), 
#         model_type_to_train = 'catboost'
#         output_dir = str(out_dir), 
#         return np.column_stack([np.ones(len(X))*0.4, np.ones(len(X))*0.6])
#         return np.zeros(len(X), dtype = int)
#         self.data = X
#         self.feature_names_ = list(X.columns)
#         self.feature_names_ = None
#         self.params = kwargs
#         trade_log_df_override = trade_log, 
#     )
#     assert feats == []
#     assert saved is None
#     def __init__(self, **kwargs):
#     def __init__(self, X, label = None, cat_features = None):
#     def fit(self, X, y, cat_features = None, eval_set = None):
#     def predict(self, X):
#     def predict_proba(self, X):
#     m1 = pd.DataFrame({
#     m1.to_csv(m1_path)
#     m1_path = tmp_path / 'm1.csv'
#     monkeypatch.setattr(strategy, 'CatBoostClassifier', DummyCat)
#     monkeypatch.setattr(strategy, 'joblib_dump', lambda obj, path: open(path, 'wb').write(b'dummy'))
#     monkeypatch.setattr(strategy, 'Pool', DummyPool)
#     monkeypatch.setattr(strategy, 'safe_load_csv_auto', lambda p, **k: pd.read_csv(p, index_col = 0), raising = False)
#     monkeypatch.setattr(strategy, 'USE_GPU_ACCELERATION', False, raising = False)
#     out_dir = tmp_path / 'out'
#     out_dir.mkdir()
#     saved, feats = strategy.train_and_export_meta_model(
#     trade_log = pd.DataFrame({
#     })
#     }, index = pd.date_range('2023 - 01 - 01', periods = 5, freq = 'min'))
# class DummyCat:
# class DummyPool:
# def test_train_and_export_meta_model(tmp_path, monkeypatch):
# from src import strategy
# import numpy as np
# import os
# import pandas as pd