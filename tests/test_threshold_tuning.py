

#             "best_f1": 0.5, 
#             "best_threshold": 0.5, 
#             "precision": 0.5, 
#             "recall": 0.5, 
#         'ATR_14': np.ones(6)
#         'Close': np.arange(6) + 1, 
#         'entry_time': pd.date_range('2023 - 01 - 01', periods = 6, freq = 'min'), 
#         'exit_reason': ['TP', 'SL', 'TP', 'SL', 'TP', 'SL']
#         'High': np.arange(6) + 1.1, 
#         'Low': np.arange(6) + 0.9, 
#         'Open': np.arange(6) + 1, 
#         called['hit'] = True
#         enable_dynamic_feature_selection = False, 
#         enable_optuna_tuning = False, 
#         enable_threshold_tuning = True
#         json.dump(['Open', 'High', 'Low', 'Close', 'ATR_14'], f)
#         m1_data_path = str(m1_path), 
#         model_type_to_train = 'catboost', 
#         n = len(X)
#         output_dir = str(out_dir), 
#         p = np.linspace(0.1, 0.8, n)
#         return np.column_stack([1 - p, p])
#         return np.zeros(len(X), dtype = int)
#         return {
#         self.data = X
#         self.feature_names_ = list(X.columns)
#         self.feature_names_ = None
#         self.params = kwargs
#         trade_log_df_override = trade_log, 
#         }
#     )
#     assert called.get('hit', False)
#     assert res["best_f1"] == pytest.approx(1.0)
#     assert res["best_threshold"] == pytest.approx(0.4)
#     assert res["precision"] == pytest.approx(1.0)
#     assert res["recall"] == pytest.approx(1.0)
#     called = {}
#     def __init__(self, **kwargs):
#     def __init__(self, X, label = None, cat_features = None):
#     def fake_find(proba, y):
#     def fit(self, X, y, cat_features = None, eval_set = None):
#     def predict(self, X):
#     def predict_proba(self, X):
#     from src import data_loader
#     m1 = pd.DataFrame({
#     m1.to_csv(m1_path)
#     m1_path = tmp_path / 'm1.csv'
#     monkeypatch.setattr(data_loader, 'validate_m1_data_path', lambda p: True)
#     monkeypatch.setattr(strategy, 'CatBoostClassifier', DummyCat)
#     monkeypatch.setattr(strategy, 'check_model_overfit', lambda *a, **k: None)
#     monkeypatch.setattr(strategy, 'find_best_threshold', fake_find)
#     monkeypatch.setattr(strategy, 'joblib_dump', lambda obj, path: open(path, 'wb').write(b'd'))
#     monkeypatch.setattr(strategy, 'Pool', DummyPool)
#     monkeypatch.setattr(strategy, 'safe_load_csv_auto', lambda p, **k: pd.read_csv(p, index_col = 0), raising = False)
#     monkeypatch.setattr(strategy, 'shap', None, raising = False)
#     monkeypatch.setattr(strategy, 'USE_GPU_ACCELERATION', False, raising = False)
#     out_dir = tmp_path / 'out'
#     out_dir.mkdir()
#     proba = np.array([0.1, 0.4, 0.6, 0.8])
#     res = find_best_threshold(proba, y)
#     strategy.train_and_export_meta_model(
#     trade_log = pd.DataFrame({
#     with open(out_dir / 'features_main.json', 'w', encoding = 'utf - 8') as f:
#     y = np.array([0, 0, 1, 1])
#     })
#     }, index = pd.date_range('2023 - 01 - 01', periods = 6, freq = 'min'))
# class DummyCat:
# class DummyPool:
# def test_find_best_threshold():
# def test_threshold_tuning_called(tmp_path, monkeypatch):
# from src import strategy
# from src.evaluation import find_best_threshold
# import json
# import numpy as np
# import pandas as pd
# import pytest