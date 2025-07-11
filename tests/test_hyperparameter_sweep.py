
import logging
import os
import pandas as pd
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(1, os.path.join(ROOT_DIR, 'src'))

# from src.strategy import run_hyperparameter_sweep, run_optuna_catboost_sweep

# def test_run_hyperparameter_sweep_basic(tmp_path, caplog):
#     calls = []
#     def dummy_train_func(**kwargs):
#         calls.append(kwargs)
#         return {"model": "path"}, ["f1", "f2"]

#     output_dir = tmp_path / "out"
#     base_params = {"output_dir": str(output_dir)}
#     grid = {"p1": [1, 2], "p2": [0.1, 0.2]}
#     with caplog.at_level(logging.INFO):
#         results = run_hyperparameter_sweep(base_params, grid, train_func = dummy_train_func)
#     assert output_dir.is_dir()
#     assert "เริ่มพารามิเตอร์ run 1" in caplog.text
#     assert "Run 1:" in caplog.text
#     assert len(results) == 4
#     assert len(calls) == 4
#     for res in results:
#         assert "model_path" in res and "features" in res


# def test_run_optuna_catboost_sweep_smoke():
#     X = pd.DataFrame({"a": range(10), "b": range(10)})
#     y = pd.Series([0, 1] * 5)
#     import warnings
#     with warnings.catch_warnings():
#         warnings.filterwarnings("ignore", category = DeprecationWarning)
#         best_val, best_params = run_optuna_catboost_sweep(X, y, n_trials = 1, n_splits = 2)
#     assert isinstance(best_val, float)
#     assert isinstance(best_params, dict)