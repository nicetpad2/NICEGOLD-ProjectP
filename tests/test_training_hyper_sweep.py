
import logging
import os
import pandas as pd
import src.training as training
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def test_run_hyperparameter_sweep_single_row(monkeypatch, caplog):
    df = pd.DataFrame({'f': [1], 'target': [1]})
    grid = [{'a': 1}, {'a': 2}]

    calls = []

    def dummy_train_full(_):
        calls.append('full')

    def dummy_metrics(_):
        return {'accuracy': -1.0}

    res = training.run_hyperparameter_sweep(
        df, 
        grid, 
        '5.9.1', 
        train_full_fn = dummy_train_full, 
        compute_fallback_fn = dummy_metrics, 
        train_eval_fn = lambda d, p: {}, 
        select_best_fn = lambda r: {}, 
    )
    assert calls == []
    assert res == {'accuracy': -1.0}


def test_run_hyperparameter_sweep_multi_row():
    df = pd.DataFrame({'f': [1, 2, 3], 'target': [1, 0, 1]})
    grid = [{'a': 1}, {'a': 2}]
    def dummy_eval(_df, params):
        return {'params': params, 'metrics': {'accuracy': params['a']}}

    def dummy_best(res):
        assert len(res) == 2
        assert sorted(r['params']['a'] for r in res) == [1, 2]
        return res[ - 1]

    res = training.run_hyperparameter_sweep(
        df, 
        grid, 
        '5.9.1', 
        train_eval_fn = dummy_eval, 
        select_best_fn = dummy_best, 
    )
    assert res == {'params': {'a': 2}, 'metrics': {'accuracy': 2}}