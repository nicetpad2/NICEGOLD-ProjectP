
from strategy import strategy as strategy_module
import numpy as np
import os
import pandas as pd
import pytest
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)


def test_public_api():
    """Ensure __all__ exports all public functions."""
    assert sorted(strategy_module.__all__) == ["apply_strategy", "run_backtest"]


def test_apply_strategy_adds_columns(monkeypatch):
    df = pd.DataFrame({"Close": [1, 2, 3]})
    monkeypatch.setattr(strategy_module, "generate_open_signals", lambda x: np.array([1, 0, 1]))
    monkeypatch.setattr(strategy_module, "generate_close_signals", lambda x: np.array([0, 1, 0]))
    res = strategy_module.apply_strategy(df)
    assert list(res["Entry"]) == [1, 0, 1]
    assert list(res["Exit"]) == [0, 1, 0]
    assert "Entry" in res and "Exit" in res


def test_apply_strategy_does_not_mutate(monkeypatch):
    df = pd.DataFrame({"Close": [1, 2, 3]})
    calls = {"open": 0, "close": 0}

    def fake_open(arg):
        calls["open"] += 1
        return np.array([0, 0, 0])

    def fake_close(arg):
        calls["close"] += 1
        return np.array([1, 1, 1])

    monkeypatch.setattr(strategy_module, "generate_open_signals", fake_open)
    monkeypatch.setattr(strategy_module, "generate_close_signals", fake_close)
    original = df.copy()

    result = strategy_module.apply_strategy(df)
    assert calls == {"open": 1, "close": 1}
    assert list(result["Entry"]) == [0, 0, 0]
    assert list(result["Exit"]) == [1, 1, 1]
    assert "Entry" not in original.columns and "Exit" not in original.columns


def test_apply_strategy_type_error():
    with pytest.raises(TypeError):
        strategy_module.apply_strategy([1, 2, 3])


def test_run_backtest_returns_list():
    df = pd.DataFrame({"Close": [1.0, 1.1]})
    assert strategy_module.run_backtest(df, 1000.0) == []


def test_run_backtest_type_error():
    with pytest.raises(TypeError):
        strategy_module.run_backtest({}, 1000.0)