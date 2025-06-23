import pandas as pd
import numpy as np
from src.metrics import performance

def test_sharpe_ratio():
    returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.00])
    result = performance.sharpe_ratio(returns)
    assert isinstance(result, float)

def test_max_drawdown():
    equity = pd.Series([100, 110, 105, 120, 90, 130])
    result = performance.max_drawdown(equity)
    # max_drawdown = (90-120)/120 = -0.25, return as positive 0.25
    assert np.isclose(result, 0.25, atol=1e-2)

def test_win_rate():
    trades = pd.DataFrame({'pnl': [10, -5, 20, -2, 0]})
    result = performance.win_rate(trades)
    assert np.isclose(result, 0.4, atol=1e-2)

def test_profit_factor():
    trades = pd.DataFrame({'pnl': [10, -5, 20, -2, 0]})
    result = performance.profit_factor(trades)
    assert np.isclose(result, 30/7, atol=1e-2)
