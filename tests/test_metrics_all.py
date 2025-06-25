
from src.metrics import performance
import numpy as np
import pandas as pd
def test_sharpe_ratio_zero():
    returns = pd.Series([0, 0, 0, 0])
    result = performance.sharpe_ratio(returns)
    assert result == 0 or np.isnan(result)

def test_max_drawdown_monotonic():
    equity = pd.Series([100, 110, 120, 130])
    result = performance.max_drawdown(equity)
    assert result == 0

def test_win_rate_all_win():
    trades = pd.DataFrame({'pnl': [1, 2, 3]})
    result = performance.win_rate(trades)
    assert result == 1

def test_profit_factor_all_loss():
    trades = pd.DataFrame({'pnl': [ - 1, -2, -3]})
    result = performance.profit_factor(trades)
    assert result == 0