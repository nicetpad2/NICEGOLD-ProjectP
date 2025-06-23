"""
Performance metrics for trading/backtest (เทพ)
"""
import numpy as np
import pandas as pd
from typing import Optional

def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate annualized Sharpe ratio."""
    excess = returns - risk_free_rate
    return np.nan_to_num(excess.mean() / excess.std(ddof=1) * np.sqrt(252))

def max_drawdown(equity_curve: pd.Series) -> float:
    """Calculate maximum drawdown (as a positive number)."""
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    return -drawdown.min()

def win_rate(trades: pd.DataFrame) -> float:
    """Calculate win rate from trade log DataFrame (expects 'pnl' column)."""
    return (trades['pnl'] > 0).mean() if 'pnl' in trades else np.nan

def profit_factor(trades: pd.DataFrame) -> float:
    """Calculate profit factor from trade log DataFrame (expects 'pnl' column)."""
    gross_profit = trades.loc[trades['pnl'] > 0, 'pnl'].sum()
    gross_loss = -trades.loc[trades['pnl'] < 0, 'pnl'].sum()
    return gross_profit / gross_loss if gross_loss > 0 else np.nan
