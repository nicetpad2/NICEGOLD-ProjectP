#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NICEGOLD ProjectP - Production Trading Strategy
Optimized for frequent profitable orders with $100 capital
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class ProductionTradingStrategy:
    """
    Production-optimized trading strategy for frequent profitable orders
    """
    
    def __init__(self, initial_capital: float = 100.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.logger = logging.getLogger(__name__)
        
        # Trading parameters
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.min_profit_target = 0.005  # 0.5% minimum profit target
        self.max_position_size = 0.1  # 10% max position size
        self.min_trades_per_day = 3  # Minimum trades for frequency requirement
        
        # Strategy state
        self.positions = []
        self.trade_history = []
        self.daily_stats = {}
        
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """
        Calculate optimal position size based on risk management
        """
        # Risk amount in dollars
        risk_amount = self.current_capital * self.risk_per_trade
        
        # Price risk per share
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return 0
        
        # Position size calculation
        position_size = risk_amount / price_risk
        
        # Apply maximum position size constraint
        max_dollar_position = self.current_capital * self.max_position_size
        max_shares = max_dollar_position / entry_price
        
        position_size = min(position_size, max_shares)
        
        return max(0, position_size)
    
    def generate_signals(self, df: pd.DataFrame, model_predictions: np.ndarray,
                        confidence_threshold: float = 0.6) -> pd.DataFrame:
        """
        Generate trading signals with multiple confirmation layers
        """
        self.logger.info("🎯 Generating trading signals...")
        
        df_signals = df.copy()
        df_signals['prediction'] = model_predictions
        df_signals['signal'] = 0  # 0: Hold, 1: Buy, -1: Sell
        df_signals['confidence'] = model_predictions
        
        # Primary signal based on model prediction
        df_signals.loc[df_signals['prediction'] > confidence_threshold, 'signal'] = 1
        df_signals.loc[df_signals['prediction'] < (1 - confidence_threshold), 'signal'] = -1
        
        # Add technical confirmation
        df_signals = self._add_technical_confirmation(df_signals)
        
        # Add volume confirmation
        df_signals = self._add_volume_confirmation(df_signals)
        
        # Add volatility filter
        df_signals = self._add_volatility_filter(df_signals)
        
        # Filter for high-quality signals only
        df_signals = self._filter_quality_signals(df_signals)
        
        return df_signals
    
    def _add_technical_confirmation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical analysis confirmation
        """
        # Simple moving averages
        df['sma_5'] = df['Close'].rolling(5).mean()
        df['sma_20'] = df['Close'].rolling(20).mean()
        
        # RSI for momentum
        df['rsi'] = self._calculate_rsi(df['Close'])
        
        # Technical confirmation
        tech_confirmation = (
            (df['signal'] == 1) & 
            (df['Close'] > df['sma_5']) & 
            (df['sma_5'] > df['sma_20']) &
            (df['rsi'] < 80)  # Not overbought
        ) | (
            (df['signal'] == -1) & 
            (df['Close'] < df['sma_5']) & 
            (df['sma_5'] < df['sma_20']) &
            (df['rsi'] > 20)  # Not oversold
        )
        
        # Keep only confirmed signals
        df.loc[~tech_confirmation, 'signal'] = 0
        
        return df
    
    def _add_volume_confirmation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume confirmation for signals
        """
        if 'Volume' in df.columns:
            df['volume_ma'] = df['Volume'].rolling(20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_ma']
            
            # Require above-average volume for signals
            volume_confirmed = df['volume_ratio'] > 1.2
            df.loc[~volume_confirmed, 'signal'] = 0
        
        return df
    
    def _add_volatility_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter signals based on volatility
        """
        # Calculate volatility
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['volatility_percentile'] = df['volatility'].rolling(100).rank(pct=True)
        
        # Avoid extremely high volatility periods
        high_vol_filter = df['volatility_percentile'] < 0.9
        df.loc[~high_vol_filter, 'signal'] = 0
        
        # Avoid extremely low volatility periods (no opportunities)
        low_vol_filter = df['volatility_percentile'] > 0.1
        df.loc[~low_vol_filter, 'signal'] = 0
        
        return df
    
    def _filter_quality_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply additional quality filters
        """
        # Minimum confidence threshold
        min_confidence = 0.65
        low_confidence = (
            ((df['signal'] == 1) & (df['confidence'] < min_confidence)) |
            ((df['signal'] == -1) & (df['confidence'] > (1 - min_confidence)))
        )
        df.loc[low_confidence, 'signal'] = 0
        
        # Avoid consecutive identical signals (reduce overtrading)
        df['prev_signal'] = df['signal'].shift(1)
        consecutive_signals = (df['signal'] != 0) & (df['signal'] == df['prev_signal'])
        df.loc[consecutive_signals, 'signal'] = 0
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI indicator
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def execute_strategy(self, df_signals: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute the trading strategy with optimized entry/exit
        """
        self.logger.info("💰 Executing trading strategy...")
        
        portfolio_values = [self.initial_capital]
        trades = []
        open_positions = []
        
        for i in range(len(df_signals)):
            current_row = df_signals.iloc[i]
            current_price = current_row['Close']
            signal = current_row['signal']
            timestamp = current_row.get('Date', i)
            
            # Check for exit conditions on open positions
            open_positions = self._check_exit_conditions(
                open_positions, current_row, trades
            )
            
            # Process new signals
            if signal != 0 and len(open_positions) < 3:  # Max 3 concurrent positions
                position = self._enter_position(current_row, signal)
                if position:
                    open_positions.append(position)
                    trades.append({
                        'entry_time': timestamp,
                        'type': 'entry',
                        'signal': signal,
                        'price': current_price,
                        'position_size': position['size']
                    })
            
            # Update portfolio value
            portfolio_value = self._calculate_portfolio_value(
                open_positions, current_price
            )
            portfolio_values.append(portfolio_value)
            self.current_capital = portfolio_value
        
        # Close all remaining positions
        final_price = df_signals.iloc[-1]['Close']
        for position in open_positions:
            exit_trade = self._close_position(position, final_price, "strategy_end")
            if exit_trade:
                trades.append(exit_trade)
        
        # Calculate strategy metrics
        strategy_results = self._calculate_strategy_metrics(
            portfolio_values, trades, df_signals
        )
        
        return strategy_results
    
    def _check_exit_conditions(self, positions: List[Dict], current_row: pd.DataFrame,
                              trades: List[Dict]) -> List[Dict]:
        """
        Check exit conditions for open positions
        """
        remaining_positions = []
        current_price = current_row['Close']
        
        for position in positions:
            should_exit, exit_reason = self._should_exit_position(position, current_row)
            
            if should_exit:
                exit_trade = self._close_position(position, current_price, exit_reason)
                if exit_trade:
                    trades.append(exit_trade)
            else:
                remaining_positions.append(position)
        
        return remaining_positions
    
    def _should_exit_position(self, position: Dict, current_row: pd.DataFrame) -> Tuple[bool, str]:
        """
        Determine if position should be exited
        """
        current_price = current_row['Close']
        entry_price = position['entry_price']
        signal_type = position['signal']
        
        # Calculate current P&L percentage
        if signal_type == 1:  # Long position
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # Short position
            pnl_pct = (entry_price - current_price) / entry_price
        
        # Profit target (dynamic based on volatility)
        volatility = current_row.get('volatility', 0.01)
        profit_target = max(self.min_profit_target, volatility * 2)
        
        # Stop loss
        stop_loss = -self.risk_per_trade
        
        # Exit conditions
        if pnl_pct >= profit_target:
            return True, "profit_target"
        elif pnl_pct <= stop_loss:
            return True, "stop_loss"
        elif position.get('holding_period', 0) > 50:  # Max holding period
            return True, "max_holding"
        
        # Update holding period
        position['holding_period'] = position.get('holding_period', 0) + 1
        
        return False, ""
    
    def _enter_position(self, current_row: pd.DataFrame, signal: int) -> Dict[str, Any]:
        """
        Enter a new position
        """
        current_price = current_row['Close']
        
        # Calculate stop loss
        volatility = current_row.get('volatility', 0.01)
        if signal == 1:  # Long position
            stop_loss = current_price * (1 - volatility * 3)
        else:  # Short position
            stop_loss = current_price * (1 + volatility * 3)
        
        # Calculate position size
        position_size = self.calculate_position_size(current_price, stop_loss)
        
        if position_size > 0:
            position = {
                'signal': signal,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'size': position_size,
                'holding_period': 0,
                'entry_time': current_row.get('Date', 0)
            }
            return position
        
        return None
    
    def _close_position(self, position: Dict, exit_price: float, 
                       exit_reason: str) -> Dict[str, Any]:
        """
        Close a position and calculate P&L
        """
        entry_price = position['entry_price']
        size = position['size']
        signal = position['signal']
        
        # Calculate P&L
        if signal == 1:  # Long position
            pnl = (exit_price - entry_price) * size
        else:  # Short position
            pnl = (entry_price - exit_price) * size
        
        # Update capital
        self.current_capital += pnl
        
        return {
            'type': 'exit',
            'signal': signal,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': size,
            'pnl': pnl,
            'pnl_pct': pnl / (entry_price * size),
            'exit_reason': exit_reason,
            'holding_period': position.get('holding_period', 0)
        }
    
    def _calculate_portfolio_value(self, positions: List[Dict], 
                                  current_price: float) -> float:
        """
        Calculate current portfolio value including open positions
        """
        portfolio_value = self.current_capital
        
        for position in positions:
            entry_price = position['entry_price']
            size = position['size']
            signal = position['signal']
            
            if signal == 1:  # Long position
                unrealized_pnl = (current_price - entry_price) * size
            else:  # Short position
                unrealized_pnl = (entry_price - current_price) * size
            
            portfolio_value += unrealized_pnl
        
        return portfolio_value
    
    def _calculate_strategy_metrics(self, portfolio_values: List[float],
                                  trades: List[Dict], df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive strategy performance metrics
        """
        # Basic metrics
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        final_capital = portfolio_values[-1]
        
        # Trade statistics
        exit_trades = [t for t in trades if t['type'] == 'exit']
        profitable_trades = [t for t in exit_trades if t['pnl'] > 0]
        
        win_rate = len(profitable_trades) / len(exit_trades) if exit_trades else 0
        avg_profit = np.mean([t['pnl'] for t in profitable_trades]) if profitable_trades else 0
        avg_loss = np.mean([t['pnl'] for t in exit_trades if t['pnl'] < 0])
        avg_loss = avg_loss if not np.isnan(avg_loss) else 0
        
        # Risk metrics
        daily_returns = pd.Series(portfolio_values).pct_change().dropna()
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        
        # Frequency metrics
        total_days = len(df) / 1440 if len(df) > 1440 else 1  # Assuming M1 data
        trades_per_day = len(exit_trades) / total_days
        
        # Profit factor
        total_profit = sum([t['pnl'] for t in profitable_trades])
        total_loss = abs(sum([t['pnl'] for t in exit_trades if t['pnl'] < 0]))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        return {
            'total_return': total_return,
            'final_capital': final_capital,
            'win_rate': win_rate,
            'total_trades': len(exit_trades),
            'profitable_trades': len(profitable_trades),
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades_per_day': trades_per_day,
            'frequent_trading': trades_per_day >= self.min_trades_per_day,
            'portfolio_curve': portfolio_values,
            'trade_details': exit_trades
        }
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, 
                               risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio
        """
        if returns.std() == 0:
            return 0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return excess_returns.mean() / returns.std() * np.sqrt(252)
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """
        Calculate maximum drawdown
        """
        peak = portfolio_values[0]
        max_dd = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def optimize_parameters(self, df: pd.DataFrame, predictions: np.ndarray) -> Dict[str, Any]:
        """
        Optimize strategy parameters for maximum profitability and frequency
        """
        self.logger.info("🔧 Optimizing strategy parameters...")
        
        best_params = None
        best_score = -float('inf')
        
        # Parameter ranges to test
        confidence_thresholds = [0.55, 0.60, 0.65, 0.70]
        risk_levels = [0.015, 0.02, 0.025]
        profit_targets = [0.003, 0.005, 0.008]
        
        for conf_thresh in confidence_thresholds:
            for risk_level in risk_levels:
                for profit_target in profit_targets:
                    # Reset capital for each test
                    self.current_capital = self.initial_capital
                    self.risk_per_trade = risk_level
                    self.min_profit_target = profit_target
                    
                    # Generate signals and execute strategy
                    df_signals = self.generate_signals(df, predictions, conf_thresh)
                    results = self.execute_strategy(df_signals)
                    
                    # Score function (weighted combination of return and frequency)
                    frequency_bonus = 1.0 if results['frequent_trading'] else 0.5
                    score = (results['total_return'] * frequency_bonus * 
                            results['win_rate'] * results['profit_factor'])
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'confidence_threshold': conf_thresh,
                            'risk_per_trade': risk_level,
                            'min_profit_target': profit_target,
                            'results': results
                        }
        
        # Apply best parameters
        if best_params:
            self.risk_per_trade = best_params['risk_per_trade']
            self.min_profit_target = best_params['min_profit_target']
            
            self.logger.info(f"✅ Optimized parameters:")
            self.logger.info(f"   Confidence threshold: {best_params['confidence_threshold']}")
            self.logger.info(f"   Risk per trade: {best_params['risk_per_trade']:.1%}")
            self.logger.info(f"   Profit target: {best_params['min_profit_target']:.1%}")
        
        return best_params or {}
    
    def generate_strategy_report(self, results: Dict[str, Any]) -> str:
        """
        Generate detailed strategy performance report
        """
        report = f"""
# 💰 PRODUCTION TRADING STRATEGY REPORT

## 📊 PERFORMANCE SUMMARY
- **Total Return**: {results['total_return']:.2%}
- **Final Capital**: ${results['final_capital']:.2f}
- **Win Rate**: {results['win_rate']:.2%}
- **Sharpe Ratio**: {results['sharpe_ratio']:.2f}
- **Max Drawdown**: {results['max_drawdown']:.2%}
- **Profit Factor**: {results['profit_factor']:.2f}

## 📈 TRADING FREQUENCY
- **Total Trades**: {results['total_trades']}
- **Trades per Day**: {results['trades_per_day']:.1f}
- **Frequent Trading**: {'✅ YES' if results['frequent_trading'] else '❌ NO'}
- **Profitable Trades**: {results['profitable_trades']}

## 💡 TRADE ANALYSIS
- **Average Profit**: ${results['avg_profit']:.2f}
- **Average Loss**: ${results['avg_loss']:.2f}
- **Risk-Reward Ratio**: {abs(results['avg_profit']/results['avg_loss']):.2f} if results['avg_loss'] != 0 else 'N/A'

## ⚡ QUICK STATS
- **Starting Capital**: $100.00
- **Ending Capital**: ${results['final_capital']:.2f}
- **Net P&L**: ${results['final_capital'] - 100:.2f}
- **ROI**: {((results['final_capital'] - 100)/100)*100:.1f}%

## 🎯 PRODUCTION READINESS
- **Profitable**: {'✅' if results['total_return'] > 0 else '❌'}
- **Frequent Orders**: {'✅' if results['frequent_trading'] else '❌'}
- **Good Win Rate**: {'✅' if results['win_rate'] > 0.5 else '❌'}
- **Positive Sharpe**: {'✅' if results['sharpe_ratio'] > 0 else '❌'}

---
Generated by NICEGOLD Production Trading Strategy
"""
        
        return report
