#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NICEGOLD ProjectP - Backtester Module
Enterprise-grade backtesting system for trading strategies
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class Backtester:
    """
    Enterprise-grade backtesting system for trading strategies
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Backtester

        Args:
            config: Configuration dictionary for backtesting
        """
        self.config = config or self._get_default_config()
        self.results = {}
        self.trades = []
        self.portfolio_value = []
        self.positions = []

        # Setup logging
        self._setup_logging()

        logger.info("Backtester initialized successfully")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for backtesting"""
        return {
            "initial_capital": 100000,
            "commission": 0.001,  # 0.1%
            "slippage": 0.0005,  # 0.05%
            "position_size": 0.1,  # 10% of portfolio per trade
            "stop_loss": 0.05,  # 5% stop loss
            "take_profit": 0.15,  # 15% take profit
            "max_positions": 5,  # Maximum concurrent positions
            "min_trade_interval": 1,  # Minimum days between trades
            "benchmark_return": 0.08,  # 8% annual benchmark
            "risk_free_rate": 0.02,  # 2% risk-free rate
            "verbose": True,
        }

    def _setup_logging(self):
        """Setup logging for backtester"""
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

    def prepare_backtest_data(
        self, data: pd.DataFrame, predictions: np.ndarray, price_column: str = "close"
    ) -> pd.DataFrame:
        """
        Prepare data for backtesting with predictions

        Args:
            data: Historical price data
            predictions: Model predictions
            price_column: Column name for price data

        Returns:
            DataFrame ready for backtesting
        """
        try:
            logger.info("Preparing backtest data")

            # Validate input
            if data.empty:
                raise ValueError("Input data is empty")

            if len(predictions) != len(data):
                raise ValueError("Predictions length must match data length")

            if price_column not in data.columns:
                raise ValueError(f"Price column '{price_column}' not found in data")

            # Create backtest dataframe
            backtest_data = data.copy()
            backtest_data["predictions"] = predictions
            backtest_data["price"] = backtest_data[price_column]

            # Calculate returns
            backtest_data["returns"] = backtest_data["price"].pct_change()

            # Generate trading signals based on predictions
            backtest_data["signal"] = self._generate_signals(predictions)

            # Ensure datetime index
            if not isinstance(backtest_data.index, pd.DatetimeIndex):
                if "date" in backtest_data.columns:
                    backtest_data["date"] = pd.to_datetime(backtest_data["date"])
                    backtest_data = backtest_data.set_index("date")
                else:
                    backtest_data.index = pd.date_range(
                        start="2020-01-01", periods=len(backtest_data), freq="D"
                    )

            # Sort by date
            backtest_data = backtest_data.sort_index()

            logger.info(f"Backtest data prepared: {len(backtest_data)} periods")

            return backtest_data

        except Exception as e:
            logger.error(f"Error preparing backtest data: {str(e)}")
            raise

    def _generate_signals(self, predictions: np.ndarray) -> np.ndarray:
        """
        Generate trading signals from model predictions

        Args:
            predictions: Model predictions

        Returns:
            Array of trading signals (1=buy, -1=sell, 0=hold)
        """
        try:
            # Normalize predictions to z-scores
            pred_mean = np.mean(predictions)
            pred_std = np.std(predictions)

            if pred_std == 0:
                return np.zeros(len(predictions))

            z_scores = (predictions - pred_mean) / pred_std

            # Generate signals based on z-score thresholds
            signals = np.zeros(len(predictions))

            # Buy signal: z-score > 1 (prediction is 1 std dev above mean)
            signals[z_scores > 1] = 1

            # Sell signal: z-score < -1 (prediction is 1 std dev below mean)
            signals[z_scores < -1] = -1

            return signals

        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return np.zeros(len(predictions))

    def run_backtest(self, backtest_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run the backtest simulation

        Args:
            backtest_data: Prepared backtest data

        Returns:
            Dictionary containing backtest results
        """
        try:
            logger.info("Starting backtest simulation")

            # Initialize variables
            capital = self.config["initial_capital"]
            positions = {}  # symbol -> position info
            portfolio_values = []
            trades = []
            daily_returns = []

            # Track portfolio metrics
            max_drawdown = 0
            peak_value = capital

            for date, row in backtest_data.iterrows():
                current_price = row["price"]
                signal = row["signal"]
                current_return = row["returns"] if not pd.isna(row["returns"]) else 0

                # Update existing positions
                position_value = 0
                for symbol, pos in positions.items():
                    pos_return = current_return  # Simplified - assume all positions follow same price
                    pos["current_value"] = pos["quantity"] * current_price
                    position_value += pos["current_value"]

                    # Check stop loss and take profit
                    price_change = (current_price - pos["entry_price"]) / pos[
                        "entry_price"
                    ]

                    if pos["side"] == "long":
                        if price_change <= -self.config["stop_loss"]:
                            # Stop loss triggered
                            trades.append(
                                self._close_position(
                                    pos, current_price, date, "stop_loss"
                                )
                            )
                            del positions[symbol]
                        elif price_change >= self.config["take_profit"]:
                            # Take profit triggered
                            trades.append(
                                self._close_position(
                                    pos, current_price, date, "take_profit"
                                )
                            )
                            del positions[symbol]

                    elif pos["side"] == "short":
                        if price_change >= self.config["stop_loss"]:
                            # Stop loss triggered
                            trades.append(
                                self._close_position(
                                    pos, current_price, date, "stop_loss"
                                )
                            )
                            del positions[symbol]
                        elif price_change <= -self.config["take_profit"]:
                            # Take profit triggered
                            trades.append(
                                self._close_position(
                                    pos, current_price, date, "take_profit"
                                )
                            )
                            del positions[symbol]

                # Calculate current portfolio value
                cash = capital - sum(
                    pos["quantity"] * pos["entry_price"] for pos in positions.values()
                )
                portfolio_value = cash + position_value
                portfolio_values.append(portfolio_value)

                # Calculate daily return
                if len(portfolio_values) > 1:
                    daily_return = (
                        portfolio_value - portfolio_values[-2]
                    ) / portfolio_values[-2]
                    daily_returns.append(daily_return)
                else:
                    daily_returns.append(0)

                # Update drawdown
                if portfolio_value > peak_value:
                    peak_value = portfolio_value

                current_drawdown = (peak_value - portfolio_value) / peak_value
                max_drawdown = max(max_drawdown, current_drawdown)

                # Process new signals
                if signal != 0 and len(positions) < self.config["max_positions"]:
                    symbol = "GOLD"  # Simplified - single asset

                    if symbol not in positions:
                        trade = self._execute_trade(signal, current_price, date, cash)
                        if trade:
                            trades.append(trade)
                            positions[symbol] = {
                                "side": "long" if signal > 0 else "short",
                                "quantity": trade["quantity"],
                                "entry_price": trade["price"],
                                "entry_date": date,
                                "current_value": trade["quantity"] * current_price,
                            }

            # Close any remaining positions at the end
            final_date = backtest_data.index[-1]
            final_price = backtest_data.iloc[-1]["price"]

            for symbol, pos in positions.items():
                trades.append(
                    self._close_position(
                        pos, final_price, final_date, "end_of_backtest"
                    )
                )

            # Calculate final metrics
            results = self._calculate_backtest_metrics(
                portfolio_values, daily_returns, trades, capital, max_drawdown
            )

            # Store results
            self.results = results
            self.trades = trades
            self.portfolio_value = portfolio_values

            logger.info(
                f"Backtest completed: {len(trades)} trades, {results['total_return']:.2%} return"
            )

            return results

        except Exception as e:
            logger.error(f"Error during backtest: {str(e)}")
            raise

    def _execute_trade(
        self, signal: float, price: float, date: pd.Timestamp, available_cash: float
    ) -> Optional[Dict[str, Any]]:
        """Execute a trade based on signal"""
        try:
            # Calculate position size
            position_value = available_cash * self.config["position_size"]

            # Apply slippage
            execution_price = price * (1 + self.config["slippage"] * np.sign(signal))

            # Calculate quantity
            quantity = position_value / execution_price

            # Calculate commission
            commission = position_value * self.config["commission"]

            # Check if we have enough cash
            total_cost = position_value + commission
            if total_cost > available_cash:
                return None

            trade = {
                "date": date,
                "signal": signal,
                "price": execution_price,
                "quantity": quantity,
                "value": position_value,
                "commission": commission,
                "type": "open",
                "side": "long" if signal > 0 else "short",
            }

            return trade

        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return None

    def _close_position(
        self, position: Dict[str, Any], price: float, date: pd.Timestamp, reason: str
    ) -> Dict[str, Any]:
        """Close an existing position"""
        # Apply slippage (opposite direction for closing)
        exit_signal = -1 if position["side"] == "long" else 1
        execution_price = price * (1 + self.config["slippage"] * np.sign(exit_signal))

        # Calculate P&L
        if position["side"] == "long":
            pnl = (execution_price - position["entry_price"]) * position["quantity"]
        else:
            pnl = (position["entry_price"] - execution_price) * position["quantity"]

        # Calculate commission
        position_value = position["quantity"] * execution_price
        commission = position_value * self.config["commission"]

        trade = {
            "date": date,
            "signal": exit_signal,
            "price": execution_price,
            "quantity": position["quantity"],
            "value": position_value,
            "commission": commission,
            "type": "close",
            "side": position["side"],
            "entry_price": position["entry_price"],
            "entry_date": position["entry_date"],
            "pnl": pnl - commission,
            "return": (pnl - commission)
            / (position["entry_price"] * position["quantity"]),
            "days_held": (date - position["entry_date"]).days,
            "reason": reason,
        }

        return trade

    def _calculate_backtest_metrics(
        self,
        portfolio_values: List[float],
        daily_returns: List[float],
        trades: List[Dict],
        initial_capital: float,
        max_drawdown: float,
    ) -> Dict[str, Any]:
        """Calculate comprehensive backtest metrics"""
        try:
            final_value = portfolio_values[-1] if portfolio_values else initial_capital
            total_return = (final_value - initial_capital) / initial_capital

            # Trade statistics
            winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
            losing_trades = [t for t in trades if t.get("pnl", 0) < 0]

            win_rate = len(winning_trades) / len(trades) if trades else 0

            avg_win = (
                np.mean([t["pnl"] for t in winning_trades]) if winning_trades else 0
            )
            avg_loss = (
                np.mean([t["pnl"] for t in losing_trades]) if losing_trades else 0
            )

            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

            # Return statistics
            daily_returns_array = np.array(daily_returns)
            annual_return = np.mean(daily_returns_array) * 252 if daily_returns else 0
            volatility = (
                np.std(daily_returns_array) * np.sqrt(252)
                if len(daily_returns) > 1
                else 0
            )

            # Risk metrics
            sharpe_ratio = (
                (annual_return - self.config["risk_free_rate"]) / volatility
                if volatility > 0
                else 0
            )

            # Calmar ratio (annual return / max drawdown)
            calmar_ratio = (
                annual_return / max_drawdown if max_drawdown > 0 else float("inf")
            )

            metrics = {
                "initial_capital": initial_capital,
                "final_value": final_value,
                "total_return": total_return,
                "annual_return": annual_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "calmar_ratio": calmar_ratio,
                "total_trades": len(trades),
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "win_rate": win_rate,
                "average_win": avg_win,
                "average_loss": avg_loss,
                "profit_factor": profit_factor,
                "total_commission": sum(t.get("commission", 0) for t in trades),
                "benchmark_return": self.config["benchmark_return"],
                "alpha": annual_return - self.config["benchmark_return"],
            }

            return metrics

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}

    def get_trade_analysis(self) -> Dict[str, Any]:
        """Get detailed trade analysis"""
        if not self.trades:
            return {}

        try:
            closed_trades = [t for t in self.trades if t["type"] == "close"]

            if not closed_trades:
                return {}

            # Monthly performance
            monthly_pnl = {}
            for trade in closed_trades:
                month_key = trade["date"].strftime("%Y-%m")
                if month_key not in monthly_pnl:
                    monthly_pnl[month_key] = 0
                monthly_pnl[month_key] += trade.get("pnl", 0)

            # Trade duration analysis
            durations = [t["days_held"] for t in closed_trades if "days_held" in t]

            # Consecutive wins/losses
            consecutive_wins = 0
            consecutive_losses = 0
            max_consecutive_wins = 0
            max_consecutive_losses = 0

            for trade in closed_trades:
                if trade.get("pnl", 0) > 0:
                    consecutive_wins += 1
                    consecutive_losses = 0
                    max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                else:
                    consecutive_losses += 1
                    consecutive_wins = 0
                    max_consecutive_losses = max(
                        max_consecutive_losses, consecutive_losses
                    )

            analysis = {
                "monthly_pnl": monthly_pnl,
                "avg_trade_duration": np.mean(durations) if durations else 0,
                "max_trade_duration": max(durations) if durations else 0,
                "min_trade_duration": min(durations) if durations else 0,
                "max_consecutive_wins": max_consecutive_wins,
                "max_consecutive_losses": max_consecutive_losses,
                "total_pnl": sum(t.get("pnl", 0) for t in closed_trades),
                "best_trade": (
                    max(closed_trades, key=lambda x: x.get("pnl", 0))
                    if closed_trades
                    else None
                ),
                "worst_trade": (
                    min(closed_trades, key=lambda x: x.get("pnl", 0))
                    if closed_trades
                    else None
                ),
            }

            return analysis

        except Exception as e:
            logger.error(f"Error in trade analysis: {str(e)}")
            return {}

    def export_results(self, filename: str = None) -> str:
        """Export backtest results to CSV"""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"backtest_results_{timestamp}.csv"

            # Create results DataFrame
            if self.trades:
                trades_df = pd.DataFrame(self.trades)
                trades_df.to_csv(filename, index=False)
                logger.info(f"Backtest results exported to {filename}")
                return filename
            else:
                logger.warning("No trades to export")
                return ""

        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
            return ""

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of backtest results"""
        summary = {
            "backtest_completed": bool(self.results),
            "total_trades": len(self.trades),
            "config": self.config,
        }

        if self.results:
            summary.update(
                {
                    "total_return": self.results.get("total_return", 0),
                    "sharpe_ratio": self.results.get("sharpe_ratio", 0),
                    "max_drawdown": self.results.get("max_drawdown", 0),
                    "win_rate": self.results.get("win_rate", 0),
                }
            )

        return summary
