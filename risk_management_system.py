#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚠️ REAL-TIME RISK MANAGEMENT SYSTEM
Advanced risk monitoring, position sizing, and portfolio protection
for NICEGOLD ProjectP v2.1

Author: NICEGOLD Enterprise
Date: June 25, 2025
"""

import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Rich imports with fallback
RICH_AVAILABLE = False
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, TextColumn
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    pass


class RiskManagementSystem:
    """⚠️ Advanced Real-time Risk Management System"""
    
    def __init__(self, console_output: bool = True):
        self.console = Console() if RICH_AVAILABLE else None
        self.console_output = console_output
        self.risk_parameters = self._initialize_risk_parameters()
        self.position_history = []
        self.risk_alerts = []
        self.portfolio_metrics = {}
        
    def _initialize_risk_parameters(self) -> Dict[str, Any]:
        """กำหนดค่าพารามิเตอร์ความเสี่ยงเริ่มต้น"""
        return {
            'max_position_size': 0.1,  # 10% of portfolio
            'max_daily_loss': 0.05,    # 5% daily loss limit
            'max_drawdown': 0.15,      # 15% maximum drawdown
            'var_confidence': 0.95,    # VaR confidence level
            'correlation_limit': 0.7,  # Maximum correlation between assets
            'volatility_threshold': 0.3,  # Maximum volatility threshold
            'leverage_limit': 2.0,     # Maximum leverage
            'stop_loss_pct': 0.02,     # 2% stop loss
            'take_profit_pct': 0.04    # 4% take profit
        }
    
    def calculate_position_size(self, signal_strength: float, 
                              account_balance: float,
                              current_price: float,
                              volatility: float) -> Dict[str, Any]:
        """
        💰 คำนวณขนาดโพซิชั่นที่เหมาะสม
        """
        if self.console_output and RICH_AVAILABLE:
            self._show_position_sizing_header()
        
        # Kelly Criterion based position sizing
        win_rate = min(0.6, max(0.4, signal_strength))  # Clamp between 40-60%
        avg_win = 0.03  # Average 3% win
        avg_loss = 0.02  # Average 2% loss
        
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        # Volatility-based adjustment
        vol_adjustment = 1 / (1 + volatility * 10)  # Reduce size for high volatility
        
        # Calculate base position size
        base_position_size = kelly_fraction * vol_adjustment
        
        # Apply risk limits
        max_allowed = self.risk_parameters['max_position_size']
        position_fraction = min(base_position_size, max_allowed)
        
        # Calculate dollar amount and shares
        dollar_amount = account_balance * position_fraction
        shares = int(dollar_amount / current_price)
        actual_dollar_amount = shares * current_price
        
        position_info = {
            'signal_strength': signal_strength,
            'kelly_fraction': kelly_fraction,
            'vol_adjustment': vol_adjustment,
            'position_fraction': position_fraction,
            'dollar_amount': actual_dollar_amount,
            'shares': shares,
            'current_price': current_price,
            'stop_loss_price': current_price * (1 - self.risk_parameters['stop_loss_pct']),
            'take_profit_price': current_price * (1 + self.risk_parameters['take_profit_pct']),
            'max_loss': actual_dollar_amount * self.risk_parameters['stop_loss_pct'],
            'risk_score': self._calculate_position_risk_score(position_fraction, volatility)
        }
        
        if self.console_output:
            self._display_position_sizing_results(position_info)
        
        return position_info
    
    def monitor_portfolio_risk(self, portfolio_data: Dict[str, Any],
                              market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        📊 ตรวจสอบความเสี่ยงของพอร์ตโฟลิโอแบบ real-time
        """
        if self.console_output and RICH_AVAILABLE:
            self._show_portfolio_monitoring_header()
        
        risk_report = {
            'timestamp': datetime.now(),
            'portfolio_value': portfolio_data.get('total_value', 0),
            'risk_metrics': {},
            'alerts': [],
            'recommendations': []
        }
        
        # 1. Calculate Value at Risk (VaR)
        returns = market_data['close'].pct_change().dropna()
        var_95 = np.percentile(returns, (1 - self.risk_parameters['var_confidence']) * 100)
        var_dollar = abs(var_95 * risk_report['portfolio_value'])
        
        risk_report['risk_metrics']['var_95_pct'] = var_95
        risk_report['risk_metrics']['var_95_dollar'] = var_dollar
        
        # 2. Calculate Current Drawdown
        if len(self.position_history) > 0:
            peak_value = max([pos['portfolio_value'] for pos in self.position_history[-30:]])
            current_drawdown = (risk_report['portfolio_value'] - peak_value) / peak_value
            risk_report['risk_metrics']['current_drawdown'] = current_drawdown
            
            # Check drawdown limit
            if abs(current_drawdown) > self.risk_parameters['max_drawdown']:
                risk_report['alerts'].append({
                    'type': 'CRITICAL',
                    'message': f"Maximum drawdown exceeded: {current_drawdown:.2%}",
                    'threshold': self.risk_parameters['max_drawdown']
                })
        
        # 3. Calculate Portfolio Volatility
        portfolio_volatility = returns.std() * np.sqrt(252)  # Annualized
        risk_report['risk_metrics']['annualized_volatility'] = portfolio_volatility
        
        if portfolio_volatility > self.risk_parameters['volatility_threshold']:
            risk_report['alerts'].append({
                'type': 'WARNING',
                'message': f"High portfolio volatility: {portfolio_volatility:.2%}",
                'threshold': self.risk_parameters['volatility_threshold']
            })
        
        # 4. Check Position Concentration
        positions = portfolio_data.get('positions', {})
        if positions:
            max_position_pct = max([pos['weight'] for pos in positions.values()])
            if max_position_pct > self.risk_parameters['max_position_size']:
                risk_report['alerts'].append({
                    'type': 'WARNING',
                    'message': f"Position concentration risk: {max_position_pct:.2%}",
                    'threshold': self.risk_parameters['max_position_size']
                })
        
        # 5. Generate Risk Score
        risk_report['overall_risk_score'] = self._calculate_overall_risk_score(risk_report)
        
        # 6. Generate Recommendations
        risk_report['recommendations'] = self._generate_risk_recommendations(risk_report)
        
        # Store in history
        self.portfolio_metrics = risk_report
        
        if self.console_output:
            self._display_risk_monitoring_results(risk_report)
        
        return risk_report
    
    def execute_risk_controls(self, current_positions: Dict[str, Any],
                            market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        🛡️ ดำเนินการควบคุมความเสี่ยงอัตโนมัติ
        """
        if self.console_output and RICH_AVAILABLE:
            self._show_risk_controls_header()
        
        control_actions = {
            'timestamp': datetime.now(),
            'actions_taken': [],
            'positions_adjusted': [],
            'alerts_generated': []
        }
        
        current_price = market_data['close'].iloc[-1]
        
        for position_id, position in current_positions.items():
            # Check stop loss
            if position['type'] == 'LONG' and current_price <= position['stop_loss_price']:
                control_actions['actions_taken'].append({
                    'action': 'STOP_LOSS_HIT',
                    'position_id': position_id,
                    'trigger_price': current_price,
                    'stop_price': position['stop_loss_price'],
                    'loss_amount': (position['entry_price'] - current_price) * position['shares']
                })
            
            elif position['type'] == 'SHORT' and current_price >= position['stop_loss_price']:
                control_actions['actions_taken'].append({
                    'action': 'STOP_LOSS_HIT',
                    'position_id': position_id,
                    'trigger_price': current_price,
                    'stop_price': position['stop_loss_price'],
                    'loss_amount': (current_price - position['entry_price']) * position['shares']
                })
            
            # Check take profit
            if position['type'] == 'LONG' and current_price >= position['take_profit_price']:
                control_actions['actions_taken'].append({
                    'action': 'TAKE_PROFIT_HIT',
                    'position_id': position_id,
                    'trigger_price': current_price,
                    'target_price': position['take_profit_price'],
                    'profit_amount': (current_price - position['entry_price']) * position['shares']
                })
            
            elif position['type'] == 'SHORT' and current_price <= position['take_profit_price']:
                control_actions['actions_taken'].append({
                    'action': 'TAKE_PROFIT_HIT',
                    'position_id': position_id,
                    'trigger_price': current_price,
                    'target_price': position['take_profit_price'],
                    'profit_amount': (position['entry_price'] - current_price) * position['shares']
                })
            
            # Check for trailing stop adjustment
            trailing_stop = self._calculate_trailing_stop(position, current_price)
            if trailing_stop != position['stop_loss_price']:
                control_actions['positions_adjusted'].append({
                    'position_id': position_id,
                    'adjustment': 'TRAILING_STOP',
                    'old_stop': position['stop_loss_price'],
                    'new_stop': trailing_stop
                })
        
        if self.console_output:
            self._display_risk_controls_results(control_actions)
        
        return control_actions
    
    def _calculate_position_risk_score(self, position_fraction: float, 
                                     volatility: float) -> float:
        """คำนวณคะแนนความเสี่ยงของโพซิชั่น"""
        size_risk = position_fraction / self.risk_parameters['max_position_size']
        vol_risk = volatility / self.risk_parameters['volatility_threshold']
        
        # Combine risks (weighted average)
        risk_score = (size_risk * 0.6 + vol_risk * 0.4)
        return min(1.0, risk_score)  # Cap at 1.0
    
    def _calculate_overall_risk_score(self, risk_report: Dict[str, Any]) -> float:
        """คำนวณคะแนนความเสี่ยงโดยรวม"""
        metrics = risk_report['risk_metrics']
        
        # Normalize each metric to 0-1 scale
        drawdown_risk = abs(metrics.get('current_drawdown', 0)) / self.risk_parameters['max_drawdown']
        vol_risk = metrics.get('annualized_volatility', 0) / self.risk_parameters['volatility_threshold']
        var_risk = abs(metrics.get('var_95_pct', 0)) / 0.05  # 5% daily VaR threshold
        
        # Alert penalty
        alert_penalty = len(risk_report['alerts']) * 0.1
        
        # Weighted combination
        overall_risk = (drawdown_risk * 0.4 + vol_risk * 0.3 + var_risk * 0.2 + alert_penalty * 0.1)
        
        return min(1.0, overall_risk)
    
    def _calculate_trailing_stop(self, position: Dict[str, Any], 
                               current_price: float) -> float:
        """คำนวณ trailing stop ใหม่"""
        trailing_pct = self.risk_parameters['stop_loss_pct']
        
        if position['type'] == 'LONG':
            # For long positions, stop loss moves up with price
            new_stop = current_price * (1 - trailing_pct)
            return max(position['stop_loss_price'], new_stop)
        else:
            # For short positions, stop loss moves down with price
            new_stop = current_price * (1 + trailing_pct)
            return min(position['stop_loss_price'], new_stop)
    
    def _generate_risk_recommendations(self, risk_report: Dict[str, Any]) -> List[str]:
        """สร้างคำแนะนำการจัดการความเสี่ยง"""
        recommendations = []
        
        risk_score = risk_report['overall_risk_score']
        
        if risk_score > 0.8:
            recommendations.append("🚨 CRITICAL: Consider reducing position sizes immediately")
            recommendations.append("💰 Consider hedging with protective options")
        elif risk_score > 0.6:
            recommendations.append("⚠️ HIGH RISK: Monitor positions closely")
            recommendations.append("📊 Review position concentration")
        elif risk_score > 0.4:
            recommendations.append("📈 MODERATE RISK: Consider taking profits on winners")
        else:
            recommendations.append("✅ LOW RISK: Portfolio within acceptable risk limits")
        
        # Specific recommendations based on metrics
        if abs(risk_report['risk_metrics'].get('current_drawdown', 0)) > 0.1:
            recommendations.append("📉 Consider reducing overall exposure due to drawdown")
        
        if risk_report['risk_metrics'].get('annualized_volatility', 0) > 0.25:
            recommendations.append("🔄 High volatility detected - consider rebalancing")
        
        return recommendations
    
    # Display methods for Rich UI
    def _show_position_sizing_header(self):
        """แสดงหัวข้อการคำนวณขนาดโพซิชั่น"""
        if RICH_AVAILABLE:
            header = Panel(
                "[bold cyan]💰 POSITION SIZING CALCULATION[/bold cyan]\n"
                "[yellow]Kelly Criterion with volatility adjustment[/yellow]",
                border_style="cyan"
            )
            self.console.print(header)
    
    def _show_portfolio_monitoring_header(self):
        """แสดงหัวข้อการตรวจสอบพอร์ตโฟลิโอ"""
        if RICH_AVAILABLE:
            header = Panel(
                "[bold yellow]📊 PORTFOLIO RISK MONITORING[/bold yellow]\n"
                "[yellow]Real-time risk assessment and alerts[/yellow]",
                border_style="yellow"
            )
            self.console.print(header)
    
    def _show_risk_controls_header(self):
        """แสดงหัวข้อการควบคุมความเสี่ยง"""
        if RICH_AVAILABLE:
            header = Panel(
                "[bold red]🛡️ AUTOMATED RISK CONTROLS[/bold red]\n"
                "[yellow]Stop loss, take profit, and position management[/yellow]",
                border_style="red"
            )
            self.console.print(header)
    
    def _display_position_sizing_results(self, position_info: Dict[str, Any]):
        """แสดงผลลัพธ์การคำนวณขนาดโพซิชั่น"""
        if RICH_AVAILABLE:
            table = Table(title="💰 Position Sizing Results", border_style="green")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="yellow")
            table.add_column("Risk Level", style="green")
            
            risk_score = position_info['risk_score']
            risk_level = "🟢 Low" if risk_score < 0.3 else "🟡 Medium" if risk_score < 0.7 else "🔴 High"
            
            table.add_row("Position Size", f"${position_info['dollar_amount']:,.2f}", "")
            table.add_row("Shares", f"{position_info['shares']:,}", "")
            table.add_row("Portfolio %", f"{position_info['position_fraction']:.2%}", "")
            table.add_row("Stop Loss", f"${position_info['stop_loss_price']:.2f}", "")
            table.add_row("Take Profit", f"${position_info['take_profit_price']:.2f}", "")
            table.add_row("Max Loss", f"${position_info['max_loss']:.2f}", "")
            table.add_row("Risk Score", f"{risk_score:.2f}", risk_level)
            
            self.console.print(table)
    
    def _display_risk_monitoring_results(self, risk_report: Dict[str, Any]):
        """แสดงผลลัพธ์การตรวจสอบความเสี่ยง"""
        if RICH_AVAILABLE:
            # Risk metrics table
            table = Table(title="📊 Risk Metrics", border_style="yellow")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="yellow")
            table.add_column("Status", style="green")
            
            metrics = risk_report['risk_metrics']
            overall_risk = risk_report['overall_risk_score']
            
            # Determine overall status
            if overall_risk < 0.3:
                status_color = "🟢"
                status_text = "Low Risk"
            elif overall_risk < 0.7:
                status_color = "🟡"
                status_text = "Medium Risk"
            else:
                status_color = "🔴"
                status_text = "High Risk"
            
            table.add_row("Portfolio Value", f"${risk_report['portfolio_value']:,.2f}", "")
            table.add_row("VaR (95%)", f"{metrics.get('var_95_pct', 0):.2%}", "")
            table.add_row("Current Drawdown", f"{metrics.get('current_drawdown', 0):.2%}", "")
            table.add_row("Volatility", f"{metrics.get('annualized_volatility', 0):.2%}", "")
            table.add_row("Overall Risk Score", f"{overall_risk:.2f}", f"{status_color} {status_text}")
            
            self.console.print(table)
            
            # Alerts
            if risk_report['alerts']:
                alert_panel = Panel(
                    "\n".join([f"• {alert['message']}" for alert in risk_report['alerts']]),
                    title="🚨 Risk Alerts",
                    border_style="red"
                )
                self.console.print(alert_panel)
            
            # Recommendations
            if risk_report['recommendations']:
                rec_panel = Panel(
                    "\n".join([f"• {rec}" for rec in risk_report['recommendations']]),
                    title="💡 Risk Management Recommendations",
                    border_style="blue"
                )
                self.console.print(rec_panel)
    
    def _display_risk_controls_results(self, control_actions: Dict[str, Any]):
        """แสดงผลลัพธ์การควบคุมความเสี่ยง"""
        if RICH_AVAILABLE:
            if control_actions['actions_taken']:
                table = Table(title="🛡️ Risk Control Actions", border_style="red")
                table.add_column("Action", style="cyan")
                table.add_column("Position", style="yellow")
                table.add_column("Details", style="green")
                
                for action in control_actions['actions_taken']:
                    details = f"${action.get('loss_amount', action.get('profit_amount', 0)):.2f}"
                    table.add_row(action['action'], action['position_id'], details)
                
                self.console.print(table)
            
            if control_actions['positions_adjusted']:
                adj_panel = Panel(
                    "\n".join([f"• {adj['position_id']}: {adj['adjustment']}" 
                              for adj in control_actions['positions_adjusted']]),
                    title="🔧 Position Adjustments",
                    border_style="yellow"
                )
                self.console.print(adj_panel)


if __name__ == "__main__":
    # Demo/Test the Risk Management System
    print("🚀 NICEGOLD ProjectP - Risk Management Demo")
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='1H')
    np.random.seed(42)
    
    market_data = pd.DataFrame({
        'close': 2000 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Sample portfolio data
    portfolio_data = {
        'total_value': 100000,
        'positions': {
            'GOLD_1': {'weight': 0.08, 'shares': 40, 'entry_price': 2000},
            'GOLD_2': {'weight': 0.05, 'shares': 25, 'entry_price': 1995}
        }
    }
    
    # Sample positions for risk control testing
    current_positions = {
        'GOLD_1': {
            'type': 'LONG',
            'shares': 40,
            'entry_price': 2000,
            'stop_loss_price': 1960,  # 2% stop loss
            'take_profit_price': 2080  # 4% take profit
        }
    }
    
    # Test the risk management system
    risk_mgr = RiskManagementSystem()
    
    # 1. Test position sizing
    print("\n1. Testing Position Sizing...")
    position_size = risk_mgr.calculate_position_size(
        signal_strength=0.75,
        account_balance=100000,
        current_price=2000,
        volatility=0.2
    )
    
    # 2. Test portfolio risk monitoring
    print("\n2. Testing Portfolio Risk Monitoring...")
    risk_report = risk_mgr.monitor_portfolio_risk(portfolio_data, market_data)
    
    # 3. Test risk controls
    print("\n3. Testing Risk Controls...")
    control_actions = risk_mgr.execute_risk_controls(current_positions, market_data)
    
    print(f"\n✅ Demo completed successfully!")
    print(f"💰 Position sizing calculated")
    print(f"📊 Risk score: {risk_report['overall_risk_score']:.2f}")
    print(f"🛡️ Control actions: {len(control_actions['actions_taken'])}")
