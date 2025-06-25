"""
Order Management System (OMS) - แยก logic OMS/risk management ออกจาก strategy.py
"""

class OMS:
    def __init__(self, initial_equity = 100, max_leverage = 100, max_drawdown_pct = 0.5, max_daily_loss_pct = 0.2, risk_per_trade_pct = 0.01):
        self.initial_equity = initial_equity
        self.equity = initial_equity
        self.balance = initial_equity
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.margin = 0.0
        self.max_leverage = max_leverage
        self.max_drawdown_pct = max_drawdown_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_equity = initial_equity
        self.daily_loss = 0.0
        self.last_equity = initial_equity
        self.trades_today = 0
        self.kill_switch = False
        self.log = []
    def update_equity(self, realized_pnl, unrealized_pnl = 0.0):
        self.realized_pnl += realized_pnl
        self.unrealized_pnl = unrealized_pnl
        self.equity = self.initial_equity + self.realized_pnl + self.unrealized_pnl
        self.max_equity = max(self.max_equity, self.equity)
        drawdown = (self.max_equity - self.equity) / self.max_equity
        if drawdown > self.max_drawdown_pct:
            self.kill_switch = True
        self.log.append({"equity": self.equity, "realized": self.realized_pnl, "unrealized": self.unrealized_pnl, "drawdown": drawdown, "kill_switch": self.kill_switch})
    def can_open_trade(self, trade_risk):
        if self.kill_switch:
            return False
        if trade_risk > self.equity * self.risk_per_trade_pct:
            return False
        return True
    def on_new_day(self):
        self.daily_loss = 0.0
        self.trades_today = 0
    def update_daily_loss(self, pnl):
        self.daily_loss += pnl
        if self.daily_loss < -self.initial_equity * self.max_daily_loss_pct:
            self.kill_switch = True
    def get_position_size(self, stop_loss_points, point_value = 10.0):
        risk_amount = self.equity * self.risk_per_trade_pct
        lot = risk_amount / (stop_loss_points * point_value)
        return max(lot, 0.01)