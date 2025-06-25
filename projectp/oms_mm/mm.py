
# MM (Money Management) - เทพ
from typing import List, Dict, Optional, Any
class Portfolio:
    """Portfolio Management (multi - asset, multi - currency, risk, plug - in, robust, usable)"""
    def __init__(self, initial_capital: float = 100.0, risk_config: Optional[Dict[str, Any]] = None, currency: str = 'USD', fee_model: Optional[Any] = None):
        self.initial_capital: float = initial_capital
        self.cash: float = initial_capital
        self.currency: str = currency
        self.positions: Dict[str, float] = {}  # symbol -> qty
        self.position_cost: Dict[str, float] = {}  # symbol -> cost basis
        self.trade_history: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = [initial_capital]
        self.risk_config: Dict[str, Any] = risk_config or {}
        self.risk_violations: List[str] = []
        self.fee_model = fee_model  # plug - in: function(symbol, qty, price, side) -> fee
        self.unrealized_pnl: float = 0.0
        self.realized_pnl: float = 0.0
        self.snapshots: List[Dict[str, Any]] = []  # portfolio snapshot/history
        # ...future: multi - account, plug - in, logging...

    def check_risk(self, symbol: str, qty: float, price: float, side: str) -> bool:
        """ตรวจสอบ risk ก่อนเทรด (max position, max loss, max drawdown)"""
        # Max position per symbol
        max_pos = self.risk_config.get('max_position', {}).get(symbol, None)
        new_pos = self.positions.get(symbol, 0) + (qty if side == 'BUY' else -qty)
        if max_pos is not None and abs(new_pos) > max_pos:
            self.risk_violations.append(f"Max position exceeded for {symbol}: {new_pos} > {max_pos}")
            return False
        # Max loss (absolute loss from initial capital)
        max_loss = self.risk_config.get('max_loss', None)
        if max_loss is not None:
            equity = self.get_equity() - (qty * price if side == 'BUY' else 0)
            if equity < self.initial_capital * (1 - max_loss):
                self.risk_violations.append(f"Max loss exceeded: equity {equity} < {self.initial_capital * (1 - max_loss)}")
                return False
        # Max drawdown
        max_dd = self.risk_config.get('max_drawdown', None)
        if max_dd is not None:
            dd = self.get_drawdown()
            if dd > max_dd:
                self.risk_violations.append(f"Max drawdown exceeded: {dd} > {max_dd}")
                return False
        return True

    def update_on_fill(self, symbol: str, qty: float, price: float, side: str) -> bool:
        if not self.check_risk(symbol, qty, price, side):
            return False
        cost = qty * price
        fee = self.fee_model(symbol, qty, price, side) if self.fee_model else 0.0
        if side == 'BUY':
            self.cash -= (cost + fee)
            self.positions[symbol] = self.positions.get(symbol, 0) + qty
            self.position_cost[symbol] = price  # simple avg cost, can be improved
        elif side == 'SELL':
            self.cash += (cost - fee)
            self.positions[symbol] = self.positions.get(symbol, 0) - qty
            self.realized_pnl += (price - self.position_cost.get(symbol, price)) * qty - fee
        self.trade_history.append({'symbol': symbol, 'qty': qty, 'price': price, 'side': side, 'fee': fee})
        self.equity_curve.append(self.get_equity())
        self.snapshots.append(self.get_stats())
        return True

    def get_unrealized_pnl(self, price_map: Optional[Dict[str, float]] = None) -> float:
        # price_map: symbol -> last price
        if not price_map:
            return 0.0
        pnl = 0.0
        for s, qty in self.positions.items():
            last_price = price_map.get(s, self.position_cost.get(s, 0))
            pnl += (last_price - self.position_cost.get(s, 0)) * qty
        self.unrealized_pnl = pnl
        return pnl

    def get_equity(self) -> float:
        """คำนวณ equity ปัจจุบัน (multi - asset)"""
        return self.cash + sum(self.positions.get(s, 0) * self.position_cost.get(s, 0) for s in self.positions)

    def get_drawdown(self) -> float:
        """คำนวณ drawdown ปัจจุบัน"""
        peak = max(self.equity_curve)
        trough = min(self.equity_curve)
        return (peak - trough) / peak if peak > 0 else 0.0

    def get_position(self, symbol: str) -> float:
        """ดู position ของ symbol นั้น ๆ"""
        return self.positions.get(symbol, 0)

    def get_trade_history(self) -> List[Dict[str, Any]]:
        """ประวัติการเทรดทั้งหมด"""
        return self.trade_history

    def get_risk_status(self) -> Dict[str, Any]:
        """ดูสถานะ risk ล่าสุด"""
        return {
            'risk_violations': self.risk_violations[ - 10:], 
            'current_drawdown': self.get_drawdown(), 
            'positions': self.positions.copy(), 
            'equity': self.get_equity(), 
        }

    def get_stats(self) -> Dict[str, Any]:
        """สรุปสถิติพอร์ต (multi - asset, risk - aware)"""
        stats = {
            'initial_capital': self.initial_capital, 
            'cash': self.cash, 
            'currency': self.currency, 
            'equity': self.get_equity(), 
            'drawdown': self.get_drawdown(), 
            'positions': self.positions.copy(), 
            'trades': len(self.trade_history), 
            'risk_violations': len(self.risk_violations), 
            'realized_pnl': self.realized_pnl, 
            'unrealized_pnl': self.unrealized_pnl, 
        }
        return stats
    # ... จุดขยาย: custom risk model, plug - in, multi - account, logging, ...