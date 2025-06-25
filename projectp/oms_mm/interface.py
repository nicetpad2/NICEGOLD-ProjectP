
# OMS/MM Interface สำหรับเชื่อมกับ pipeline ทุกโหมด (เทพ, risk - aware, multi - asset)
from .mm import Portfolio
from .oms import OMS, OrderType
from typing import Optional, Dict, Any, Callable, List
class OMSMMEngine:
    """Unified OMS/MM interface (เทพ: event/callback, plug - in, batch, logging, serialization, multi - asset, risk - aware)"""
    def __init__(self, initial_capital: float = 100.0, risk_config: Optional[Dict[str, Any]] = None, fee_model: Optional[Any] = None):
        self.oms = OMS()
        self.mm = Portfolio(initial_capital = initial_capital, risk_config = risk_config, fee_model = fee_model)
        # Event/callback hook
        self.on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None
        self.oms.on_fill = self._on_fill
        self.oms.on_cancel = self._on_cancel
        self.oms.on_reject = self._on_reject
        # ...future: plug - in, logging, monitoring, ...

    def _on_fill(self, event: Dict[str, Any]):
        if self.on_event:
            self.on_event('fill', event)

    def _on_cancel(self, event: Dict[str, Any]):
        if self.on_event:
            self.on_event('cancel', event)

    def _on_reject(self, event: Dict[str, Any]):
        if self.on_event:
            self.on_event('reject', event)

    def send_order(self, symbol: str, qty: float, order_type: str, price: float, user_id: Optional[str] = None, tag: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> Optional[str]:
        order_type_enum = OrderType[order_type.upper()] if order_type.upper() in OrderType.__members__ else OrderType.BUY
        if not self.mm.check_risk(symbol, qty, price, order_type_enum.value):
            return None
        order_id = self.oms.send_order(symbol, qty, order_type_enum, price, user_id, tag, meta)
        return order_id

    def fill_order(self, order_id: str, fill_qty: Optional[float] = None) -> bool:
        order = self.oms.get_order(order_id)
        if order:
            if not self.mm.check_risk(order.symbol, fill_qty or order.qty, order.price, order.order_type.value):
                return False
            self.oms.fill_order(order_id, fill_qty)
            self.mm.update_on_fill(order.symbol, fill_qty or order.qty, order.price, order.order_type.value)
            return True
        return False

    def cancel_order(self, order_id: str) -> bool:
        """ยกเลิกออเดอร์"""
        return self.oms.cancel_order(order_id)

    def get_open_orders(self):
        """ดูออเดอร์ที่ยังไม่ถูก fill/cancel"""
        return self.oms.get_open_orders()

    def get_order_history(self):
        """ประวัติออเดอร์ทั้งหมด"""
        return self.oms.get_order_history()

    def get_portfolio_stats(self) -> Dict[str, Any]:
        """สถิติพอร์ต (multi - asset, risk - aware)"""
        return self.mm.get_stats()

    def get_equity_curve(self) -> List[float]:
        """ดู equity curve"""
        return self.mm.equity_curve

    def get_position(self, symbol: str) -> float:
        """ดู position ของ symbol นั้น ๆ"""
        return self.mm.get_position(symbol)

    def get_trade_history(self):
        """ประวัติการเทรดทั้งหมด"""
        return self.mm.get_trade_history()

    def get_risk_status(self) -> Dict[str, Any]:
        """ดูสถานะ risk ล่าสุด"""
        return self.mm.get_risk_status()

    def serialize_state(self) -> Dict[str, Any]:
        """Export OMS/MM state (snapshot)"""
        return {
            'oms': self.oms.to_dict(), 
            'portfolio': self.mm.get_stats(), 
        }
    # ... จุดขยาย: plug - in, logging, monitoring, atomic transaction, ...

# ตัวอย่างการใช้งาน (import OMSMMEngine แล้วใช้ใน pipeline หรือ simulation loop)
# engine = OMSMMEngine(initial_capital = 100)
# order_id = engine.send_order('XAUUSD', 1, 'BUY', 2400)
# engine.fill_order(order_id)
# print(engine.get_portfolio_stats())