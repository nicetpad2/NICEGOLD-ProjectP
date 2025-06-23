# OMS (Order Management System) - เทพ
import uuid
from enum import Enum
from typing import List, Dict, Optional, Any, Callable, Tuple
from datetime import datetime, timezone

class OrderStatus(Enum):
    """สถานะออเดอร์"""
    NEW = 'NEW'
    FILLED = 'FILLED'
    CANCELLED = 'CANCELLED'
    REJECTED = 'REJECTED'
    PARTIALLY_FILLED = 'PARTIALLY_FILLED'

class OrderType(Enum):
    """ประเภทออเดอร์ (ขยายรองรับ MARKET, LIMIT, STOP, OCO, ฯลฯ)"""
    BUY = 'BUY'
    SELL = 'SELL'
    MARKET = 'MARKET'
    LIMIT = 'LIMIT'
    STOP = 'STOP'
    OCO = 'OCO'  # One Cancels Other (future)
    # ...ขยายเพิ่มได้...

class Order:
    """ออเดอร์เดียว (multi-asset, multi-type, metadata, event/callback ready)"""
    def __init__(self, symbol: str, qty: float, order_type: OrderType, price: float, user_id: Optional[str] = None, tag: Optional[str] = None, meta: Optional[Dict[str, Any]] = None):
        self.id: str = str(uuid.uuid4())
        self.symbol: str = symbol
        self.qty: float = qty
        self.order_type: OrderType = order_type
        self.price: float = price
        self.status: OrderStatus = OrderStatus.NEW
        self.filled_qty: float = 0.0
        self.timestamp: datetime = datetime.now(timezone.utc)
        self.history: List[Dict[str, Any]] = []
        self.user_id: Optional[str] = user_id
        self.tag: Optional[str] = tag
        self.meta: Dict[str, Any] = meta or {}
        self.reject_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'qty': self.qty,
            'order_type': self.order_type.value,
            'price': self.price,
            'status': self.status.value,
            'filled_qty': self.filled_qty,
            'timestamp': self.timestamp.isoformat(),
            'history': self.history,
        }

class OMS:
    """Order Management System (เทพ: multi-asset, multi-type, event/callback, validation, plug-in)"""
    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.on_fill: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_cancel: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_reject: Optional[Callable[[Dict[str, Any]], None]] = None
        self.custom_validator: Optional[Callable[[Order], Tuple[bool, str]]] = None
        # ...future: multi-user/account, plug-in, logging...

    def send_order(self, symbol: str, qty: float, order_type: OrderType, price: float, user_id: Optional[str] = None, tag: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> str:
        if qty <= 0 or price <= 0:
            raise ValueError("qty and price must be positive")
        order = Order(symbol, qty, order_type, price, user_id, tag, meta)
        # Custom validation hook
        if self.custom_validator:
            valid, reason = self.custom_validator(order)
            if not valid:
                order.status = OrderStatus.REJECTED
                order.reject_reason = reason
                self.order_history.append(order)
                if self.on_reject:
                    self.on_reject({'order': order, 'reason': reason})
                return order.id
        self.orders[order.id] = order
        self.order_history.append(order)
        return order.id

    def cancel_order(self, order_id: str) -> bool:
        order = self.orders.get(order_id)
        if order and order.status == OrderStatus.NEW:
            order.status = OrderStatus.CANCELLED
            order.history.append({'event': 'cancel', 'timestamp': datetime.now(timezone.utc).isoformat()})
            if self.on_cancel:
                self.on_cancel({'order': order})
            return True
        return False

    def fill_order(self, order_id: str, fill_qty: Optional[float] = None) -> bool:
        order = self.orders.get(order_id)
        if order and order.status == OrderStatus.NEW:
            qty_to_fill = fill_qty if fill_qty is not None else order.qty
            if qty_to_fill > order.qty:
                qty_to_fill = order.qty
            order.filled_qty = qty_to_fill
            order.status = OrderStatus.FILLED if qty_to_fill == order.qty else OrderStatus.PARTIALLY_FILLED
            order.history.append({'event': 'fill', 'qty': qty_to_fill, 'timestamp': datetime.now(timezone.utc).isoformat()})
            if self.on_fill:
                self.on_fill({'order': order, 'qty': qty_to_fill})
            return True
        return False

    def get_order(self, order_id: str) -> Optional[Order]:
        return self.orders.get(order_id)

    def get_open_orders(self) -> List[Order]:
        return [o for o in self.orders.values() if o.status == OrderStatus.NEW]

    def get_order_history(self) -> List[Order]:
        return self.order_history

    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        return [o for o in self.order_history if o.symbol == symbol]

    def to_dict(self) -> List[Dict[str, Any]]:
        return [o.to_dict() for o in self.order_history]
    # จุดขยาย: add batch_order, plug-in, logging, multi-account, ...
