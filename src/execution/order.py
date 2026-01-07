"""订单管理"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
import uuid


class OrderStatus(Enum):
    """订单状态"""

    PENDING = "pending"  # 待提交
    SUBMITTED = "submitted"  # 已提交
    PARTIAL = "partial"  # 部分成交
    FILLED = "filled"  # 全部成交
    CANCELLED = "cancelled"  # 已取消
    REJECTED = "rejected"  # 已拒绝


class OrderType(Enum):
    """订单类型"""

    MARKET = "market"  # 市价单
    LIMIT = "limit"  # 限价单
    STOP = "stop"  # 止损单
    STOP_LIMIT = "stop_limit"  # 止损限价单


@dataclass
class Order:
    """订单"""

    code: str  # 股票代码
    direction: str  # buy/sell
    quantity: int  # 数量
    price: Optional[float] = None  # 委托价格
    order_type: str = "limit"  # 订单类型

    # 状态相关
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None

    # 成交相关
    filled_quantity: int = 0
    filled_price: float = 0.0
    commission: float = 0.0

    # 其他
    reject_reason: Optional[str] = None
    stop_price: Optional[float] = None  # 止损触发价
    remarks: str = ""

    @property
    def is_buy(self) -> bool:
        return self.direction == "buy"

    @property
    def is_sell(self) -> bool:
        return self.direction == "sell"

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL)

    @property
    def unfilled_quantity(self) -> int:
        return self.quantity - self.filled_quantity

    @property
    def filled_amount(self) -> float:
        return self.filled_price * self.filled_quantity

    def cancel(self) -> bool:
        """取消订单"""
        if self.is_active:
            self.status = OrderStatus.CANCELLED
            self.updated_at = datetime.now()
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "order_id": self.order_id,
            "code": self.code,
            "direction": self.direction,
            "quantity": self.quantity,
            "price": self.price,
            "order_type": self.order_type,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "filled_price": self.filled_price,
            "commission": self.commission,
            "created_at": self.created_at.isoformat(),
        }

    def __repr__(self) -> str:
        return (
            f"Order({self.order_id}, {self.code}, {self.direction}, "
            f"{self.quantity}@{self.price}, {self.status.value})"
        )


class OrderManager:
    """订单管理器"""

    def __init__(self):
        self._orders: Dict[str, Order] = {}
        self._pending_orders: Dict[str, Order] = {}

    def create_order(
        self,
        code: str,
        direction: str,
        quantity: int,
        price: Optional[float] = None,
        order_type: str = "limit",
    ) -> Order:
        """创建订单"""
        order = Order(
            code=code,
            direction=direction,
            quantity=quantity,
            price=price,
            order_type=order_type,
        )
        self._orders[order.order_id] = order
        self._pending_orders[order.order_id] = order
        return order

    def get_order(self, order_id: str) -> Optional[Order]:
        """获取订单"""
        return self._orders.get(order_id)

    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        order = self._orders.get(order_id)
        if order:
            result = order.cancel()
            if result and order_id in self._pending_orders:
                del self._pending_orders[order_id]
            return result
        return False

    def get_pending_orders(self) -> list:
        """获取待处理订单"""
        return list(self._pending_orders.values())

    def get_orders_by_code(self, code: str) -> list:
        """按股票代码获取订单"""
        return [o for o in self._orders.values() if o.code == code]

    def update_order_status(
        self,
        order_id: str,
        status: OrderStatus,
        filled_quantity: int = 0,
        filled_price: float = 0.0,
    ) -> bool:
        """更新订单状态"""
        order = self._orders.get(order_id)
        if not order:
            return False

        order.status = status
        order.filled_quantity = filled_quantity
        order.filled_price = filled_price
        order.updated_at = datetime.now()

        if not order.is_active and order_id in self._pending_orders:
            del self._pending_orders[order_id]

        return True

    def clear_completed(self) -> int:
        """清理已完成订单"""
        completed = [
            oid for oid, o in self._orders.items()
            if not o.is_active
        ]
        for oid in completed:
            del self._orders[oid]
        return len(completed)
