"""模拟交易券商"""

from typing import List, Optional, Dict, Any
from datetime import datetime

from src.execution.broker.base import BaseBroker
from src.execution.order import Order, OrderStatus
from src.execution.position import PositionDetail, PositionManager
from config.trading import TradingConfig, Direction


class SimulatorBroker(BaseBroker):
    """模拟交易券商"""

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        config: Optional[TradingConfig] = None,
    ):
        self.initial_capital = initial_capital
        self.config = config or TradingConfig()

        self._connected = False
        self._cash = initial_capital
        self._position_manager = PositionManager()
        self._orders: Dict[str, Order] = {}
        self._quotes: Dict[str, Dict] = {}

    def connect(self) -> bool:
        """连接（模拟）"""
        self._connected = True
        return True

    def disconnect(self) -> None:
        """断开连接（模拟）"""
        self._connected = False

    def is_connected(self) -> bool:
        """是否已连接"""
        return self._connected

    def get_balance(self) -> Dict[str, float]:
        """获取资金余额"""
        position_value = self._position_manager.get_total_value()
        return {
            "cash": self._cash,
            "frozen": 0.0,
            "available": self._cash,
            "position_value": position_value,
            "total_assets": self._cash + position_value,
        }

    def get_positions(self) -> List[PositionDetail]:
        """获取持仓"""
        return self._position_manager.get_all_positions()

    def get_orders(self) -> List[Order]:
        """获取当日订单"""
        today = datetime.now().date()
        return [
            o for o in self._orders.values()
            if o.created_at.date() == today
        ]

    def submit_order(self, order: Order) -> bool:
        """提交订单（立即模拟成交）"""
        # 获取当前价格
        quote = self._quotes.get(order.code)
        if not quote:
            order.status = OrderStatus.REJECTED
            order.reject_reason = "无行情数据"
            return False

        current_price = quote.get("price", order.price)
        if current_price is None:
            order.status = OrderStatus.REJECTED
            order.reject_reason = "价格无效"
            return False

        # 计算成交价格（考虑滑点）
        if order.direction == "buy":
            fill_price = current_price * (1 + self.config.slippage)
        else:
            fill_price = current_price * (1 - self.config.slippage)

        # 计算费用
        amount = fill_price * order.quantity
        direction = Direction.BUY if order.is_buy else Direction.SELL
        commission = self.config.calculate_commission(amount, direction)

        # 检查资金/持仓
        if order.is_buy:
            total_cost = amount + commission
            if total_cost > self._cash:
                order.status = OrderStatus.REJECTED
                order.reject_reason = "资金不足"
                return False

            # 扣除资金
            self._cash -= total_cost

            # 增加持仓
            self._position_manager.add_position(
                code=order.code,
                quantity=order.quantity,
                price=fill_price,
            )

        else:  # sell
            if not self._position_manager.has_position(order.code):
                order.status = OrderStatus.REJECTED
                order.reject_reason = "无持仓"
                return False

            available = self._position_manager.get_available(order.code)
            if available < order.quantity:
                order.status = OrderStatus.REJECTED
                order.reject_reason = "持仓不足"
                return False

            # 减少持仓
            self._position_manager.reduce_position(order.code, order.quantity)

            # 增加资金
            self._cash += amount - commission

        # 更新订单状态
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = fill_price
        order.commission = commission
        order.updated_at = datetime.now()

        self._orders[order.order_id] = order
        return True

    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        order = self._orders.get(order_id)
        if order and order.is_active:
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now()
            return True
        return False

    def get_quote(self, code: str) -> Optional[Dict[str, Any]]:
        """获取行情"""
        return self._quotes.get(code)

    def set_quote(self, code: str, price: float, **kwargs) -> None:
        """设置行情（用于测试）"""
        self._quotes[code] = {
            "code": code,
            "price": price,
            "time": datetime.now(),
            **kwargs,
        }

        # 更新持仓价格
        self._position_manager.update_price(code, price)

    def set_quotes(self, quotes: Dict[str, float]) -> None:
        """批量设置行情"""
        for code, price in quotes.items():
            self.set_quote(code, price)

    def reset(self) -> None:
        """重置账户"""
        self._cash = self.initial_capital
        self._position_manager.clear()
        self._orders.clear()
        self._quotes.clear()
