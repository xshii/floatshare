"""券商接口基类"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from ..order import Order
from ..position import PositionDetail


class BaseBroker(ABC):
    """券商接口基类"""

    @abstractmethod
    def connect(self) -> bool:
        """连接券商"""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """断开连接"""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """是否已连接"""
        pass

    @abstractmethod
    def get_balance(self) -> Dict[str, float]:
        """获取资金余额"""
        pass

    @abstractmethod
    def get_positions(self) -> List[PositionDetail]:
        """获取持仓"""
        pass

    @abstractmethod
    def get_orders(self) -> List[Order]:
        """获取当日订单"""
        pass

    @abstractmethod
    def submit_order(self, order: Order) -> bool:
        """提交订单"""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        pass

    @abstractmethod
    def get_quote(self, code: str) -> Optional[Dict[str, Any]]:
        """获取行情"""
        pass

    def buy(
        self,
        code: str,
        quantity: int,
        price: Optional[float] = None,
    ) -> Optional[Order]:
        """买入"""
        order = Order(
            code=code,
            direction="buy",
            quantity=quantity,
            price=price,
            order_type="limit" if price else "market",
        )

        if self.submit_order(order):
            return order
        return None

    def sell(
        self,
        code: str,
        quantity: int,
        price: Optional[float] = None,
    ) -> Optional[Order]:
        """卖出"""
        order = Order(
            code=code,
            direction="sell",
            quantity=quantity,
            price=price,
            order_type="limit" if price else "market",
        )

        if self.submit_order(order):
            return order
        return None
