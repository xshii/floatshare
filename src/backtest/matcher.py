"""撮合引擎"""

from typing import Optional
import pandas as pd

from src.execution.order import Order, OrderStatus
from config import TradingConfig, Direction


class OrderMatcher:
    """订单撮合器"""

    def __init__(self, config: Optional[TradingConfig] = None):
        self.config = config or TradingConfig()

    def match(self, order: Order, market_data: pd.DataFrame) -> Order:
        """
        撮合订单

        Args:
            order: 订单
            market_data: 当日市场数据

        Returns:
            更新后的订单
        """
        # 获取标的数据
        stock_data = market_data[market_data["code"] == order.code]

        if stock_data.empty:
            order.status = OrderStatus.REJECTED
            order.reject_reason = "无市场数据"
            return order

        row = stock_data.iloc[0]

        # 检查涨跌停
        if not self._check_price_limit(order, row):
            order.status = OrderStatus.REJECTED
            order.reject_reason = "涨跌停限制"
            return order

        # 检查成交量
        if not self._check_volume(order, row):
            order.status = OrderStatus.REJECTED
            order.reject_reason = "成交量不足"
            return order

        # 计算成交价格（考虑滑点）
        filled_price = self._calculate_fill_price(order, row)

        # 检查价格是否在当日范围内
        if not self._check_price_range(filled_price, row):
            order.status = OrderStatus.REJECTED
            order.reject_reason = "价格超出范围"
            return order

        # 成交
        order.filled_price = filled_price
        order.filled_quantity = order.quantity
        order.status = OrderStatus.FILLED

        # 计算交易费用
        order.commission = self._calculate_commission(order)

        return order

    def _check_price_limit(self, order: Order, row: pd.Series) -> bool:
        """检查涨跌停"""
        # 简化处理：如果收盘价等于最高价且方向是买入，可能涨停
        # 如果收盘价等于最低价且方向是卖出，可能跌停
        if order.direction == "buy" and row["close"] >= row["high"] * 0.999:
            # 可能涨停，检查是否封板
            if row["volume"] < row.get("avg_volume", row["volume"]) * 0.1:
                return False

        if order.direction == "sell" and row["close"] <= row["low"] * 1.001:
            if row["volume"] < row.get("avg_volume", row["volume"]) * 0.1:
                return False

        return True

    def _check_volume(self, order: Order, row: pd.Series) -> bool:
        """检查成交量是否足够"""
        # 订单数量不能超过当日成交量的10%
        max_volume = row["volume"] * 0.1
        return order.quantity <= max_volume

    def _calculate_fill_price(self, order: Order, row: pd.Series) -> float:
        """计算成交价格"""
        base_price = order.price or row["close"]

        # 应用滑点
        if order.direction == "buy":
            # 买入时价格上滑
            slippage_price = base_price * (1 + self.config.slippage)
            # 不能超过最高价
            return min(slippage_price, row["high"])
        else:
            # 卖出时价格下滑
            slippage_price = base_price * (1 - self.config.slippage)
            # 不能低于最低价
            return max(slippage_price, row["low"])

    def _check_price_range(self, price: float, row: pd.Series) -> bool:
        """检查价格是否在当日范围内"""
        return row["low"] <= price <= row["high"]

    def _calculate_commission(self, order: Order) -> float:
        """计算交易费用"""
        amount = order.filled_price * order.filled_quantity
        direction = Direction.BUY if order.direction == "buy" else Direction.SELL
        return self.config.calculate_commission(amount, direction)
