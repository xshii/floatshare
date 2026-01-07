"""交易相关配置"""

from dataclasses import dataclass
from enum import Enum


class Market(Enum):
    """市场类型"""

    SH = "SH"  # 上海
    SZ = "SZ"  # 深圳
    BJ = "BJ"  # 北京


class OrderType(Enum):
    """订单类型"""

    MARKET = "market"  # 市价单
    LIMIT = "limit"  # 限价单


class Direction(Enum):
    """交易方向"""

    BUY = "buy"
    SELL = "sell"


@dataclass
class TradingConfig:
    """交易配置类"""

    # 佣金设置
    commission_rate: float = 0.0003  # 佣金率
    min_commission: float = 5.0  # 最低佣金

    # 印花税（卖出时收取）
    stamp_duty: float = 0.001

    # 过户费
    transfer_fee: float = 0.00001

    # 滑点设置
    slippage: float = 0.001

    # 交易限制
    max_position_pct: float = 0.25  # 单只股票最大持仓比例
    min_trade_amount: int = 100  # 最小交易数量（手）

    # 涨跌停限制
    price_limit: float = 0.10  # 普通股票涨跌停10%
    price_limit_st: float = 0.05  # ST股票涨跌停5%

    def calculate_commission(self, amount: float, direction: Direction) -> float:
        """计算交易费用"""
        # 佣金
        commission = max(amount * self.commission_rate, self.min_commission)

        # 印花税（卖出时收取）
        stamp = amount * self.stamp_duty if direction == Direction.SELL else 0

        # 过户费
        transfer = amount * self.transfer_fee

        return commission + stamp + transfer
