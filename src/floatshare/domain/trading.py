"""A 股交易规则配置 — 纯值对象。"""

from __future__ import annotations

from dataclasses import dataclass

from floatshare.domain.enums import Direction


@dataclass(frozen=True, slots=True)
class TradingConfig:
    """A 股交易费率与限制。"""

    commission_rate: float = 0.0003  # 双边佣金率
    min_commission: float = 5.0  # 最低佣金
    stamp_duty: float = 0.0005  # 印花税（卖出单边，2023 减半后）
    transfer_fee: float = 0.00001  # 过户费（双边）
    slippage: float = 0.001  # 滑点

    max_position_pct: float = 0.25
    min_trade_lot: int = 100  # A 股最小交易单位（手）

    price_limit: float = 0.10
    price_limit_st: float = 0.05

    def calculate_fee(self, amount: float, direction: Direction) -> float:
        """单笔费用 = 佣金 + 过户费 + (卖出时) 印花税。"""
        commission = max(amount * self.commission_rate, self.min_commission)
        stamp = amount * self.stamp_duty if direction == Direction.SELL else 0.0
        transfer = amount * self.transfer_fee
        return commission + stamp + transfer
