"""风险限制参数 — 不可变值对象。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RiskLimits:
    """风控限制参数集合。"""

    # 仓位限制
    max_position_pct: float = 0.20
    max_total_position_pct: float = 0.80
    min_position_pct: float = 0.02

    # 订单限制
    max_order_pct: float = 0.10
    max_daily_trades: int = 50

    # 亏损限制
    max_daily_loss: float = 0.03
    max_drawdown: float = 0.15
    stop_loss_pct: float = 0.05

    # 盈利保护
    trailing_stop_pct: float = 0.10
    take_profit_pct: float = 0.20

    # 行业 / 标的限制
    max_industry_pct: float = 0.30
    no_trade_st: bool = True
    no_trade_new_stock_days: int = 5

    @classmethod
    def conservative(cls) -> RiskLimits:
        return cls(
            max_position_pct=0.10,
            max_total_position_pct=0.60,
            max_order_pct=0.05,
            max_daily_loss=0.02,
            max_drawdown=0.10,
            stop_loss_pct=0.03,
        )

    @classmethod
    def aggressive(cls) -> RiskLimits:
        return cls(
            max_position_pct=0.30,
            max_total_position_pct=0.95,
            max_order_pct=0.15,
            max_daily_loss=0.05,
            max_drawdown=0.25,
            stop_loss_pct=0.08,
        )
