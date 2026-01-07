"""回测上下文"""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Optional

from ..strategy.base import StrategyContext
from ..account.portfolio import Position


@dataclass
class BacktestContext:
    """回测上下文"""

    initial_capital: float
    start_date: date
    end_date: date
    current_date: Optional[date] = None
    cash: float = field(default=0.0)
    positions: Dict[str, Position] = field(default_factory=dict)

    def __post_init__(self):
        if self.cash == 0.0:
            self.cash = self.initial_capital

    @property
    def portfolio_value(self) -> float:
        """组合总市值"""
        position_value = sum(p.market_value for p in self.positions.values())
        return self.cash + position_value

    @property
    def position_value(self) -> float:
        """持仓市值"""
        return sum(p.market_value for p in self.positions.values())

    @property
    def position_ratio(self) -> float:
        """仓位比例"""
        total = self.portfolio_value
        if total <= 0:
            return 0.0
        return self.position_value / total

    def get_position(self, code: str) -> Optional[Position]:
        """获取持仓"""
        return self.positions.get(code)

    def has_position(self, code: str) -> bool:
        """是否有持仓"""
        return code in self.positions and self.positions[code].quantity > 0

    def to_strategy_context(self) -> StrategyContext:
        """转换为策略上下文"""
        return StrategyContext(
            current_date=self.current_date,
            cash=self.cash,
            positions={code: pos.quantity for code, pos in self.positions.items()},
            portfolio_value=self.portfolio_value,
        )

    def reset(self) -> None:
        """重置上下文"""
        self.cash = self.initial_capital
        self.positions.clear()
        self.current_date = None
