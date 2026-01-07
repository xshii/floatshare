"""组合管理"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import date, datetime


@dataclass
class Position:
    """持仓"""

    code: str
    quantity: int
    avg_cost: float
    current_price: float = 0.0
    frozen: int = 0

    @property
    def available(self) -> int:
        return self.quantity - self.frozen

    @property
    def market_value(self) -> float:
        return self.current_price * self.quantity

    @property
    def cost_value(self) -> float:
        return self.avg_cost * self.quantity

    @property
    def profit(self) -> float:
        return self.market_value - self.cost_value

    @property
    def profit_pct(self) -> float:
        if self.cost_value <= 0:
            return 0.0
        return self.profit / self.cost_value


@dataclass
class Portfolio:
    """投资组合"""

    name: str = "default"
    initial_capital: float = 1_000_000
    cash: float = field(default=0.0)
    positions: Dict[str, Position] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if self.cash == 0.0:
            self.cash = self.initial_capital

    @property
    def position_value(self) -> float:
        """持仓市值"""
        return sum(p.market_value for p in self.positions.values())

    @property
    def total_value(self) -> float:
        """总资产"""
        return self.cash + self.position_value

    @property
    def total_profit(self) -> float:
        """总盈亏"""
        return self.total_value - self.initial_capital

    @property
    def total_return(self) -> float:
        """总收益率"""
        if self.initial_capital <= 0:
            return 0.0
        return self.total_profit / self.initial_capital

    @property
    def position_ratio(self) -> float:
        """仓位比例"""
        if self.total_value <= 0:
            return 0.0
        return self.position_value / self.total_value

    def get_position(self, code: str) -> Optional[Position]:
        """获取持仓"""
        return self.positions.get(code)

    def has_position(self, code: str) -> bool:
        """是否有持仓"""
        pos = self.positions.get(code)
        return pos is not None and pos.quantity > 0

    def add_position(self, code: str, quantity: int, price: float) -> Position:
        """增加持仓"""
        if code in self.positions:
            pos = self.positions[code]
            total_cost = pos.avg_cost * pos.quantity + price * quantity
            pos.quantity += quantity
            pos.avg_cost = total_cost / pos.quantity if pos.quantity > 0 else 0
            pos.current_price = price
        else:
            pos = Position(
                code=code,
                quantity=quantity,
                avg_cost=price,
                current_price=price,
            )
            self.positions[code] = pos
        return pos

    def reduce_position(self, code: str, quantity: int) -> bool:
        """减少持仓"""
        pos = self.positions.get(code)
        if not pos or pos.quantity < quantity:
            return False

        pos.quantity -= quantity
        if pos.quantity <= 0:
            del self.positions[code]
        return True

    def update_price(self, code: str, price: float) -> bool:
        """更新价格"""
        pos = self.positions.get(code)
        if pos:
            pos.current_price = price
            return True
        return False

    def update_prices(self, prices: Dict[str, float]) -> None:
        """批量更新价格"""
        for code, price in prices.items():
            self.update_price(code, price)

    def get_weights(self) -> Dict[str, float]:
        """获取持仓权重"""
        total = self.total_value
        if total <= 0:
            return {}
        return {
            code: pos.market_value / total
            for code, pos in self.positions.items()
        }

    def summary(self) -> Dict:
        """获取组合摘要"""
        return {
            "name": self.name,
            "total_value": self.total_value,
            "cash": self.cash,
            "position_value": self.position_value,
            "position_count": len(self.positions),
            "position_ratio": f"{self.position_ratio:.2%}",
            "total_profit": self.total_profit,
            "total_return": f"{self.total_return:.2%}",
        }

    def reset(self) -> None:
        """重置组合"""
        self.cash = self.initial_capital
        self.positions.clear()
