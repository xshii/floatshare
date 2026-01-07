"""持仓管理"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import date


@dataclass
class PositionDetail:
    """持仓明细"""

    code: str
    quantity: int
    avg_cost: float
    current_price: float = 0.0
    frozen: int = 0  # 冻结数量
    buy_date: Optional[date] = None

    @property
    def available(self) -> int:
        """可用数量"""
        return self.quantity - self.frozen

    @property
    def market_value(self) -> float:
        """市值"""
        return self.current_price * self.quantity

    @property
    def cost_value(self) -> float:
        """成本"""
        return self.avg_cost * self.quantity

    @property
    def profit(self) -> float:
        """浮动盈亏"""
        return self.market_value - self.cost_value

    @property
    def profit_pct(self) -> float:
        """浮动盈亏比例"""
        if self.cost_value <= 0:
            return 0.0
        return self.profit / self.cost_value


class PositionManager:
    """持仓管理器"""

    def __init__(self):
        self._positions: Dict[str, PositionDetail] = {}

    def get_position(self, code: str) -> Optional[PositionDetail]:
        """获取持仓"""
        return self._positions.get(code)

    def has_position(self, code: str) -> bool:
        """是否有持仓"""
        pos = self._positions.get(code)
        return pos is not None and pos.quantity > 0

    def get_available(self, code: str) -> int:
        """获取可用数量"""
        pos = self._positions.get(code)
        return pos.available if pos else 0

    def add_position(
        self,
        code: str,
        quantity: int,
        price: float,
        buy_date: Optional[date] = None,
    ) -> PositionDetail:
        """增加持仓"""
        if code in self._positions:
            pos = self._positions[code]
            total_cost = pos.avg_cost * pos.quantity + price * quantity
            pos.quantity += quantity
            pos.avg_cost = total_cost / pos.quantity if pos.quantity > 0 else 0
        else:
            pos = PositionDetail(
                code=code,
                quantity=quantity,
                avg_cost=price,
                current_price=price,
                buy_date=buy_date,
            )
            self._positions[code] = pos

        return pos

    def reduce_position(self, code: str, quantity: int) -> bool:
        """减少持仓"""
        pos = self._positions.get(code)
        if not pos or pos.available < quantity:
            return False

        pos.quantity -= quantity

        if pos.quantity <= 0:
            del self._positions[code]

        return True

    def freeze(self, code: str, quantity: int) -> bool:
        """冻结持仓"""
        pos = self._positions.get(code)
        if not pos or pos.available < quantity:
            return False

        pos.frozen += quantity
        return True

    def unfreeze(self, code: str, quantity: int) -> bool:
        """解冻持仓"""
        pos = self._positions.get(code)
        if not pos or pos.frozen < quantity:
            return False

        pos.frozen -= quantity
        return True

    def update_price(self, code: str, price: float) -> bool:
        """更新价格"""
        pos = self._positions.get(code)
        if pos:
            pos.current_price = price
            return True
        return False

    def update_prices(self, prices: Dict[str, float]) -> None:
        """批量更新价格"""
        for code, price in prices.items():
            self.update_price(code, price)

    def get_all_positions(self) -> List[PositionDetail]:
        """获取所有持仓"""
        return [p for p in self._positions.values() if p.quantity > 0]

    def get_total_value(self) -> float:
        """获取总市值"""
        return sum(p.market_value for p in self._positions.values())

    def get_total_profit(self) -> float:
        """获取总盈亏"""
        return sum(p.profit for p in self._positions.values())

    def get_position_summary(self) -> Dict:
        """获取持仓汇总"""
        positions = self.get_all_positions()
        return {
            "count": len(positions),
            "total_value": self.get_total_value(),
            "total_profit": self.get_total_profit(),
            "positions": [
                {
                    "code": p.code,
                    "quantity": p.quantity,
                    "avg_cost": p.avg_cost,
                    "current_price": p.current_price,
                    "market_value": p.market_value,
                    "profit": p.profit,
                    "profit_pct": f"{p.profit_pct:.2%}",
                }
                for p in positions
            ],
        }

    def clear(self) -> None:
        """清空持仓"""
        self._positions.clear()
