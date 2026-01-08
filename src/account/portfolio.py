"""组合管理"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import date, datetime


class FlowType:
    """资金流水类型"""
    DEPOSIT = "deposit"          # 入金
    WITHDRAW = "withdraw"        # 出金
    DIVIDEND = "dividend"        # 现金分红
    TAX = "tax"                  # 扣税（红利税等）
    INTEREST = "interest"        # 利息收入
    COMMISSION = "commission"    # 手续费
    TRANSFER_FEE = "transfer"    # 过户费
    STAMP_TAX = "stamp"          # 印花税


@dataclass
class CashFlow:
    """资金流水"""

    date: datetime
    amount: float  # 正数收入，负数支出
    flow_type: str  # 流水类型，见 FlowType
    code: str = ""  # 关联股票代码（分红、扣税时使用）
    note: str = ""


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
    cash_flows: List[CashFlow] = field(default_factory=list)
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
        self.cash_flows.clear()

    # ============================================================
    # 资金流水
    # ============================================================

    def deposit(self, amount: float, note: str = "") -> CashFlow:
        """
        入金

        Args:
            amount: 入金金额（必须为正数）
            note: 备注

        Returns:
            资金流水记录
        """
        if amount <= 0:
            raise ValueError("入金金额必须为正数")

        flow = CashFlow(
            date=datetime.now(),
            amount=amount,
            flow_type=FlowType.DEPOSIT,
            note=note,
        )
        self.cash += amount
        self.cash_flows.append(flow)
        return flow

    def withdraw(self, amount: float, note: str = "") -> CashFlow:
        """
        出金

        Args:
            amount: 出金金额（必须为正数）
            note: 备注

        Returns:
            资金流水记录

        Raises:
            ValueError: 金额无效或余额不足
        """
        if amount <= 0:
            raise ValueError("出金金额必须为正数")
        if amount > self.cash:
            raise ValueError(f"余额不足，当前可用: {self.cash:.2f}")

        flow = CashFlow(
            date=datetime.now(),
            amount=-amount,  # 负数表示出金
            flow_type=FlowType.WITHDRAW,
            note=note,
        )
        self.cash -= amount
        self.cash_flows.append(flow)
        return flow

    def receive_dividend(
        self,
        code: str,
        amount: float,
        tax: float = 0.0,
        note: str = "",
    ) -> List[CashFlow]:
        """
        收到分红

        Args:
            code: 股票代码
            amount: 分红金额（税前）
            tax: 红利税
            note: 备注

        Returns:
            资金流水记录列表（分红+扣税）
        """
        if amount <= 0:
            raise ValueError("分红金额必须为正数")

        flows = []

        # 分红收入
        dividend_flow = CashFlow(
            date=datetime.now(),
            amount=amount,
            flow_type=FlowType.DIVIDEND,
            code=code,
            note=note or f"{code} 分红",
        )
        flows.append(dividend_flow)
        self.cash += amount

        # 扣税
        if tax > 0:
            tax_flow = CashFlow(
                date=datetime.now(),
                amount=-tax,
                flow_type=FlowType.TAX,
                code=code,
                note=f"{code} 红利税",
            )
            flows.append(tax_flow)
            self.cash -= tax

        self.cash_flows.extend(flows)
        return flows

    def deduct_tax(self, code: str, amount: float, note: str = "") -> CashFlow:
        """
        扣税（补扣红利税等）

        Args:
            code: 关联股票代码
            amount: 扣税金额（正数）
            note: 备注

        Returns:
            资金流水记录
        """
        if amount <= 0:
            raise ValueError("扣税金额必须为正数")

        flow = CashFlow(
            date=datetime.now(),
            amount=-amount,
            flow_type=FlowType.TAX,
            code=code,
            note=note or f"{code} 扣税",
        )
        self.cash -= amount
        self.cash_flows.append(flow)
        return flow

    def add_fee(
        self,
        amount: float,
        flow_type: str = FlowType.COMMISSION,
        code: str = "",
        note: str = "",
    ) -> CashFlow:
        """
        扣除费用（手续费、过户费、印花税等）

        Args:
            amount: 费用金额（正数）
            flow_type: 费用类型
            code: 关联股票代码
            note: 备注

        Returns:
            资金流水记录
        """
        if amount <= 0:
            raise ValueError("费用金额必须为正数")

        flow = CashFlow(
            date=datetime.now(),
            amount=-amount,
            flow_type=flow_type,
            code=code,
            note=note,
        )
        self.cash -= amount
        self.cash_flows.append(flow)
        return flow

    def add_interest(self, amount: float, note: str = "") -> CashFlow:
        """
        利息收入

        Args:
            amount: 利息金额
            note: 备注

        Returns:
            资金流水记录
        """
        if amount <= 0:
            raise ValueError("利息金额必须为正数")

        flow = CashFlow(
            date=datetime.now(),
            amount=amount,
            flow_type=FlowType.INTEREST,
            note=note or "利息收入",
        )
        self.cash += amount
        self.cash_flows.append(flow)
        return flow

    # ============================================================
    # 资金流水统计
    # ============================================================

    @property
    def total_deposits(self) -> float:
        """总入金"""
        return sum(f.amount for f in self.cash_flows if f.flow_type == FlowType.DEPOSIT)

    @property
    def total_withdrawals(self) -> float:
        """总出金（返回正数）"""
        return abs(sum(f.amount for f in self.cash_flows if f.flow_type == FlowType.WITHDRAW))

    @property
    def total_dividends(self) -> float:
        """总分红收入"""
        return sum(f.amount for f in self.cash_flows if f.flow_type == FlowType.DIVIDEND)

    @property
    def total_taxes(self) -> float:
        """总扣税（返回正数）"""
        return abs(sum(f.amount for f in self.cash_flows if f.flow_type == FlowType.TAX))

    @property
    def total_fees(self) -> float:
        """总费用（手续费+过户费+印花税，返回正数）"""
        fee_types = {FlowType.COMMISSION, FlowType.TRANSFER_FEE, FlowType.STAMP_TAX}
        return abs(sum(f.amount for f in self.cash_flows if f.flow_type in fee_types))

    @property
    def net_cash_flow(self) -> float:
        """净资金流入（入金-出金）"""
        return self.total_deposits - self.total_withdrawals

    def get_cash_flow_summary(self) -> Dict:
        """获取资金流水汇总"""
        return {
            "total_deposits": self.total_deposits,
            "total_withdrawals": self.total_withdrawals,
            "net_cash_flow": self.net_cash_flow,
            "total_dividends": self.total_dividends,
            "total_taxes": self.total_taxes,
            "total_fees": self.total_fees,
            "flow_count": len(self.cash_flows),
        }
