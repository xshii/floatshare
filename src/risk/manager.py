"""风险管理器"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from src.risk.limits import RiskLimits
from src.risk.exposure import ExposureCalculator
from src.execution.order import Order
from src.execution.position import PositionDetail


@dataclass
class RiskCheckResult:
    """风险检查结果"""

    passed: bool
    violations: List[str]
    warnings: List[str]


class RiskManager:
    """风险管理器"""

    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self.exposure_calc = ExposureCalculator()
        self._daily_loss: float = 0.0
        self._initial_value: float = 0.0

    def set_initial_value(self, value: float) -> None:
        """设置初始市值"""
        self._initial_value = value

    def update_daily_loss(self, current_value: float) -> float:
        """更新当日亏损"""
        if self._initial_value > 0:
            self._daily_loss = (self._initial_value - current_value) / self._initial_value
        return self._daily_loss

    def check_order(
        self,
        order: Order,
        positions: Dict[str, PositionDetail],
        cash: float,
        portfolio_value: float,
    ) -> RiskCheckResult:
        """
        订单风控检查

        Args:
            order: 待检查订单
            positions: 当前持仓
            cash: 可用资金
            portfolio_value: 组合总市值
        """
        violations = []
        warnings = []

        # 1. 单笔订单金额检查
        order_amount = order.price * order.quantity if order.price else 0
        max_order = portfolio_value * self.limits.max_order_pct

        if order_amount > max_order:
            violations.append(
                f"订单金额 {order_amount:.2f} 超过限制 {max_order:.2f}"
            )

        # 2. 单只股票持仓检查
        if order.is_buy:
            current_position = positions.get(order.code)
            current_value = current_position.market_value if current_position else 0
            new_value = current_value + order_amount

            max_position = portfolio_value * self.limits.max_position_pct
            if new_value > max_position:
                violations.append(
                    f"持仓市值 {new_value:.2f} 将超过限制 {max_position:.2f}"
                )

        # 3. 总仓位检查
        if order.is_buy:
            total_position = sum(p.market_value for p in positions.values())
            new_total = total_position + order_amount

            max_total = portfolio_value * self.limits.max_total_position_pct
            if new_total > max_total:
                warnings.append(
                    f"总仓位 {new_total:.2f} 将超过建议上限 {max_total:.2f}"
                )

        # 4. 资金检查
        if order.is_buy and order_amount > cash:
            violations.append(
                f"资金不足: 需要 {order_amount:.2f}, 可用 {cash:.2f}"
            )

        # 5. 日内亏损检查
        if self._daily_loss >= self.limits.max_daily_loss:
            violations.append(
                f"已触发日内止损: 亏损 {self._daily_loss:.2%}"
            )

        return RiskCheckResult(
            passed=len(violations) == 0,
            violations=violations,
            warnings=warnings,
        )

    def check_portfolio(
        self,
        positions: Dict[str, PositionDetail],
        portfolio_value: float,
    ) -> RiskCheckResult:
        """组合风控检查"""
        violations = []
        warnings = []

        # 1. 集中度检查
        for code, pos in positions.items():
            concentration = pos.market_value / portfolio_value if portfolio_value > 0 else 0

            if concentration > self.limits.max_position_pct:
                warnings.append(
                    f"{code} 持仓占比 {concentration:.2%} 超过限制"
                )

        # 2. 行业集中度检查（如果有行业信息）
        # TODO: 实现行业集中度检查

        # 3. 总仓位检查
        total_position = sum(p.market_value for p in positions.values())
        position_ratio = total_position / portfolio_value if portfolio_value > 0 else 0

        if position_ratio > self.limits.max_total_position_pct:
            warnings.append(
                f"总仓位 {position_ratio:.2%} 超过建议上限"
            )

        return RiskCheckResult(
            passed=len(violations) == 0,
            violations=violations,
            warnings=warnings,
        )

    def calculate_position_size(
        self,
        code: str,
        price: float,
        portfolio_value: float,
        risk_per_trade: float = 0.02,
        stop_loss_pct: float = 0.05,
    ) -> int:
        """
        计算建议持仓数量

        Args:
            code: 股票代码
            price: 当前价格
            portfolio_value: 组合市值
            risk_per_trade: 单笔交易风险（占组合比例）
            stop_loss_pct: 止损比例
        """
        # 基于风险的仓位计算
        risk_amount = portfolio_value * risk_per_trade
        position_value = risk_amount / stop_loss_pct

        # 不超过单只股票最大持仓
        max_position_value = portfolio_value * self.limits.max_position_pct
        position_value = min(position_value, max_position_value)

        # 计算股数（整手）
        quantity = int(position_value / price)
        quantity = (quantity // 100) * 100

        return quantity

    def should_stop_loss(
        self,
        position: PositionDetail,
        stop_loss_pct: float = 0.05,
    ) -> bool:
        """判断是否应该止损"""
        return position.profit_pct <= -stop_loss_pct

    def should_take_profit(
        self,
        position: PositionDetail,
        take_profit_pct: float = 0.20,
    ) -> bool:
        """判断是否应该止盈"""
        return position.profit_pct >= take_profit_pct

    def reset_daily(self) -> None:
        """重置日内数据"""
        self._daily_loss = 0.0
