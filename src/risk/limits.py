"""风控限制"""

from dataclasses import dataclass


@dataclass
class RiskLimits:
    """风控限制参数"""

    # 仓位限制
    max_position_pct: float = 0.20  # 单只股票最大持仓比例
    max_total_position_pct: float = 0.80  # 最大总仓位比例
    min_position_pct: float = 0.02  # 最小持仓比例（低于此值不建仓）

    # 订单限制
    max_order_pct: float = 0.10  # 单笔订单最大比例
    max_daily_trades: int = 50  # 日最大交易次数

    # 亏损限制
    max_daily_loss: float = 0.03  # 日最大亏损比例
    max_drawdown: float = 0.15  # 最大回撤比例
    stop_loss_pct: float = 0.05  # 默认止损比例

    # 盈利保护
    trailing_stop_pct: float = 0.10  # 移动止损比例
    take_profit_pct: float = 0.20  # 默认止盈比例

    # 行业限制
    max_industry_pct: float = 0.30  # 单一行业最大持仓比例

    # 交易限制
    no_trade_before_open: bool = True  # 开盘前不交易
    no_trade_after_close: bool = True  # 收盘后不交易
    no_trade_st: bool = True  # 不交易ST股票
    no_trade_new_stock_days: int = 5  # 新股上市N天内不交易

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "max_position_pct": self.max_position_pct,
            "max_total_position_pct": self.max_total_position_pct,
            "max_order_pct": self.max_order_pct,
            "max_daily_loss": self.max_daily_loss,
            "max_drawdown": self.max_drawdown,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "max_industry_pct": self.max_industry_pct,
        }

    @classmethod
    def conservative(cls) -> "RiskLimits":
        """保守型风控参数"""
        return cls(
            max_position_pct=0.10,
            max_total_position_pct=0.60,
            max_order_pct=0.05,
            max_daily_loss=0.02,
            max_drawdown=0.10,
            stop_loss_pct=0.03,
        )

    @classmethod
    def aggressive(cls) -> "RiskLimits":
        """激进型风控参数"""
        return cls(
            max_position_pct=0.30,
            max_total_position_pct=0.95,
            max_order_pct=0.15,
            max_daily_loss=0.05,
            max_drawdown=0.25,
            stop_loss_pct=0.08,
        )
