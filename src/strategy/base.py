"""策略基类"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import date
import pandas as pd


@dataclass
class Signal:
    """交易信号"""

    code: str  # 股票代码
    direction: str  # buy/sell
    strength: float = 1.0  # 信号强度 0-1
    price: Optional[float] = None  # 建议价格
    quantity: Optional[int] = None  # 建议数量
    reason: str = ""  # 信号原因


@dataclass
class StrategyContext:
    """策略上下文"""

    current_date: date
    cash: float
    positions: Dict[str, int] = field(default_factory=dict)  # code -> quantity
    portfolio_value: float = 0.0
    history: pd.DataFrame = field(default_factory=pd.DataFrame)


class Strategy(ABC):
    """策略基类"""

    name: str = "BaseStrategy"
    description: str = ""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        初始化策略

        Args:
            params: 策略参数
        """
        self.params = params or {}
        self._signals: List[Signal] = []

    @abstractmethod
    def init(self, context: StrategyContext) -> None:
        """
        策略初始化

        在回测开始前调用，用于初始化指标、变量等
        """
        pass

    @abstractmethod
    def handle_data(
        self, context: StrategyContext, data: pd.DataFrame
    ) -> List[Signal]:
        """
        处理每日数据

        Args:
            context: 策略上下文
            data: 当日市场数据

        Returns:
            交易信号列表
        """
        pass

    def before_trading(self, context: StrategyContext) -> None:
        """盘前处理"""
        pass

    def after_trading(self, context: StrategyContext) -> None:
        """盘后处理"""
        pass

    def on_order_filled(
        self, context: StrategyContext, order: Dict[str, Any]
    ) -> None:
        """订单成交回调"""
        pass

    def on_order_rejected(
        self, context: StrategyContext, order: Dict[str, Any], reason: str
    ) -> None:
        """订单拒绝回调"""
        pass

    def get_param(self, key: str, default: Any = None) -> Any:
        """获取策略参数"""
        return self.params.get(key, default)

    def set_param(self, key: str, value: Any) -> None:
        """设置策略参数"""
        self.params[key] = value

    def log(self, message: str) -> None:
        """记录日志"""
        print(f"[{self.name}] {message}")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({self.name})>"


class FactorStrategy(Strategy):
    """因子策略基类"""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.factors: List[str] = []
        self.weights: Dict[str, float] = {}

    def add_factor(self, factor_name: str, weight: float = 1.0) -> None:
        """添加因子"""
        self.factors.append(factor_name)
        self.weights[factor_name] = weight

    def calculate_composite_score(self, factor_values: pd.DataFrame) -> pd.Series:
        """计算综合因子得分"""
        score = pd.Series(0.0, index=factor_values.index)

        for factor in self.factors:
            if factor in factor_values.columns:
                weight = self.weights.get(factor, 1.0)
                # 标准化因子值
                normalized = (
                    factor_values[factor] - factor_values[factor].mean()
                ) / factor_values[factor].std()
                score += weight * normalized

        return score


class TimingStrategy(Strategy):
    """择时策略基类"""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.market_state: str = "neutral"  # bullish/bearish/neutral

    def update_market_state(self, data: pd.DataFrame) -> str:
        """更新市场状态"""
        return self.market_state

    def get_position_ratio(self) -> float:
        """根据市场状态获取仓位比例"""
        ratios = {
            "bullish": 1.0,
            "neutral": 0.5,
            "bearish": 0.0,
        }
        return ratios.get(self.market_state, 0.5)
