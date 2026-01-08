"""动量策略"""

from typing import List, Dict, Any, Optional
import pandas as pd

from src.strategy.base import Strategy, Signal, StrategyContext
from src.strategy.registry import StrategyRegistry


@StrategyRegistry.register("momentum")
class MomentumStrategy(Strategy):
    """
    动量策略

    - N日涨幅超过阈值买入
    - N日跌幅超过阈值卖出
    - 追涨杀跌，适合趋势市场
    """

    name = "Momentum"
    description = "动量追涨策略"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.lookback = self.get_param("lookback", 20)  # 回看周期
        self.buy_threshold = self.get_param("buy_threshold", 0.05)  # 买入阈值 5%
        self.sell_threshold = self.get_param("sell_threshold", -0.03)  # 卖出阈值 -3%
        self.position_pct = self.get_param("position_pct", 0.8)

        self._history: Dict[str, pd.DataFrame] = {}

    def init(self, context: StrategyContext) -> None:
        self.log(f"动量策略初始化: lookback={self.lookback}, "
                 f"buy>{self.buy_threshold:.1%}, sell<{self.sell_threshold:.1%}")

    def handle_data(
        self, context: StrategyContext, data: pd.DataFrame
    ) -> List[Signal]:
        signals = []

        for code in data["code"].unique():
            stock_data = data[data["code"] == code]

            if code not in self._history:
                self._history[code] = stock_data.copy()
            else:
                self._history[code] = pd.concat(
                    [self._history[code], stock_data]
                ).drop_duplicates(subset=["trade_date"]).tail(self.lookback + 5)

            history = self._history[code]
            if len(history) < self.lookback + 1:
                continue

            close = history["close"]
            current_price = close.iloc[-1]
            past_price = close.iloc[-self.lookback - 1]

            momentum = (current_price - past_price) / past_price
            has_position = code in context.positions and context.positions[code] > 0

            # 动量强买入
            if momentum > self.buy_threshold and not has_position:
                signals.append(
                    Signal(
                        code=code,
                        direction="buy",
                        strength=self.position_pct,
                        price=current_price,
                        reason=f"{self.lookback}日动量: {momentum:.2%}",
                    )
                )

            # 动量弱卖出
            elif momentum < self.sell_threshold and has_position:
                signals.append(
                    Signal(
                        code=code,
                        direction="sell",
                        strength=1.0,
                        price=current_price,
                        reason=f"{self.lookback}日动量: {momentum:.2%}",
                    )
                )

        return signals

    def before_trading(self, context: StrategyContext) -> None:
        pass

    def after_trading(self, context: StrategyContext) -> None:
        pass


@StrategyRegistry.register("turtle")
class TurtleStrategy(Strategy):
    """
    海龟交易策略

    - 突破N日最高价买入
    - 跌破M日最低价卖出
    - 经典趋势跟踪策略
    """

    name = "Turtle"
    description = "海龟突破策略"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.entry_period = self.get_param("entry_period", 20)  # 入场周期
        self.exit_period = self.get_param("exit_period", 10)  # 出场周期
        self.position_pct = self.get_param("position_pct", 0.8)

        self._history: Dict[str, pd.DataFrame] = {}

    def init(self, context: StrategyContext) -> None:
        self.log(f"海龟策略初始化: entry={self.entry_period}日, exit={self.exit_period}日")

    def handle_data(
        self, context: StrategyContext, data: pd.DataFrame
    ) -> List[Signal]:
        signals = []

        for code in data["code"].unique():
            stock_data = data[data["code"] == code]

            if code not in self._history:
                self._history[code] = stock_data.copy()
            else:
                self._history[code] = pd.concat(
                    [self._history[code], stock_data]
                ).drop_duplicates(subset=["trade_date"]).tail(max(self.entry_period, self.exit_period) + 5)

            history = self._history[code]
            if len(history) < self.entry_period + 1:
                continue

            high = history["high"]
            low = history["low"]
            close = history["close"]

            current_price = close.iloc[-1]
            # 不包含当日
            entry_high = high.iloc[-self.entry_period - 1:-1].max()
            exit_low = low.iloc[-self.exit_period - 1:-1].min()

            has_position = code in context.positions and context.positions[code] > 0

            # 突破N日最高价买入
            if current_price > entry_high and not has_position:
                signals.append(
                    Signal(
                        code=code,
                        direction="buy",
                        strength=self.position_pct,
                        price=current_price,
                        reason=f"突破{self.entry_period}日高点: {entry_high:.2f}",
                    )
                )

            # 跌破M日最低价卖出
            elif current_price < exit_low and has_position:
                signals.append(
                    Signal(
                        code=code,
                        direction="sell",
                        strength=1.0,
                        price=current_price,
                        reason=f"跌破{self.exit_period}日低点: {exit_low:.2f}",
                    )
                )

        return signals

    def before_trading(self, context: StrategyContext) -> None:
        pass

    def after_trading(self, context: StrategyContext) -> None:
        pass
