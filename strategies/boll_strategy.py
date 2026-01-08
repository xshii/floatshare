"""布林带策略"""

from typing import List, Dict, Any, Optional
import pandas as pd

from src.strategy.base import Strategy, Signal, StrategyContext
from src.strategy.registry import StrategyRegistry


@StrategyRegistry.register("boll")
class BollStrategy(Strategy):
    """
    布林带策略

    - 价格触及下轨买入（超卖）
    - 价格触及上轨卖出（超买）
    - 可选：突破模式（突破上轨买入，突破下轨卖出）
    """

    name = "Boll"
    description = "布林带策略"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.period = self.get_param("period", 20)
        self.std_dev = self.get_param("std_dev", 2.0)
        self.mode = self.get_param("mode", "revert")  # revert:均值回归, breakout:突破
        self.position_pct = self.get_param("position_pct", 0.8)

        self._history: Dict[str, pd.DataFrame] = {}

    def init(self, context: StrategyContext) -> None:
        self.log(f"布林带策略初始化: period={self.period}, std={self.std_dev}, mode={self.mode}")

    def _calculate_boll(self, close: pd.Series) -> tuple:
        """计算布林带"""
        middle = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()
        upper = middle + self.std_dev * std
        lower = middle - self.std_dev * std
        return upper, middle, lower

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
                ).drop_duplicates(subset=["trade_date"]).tail(self.period + 10)

            history = self._history[code]
            if len(history) < self.period:
                continue

            close = history["close"]
            upper, middle, lower = self._calculate_boll(close)

            current_price = close.iloc[-1]
            current_upper = upper.iloc[-1]
            current_lower = lower.iloc[-1]
            current_middle = middle.iloc[-1]

            has_position = code in context.positions and context.positions[code] > 0

            if self.mode == "revert":
                # 均值回归模式
                # 触及下轨买入
                if current_price <= current_lower and not has_position:
                    signals.append(
                        Signal(
                            code=code,
                            direction="buy",
                            strength=self.position_pct,
                            price=current_price,
                            reason=f"触及下轨: {current_price:.2f} <= {current_lower:.2f}",
                        )
                    )
                # 触及上轨或回到中轨卖出
                elif current_price >= current_upper and has_position:
                    signals.append(
                        Signal(
                            code=code,
                            direction="sell",
                            strength=1.0,
                            price=current_price,
                            reason=f"触及上轨: {current_price:.2f} >= {current_upper:.2f}",
                        )
                    )

            elif self.mode == "breakout":
                # 突破模式
                prev_price = close.iloc[-2] if len(close) > 1 else current_price
                prev_upper = upper.iloc[-2] if len(upper) > 1 else current_upper
                prev_lower = lower.iloc[-2] if len(lower) > 1 else current_lower

                # 突破上轨买入
                if prev_price <= prev_upper and current_price > current_upper:
                    if not has_position:
                        signals.append(
                            Signal(
                                code=code,
                                direction="buy",
                                strength=self.position_pct,
                                price=current_price,
                                reason=f"突破上轨: {current_price:.2f}",
                            )
                        )
                # 跌破下轨卖出
                elif prev_price >= prev_lower and current_price < current_lower:
                    if has_position:
                        signals.append(
                            Signal(
                                code=code,
                                direction="sell",
                                strength=1.0,
                                price=current_price,
                                reason=f"跌破下轨: {current_price:.2f}",
                            )
                        )

        return signals

    def before_trading(self, context: StrategyContext) -> None:
        pass

    def after_trading(self, context: StrategyContext) -> None:
        pass
