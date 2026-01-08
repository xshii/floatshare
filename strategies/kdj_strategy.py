"""KDJ策略"""

from typing import List, Dict, Any, Optional
import pandas as pd

from src.strategy.base import Strategy, Signal, StrategyContext
from src.strategy.registry import StrategyRegistry


@StrategyRegistry.register("kdj")
class KDJStrategy(Strategy):
    """
    KDJ随机指标策略

    - K线上穿D线买入（金叉）
    - K线下穿D线卖出（死叉）
    - 可选：配合超买超卖区域过滤
    """

    name = "KDJ"
    description = "KDJ随机指标策略"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.n = self.get_param("n", 9)  # RSV周期
        self.m1 = self.get_param("m1", 3)  # K平滑周期
        self.m2 = self.get_param("m2", 3)  # D平滑周期
        self.oversold = self.get_param("oversold", 20)  # 超卖线
        self.overbought = self.get_param("overbought", 80)  # 超买线
        self.use_zone_filter = self.get_param("use_zone_filter", True)
        self.position_pct = self.get_param("position_pct", 0.8)

        self._history: Dict[str, pd.DataFrame] = {}

    def init(self, context: StrategyContext) -> None:
        self.log(f"KDJ策略初始化: N={self.n}, M1={self.m1}, M2={self.m2}")

    def _calculate_kdj(self, high: pd.Series, low: pd.Series, close: pd.Series) -> tuple:
        """计算KDJ"""
        lowest_low = low.rolling(window=self.n).min()
        highest_high = high.rolling(window=self.n).max()

        rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
        rsv = rsv.fillna(50)

        k = rsv.ewm(com=self.m1 - 1, adjust=False).mean()
        d = k.ewm(com=self.m2 - 1, adjust=False).mean()
        j = 3 * k - 2 * d

        return k, d, j

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
                ).drop_duplicates(subset=["trade_date"]).tail(self.n + 20)

            history = self._history[code]
            if len(history) < self.n + 5:
                continue

            high = history["high"]
            low = history["low"]
            close = history["close"]

            k, d, j = self._calculate_kdj(high, low, close)

            current_k = k.iloc[-1]
            current_d = d.iloc[-1]
            prev_k = k.iloc[-2]
            prev_d = d.iloc[-2]
            current_price = close.iloc[-1]

            has_position = code in context.positions and context.positions[code] > 0

            # 金叉买入
            if prev_k <= prev_d and current_k > current_d:
                # 超卖区域金叉更有效
                if not self.use_zone_filter or current_k < self.overbought:
                    if not has_position:
                        zone = "超卖区" if current_k < self.oversold else ""
                        signals.append(
                            Signal(
                                code=code,
                                direction="buy",
                                strength=self.position_pct,
                                price=current_price,
                                reason=f"KDJ金叉{zone}: K={current_k:.1f}, D={current_d:.1f}",
                            )
                        )

            # 死叉卖出
            elif prev_k >= prev_d and current_k < current_d:
                if has_position:
                    zone = "超买区" if current_k > self.overbought else ""
                    signals.append(
                        Signal(
                            code=code,
                            direction="sell",
                            strength=1.0,
                            price=current_price,
                            reason=f"KDJ死叉{zone}: K={current_k:.1f}, D={current_d:.1f}",
                        )
                    )

        return signals

    def before_trading(self, context: StrategyContext) -> None:
        pass

    def after_trading(self, context: StrategyContext) -> None:
        pass
