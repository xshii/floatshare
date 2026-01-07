"""Dual Thrust策略"""

from typing import List, Dict, Any, Optional
import pandas as pd

from src.strategy.base import Strategy, Signal, StrategyContext
from src.strategy.registry import StrategyRegistry


@StrategyRegistry.register("dual_thrust")
class DualThrustStrategy(Strategy):
    """
    Dual Thrust策略

    基于前N日的最高价、最低价和收盘价计算区间，
    当价格突破上轨买入，突破下轨卖出。
    """

    name = "DualThrust"
    description = "Dual Thrust突破策略"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.lookback = self.get_param("lookback", 4)  # 回看天数
        self.k1 = self.get_param("k1", 0.5)  # 上轨系数
        self.k2 = self.get_param("k2", 0.5)  # 下轨系数
        self.position_pct = self.get_param("position_pct", 0.8)

        self._history: Dict[str, pd.DataFrame] = {}

    def init(self, context: StrategyContext) -> None:
        """策略初始化"""
        self.log(f"初始化DualThrust策略: lookback={self.lookback}, k1={self.k1}, k2={self.k2}")

    def _calculate_range(self, history: pd.DataFrame) -> tuple:
        """计算Range"""
        hh = history["high"].max()  # 最高价
        hc = history["close"].max()  # 最高收盘价
        lc = history["close"].min()  # 最低收盘价
        ll = history["low"].min()  # 最低价

        range_val = max(hh - lc, hc - ll)
        return range_val

    def handle_data(
        self, context: StrategyContext, data: pd.DataFrame
    ) -> List[Signal]:
        """处理每日数据"""
        signals = []

        for code in data["code"].unique():
            stock_data = data[data["code"] == code]

            # 更新历史数据
            if code not in self._history:
                self._history[code] = stock_data.copy()
            else:
                self._history[code] = pd.concat(
                    [self._history[code], stock_data]
                ).drop_duplicates(subset=["trade_date"]).tail(self.lookback + 5)

            history = self._history[code]
            if len(history) < self.lookback + 1:
                continue

            # 使用前N天数据计算Range
            prev_data = history.iloc[:-1].tail(self.lookback)
            range_val = self._calculate_range(prev_data)

            # 今日开盘价
            today_open = stock_data["open"].iloc[0]
            today_close = stock_data["close"].iloc[0]

            # 计算上下轨
            upper = today_open + self.k1 * range_val
            lower = today_open - self.k2 * range_val

            has_position = code in context.positions and context.positions[code] > 0

            # 突破上轨买入
            if today_close > upper and not has_position:
                signals.append(
                    Signal(
                        code=code,
                        direction="buy",
                        strength=self.position_pct,
                        price=today_close,
                        reason=f"突破上轨: {today_close:.2f} > {upper:.2f}",
                    )
                )

            # 突破下轨卖出
            elif today_close < lower and has_position:
                signals.append(
                    Signal(
                        code=code,
                        direction="sell",
                        strength=1.0,
                        price=today_close,
                        reason=f"突破下轨: {today_close:.2f} < {lower:.2f}",
                    )
                )

        return signals

    def before_trading(self, context: StrategyContext) -> None:
        pass

    def after_trading(self, context: StrategyContext) -> None:
        pass
