"""均线交叉策略"""

from typing import List, Dict, Any, Optional
import pandas as pd

from src.strategy.base import Strategy, Signal, StrategyContext
from src.strategy.registry import StrategyRegistry


@StrategyRegistry.register("ma_cross")
class MACrossStrategy(Strategy):
    """
    均线交叉策略

    当短期均线上穿长期均线时买入，下穿时卖出。
    """

    name = "MACross"
    description = "均线交叉策略"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.short_period = self.get_param("short_period", 5)
        self.long_period = self.get_param("long_period", 20)
        self.position_pct = self.get_param("position_pct", 0.9)  # 仓位比例

        # 内部状态
        self._ma_short: Dict[str, pd.Series] = {}
        self._ma_long: Dict[str, pd.Series] = {}
        self._history: Dict[str, pd.DataFrame] = {}

    def init(self, context: StrategyContext) -> None:
        """策略初始化"""
        self.log(f"初始化均线交叉策略: MA{self.short_period}/{self.long_period}")

    def handle_data(
        self, context: StrategyContext, data: pd.DataFrame
    ) -> List[Signal]:
        """处理每日数据"""
        signals = []

        # 按股票分组处理
        for code in data["code"].unique():
            stock_data = data[data["code"] == code]

            # 更新历史数据
            if code not in self._history:
                self._history[code] = stock_data.copy()
            else:
                self._history[code] = pd.concat(
                    [self._history[code], stock_data]
                ).drop_duplicates(subset=["trade_date"]).tail(self.long_period + 10)

            history = self._history[code]
            if len(history) < self.long_period:
                continue

            # 计算均线
            close = history["close"]
            ma_short = close.rolling(self.short_period).mean()
            ma_long = close.rolling(self.long_period).mean()

            # 获取最新值
            current_short = ma_short.iloc[-1]
            current_long = ma_long.iloc[-1]
            prev_short = ma_short.iloc[-2] if len(ma_short) > 1 else current_short
            prev_long = ma_long.iloc[-2] if len(ma_long) > 1 else current_long

            current_price = close.iloc[-1]
            has_position = code in context.positions and context.positions[code] > 0

            # 金叉买入
            if prev_short <= prev_long and current_short > current_long:
                if not has_position:
                    signals.append(
                        Signal(
                            code=code,
                            direction="buy",
                            strength=self.position_pct,
                            price=current_price,
                            reason=f"金叉: MA{self.short_period}上穿MA{self.long_period}",
                        )
                    )

            # 死叉卖出
            elif prev_short >= prev_long and current_short < current_long:
                if has_position:
                    signals.append(
                        Signal(
                            code=code,
                            direction="sell",
                            strength=1.0,  # 全部卖出
                            price=current_price,
                            reason=f"死叉: MA{self.short_period}下穿MA{self.long_period}",
                        )
                    )

        return signals

    def before_trading(self, context: StrategyContext) -> None:
        """盘前处理"""
        pass

    def after_trading(self, context: StrategyContext) -> None:
        """盘后处理"""
        pass
