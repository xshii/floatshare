"""MACD策略"""

from typing import List, Dict, Any, Optional
import pandas as pd

from src.strategy.base import Strategy, Signal, StrategyContext
from src.strategy.registry import StrategyRegistry


@StrategyRegistry.register("macd")
class MACDStrategy(Strategy):
    """
    MACD策略

    - MACD金叉（DIF上穿DEA）买入
    - MACD死叉（DIF下穿DEA）卖出
    - 可选：配合零轴过滤
    """

    name = "MACD"
    description = "MACD金叉死叉策略"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.fast = self.get_param("fast", 12)
        self.slow = self.get_param("slow", 26)
        self.signal = self.get_param("signal", 9)
        self.use_zero_filter = self.get_param("use_zero_filter", False)  # 零轴过滤
        self.position_pct = self.get_param("position_pct", 0.8)

        self._history: Dict[str, pd.DataFrame] = {}

    def init(self, context: StrategyContext) -> None:
        self.log(f"MACD策略初始化: EMA{self.fast}/{self.slow}, Signal={self.signal}")

    def _calculate_macd(self, close: pd.Series) -> tuple:
        """计算MACD"""
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=self.signal, adjust=False).mean()
        macd = (dif - dea) * 2
        return dif, dea, macd

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
                ).drop_duplicates(subset=["trade_date"]).tail(self.slow + 50)

            history = self._history[code]
            if len(history) < self.slow + self.signal:
                continue

            close = history["close"]
            dif, dea, macd = self._calculate_macd(close)

            current_dif = dif.iloc[-1]
            current_dea = dea.iloc[-1]
            prev_dif = dif.iloc[-2]
            prev_dea = dea.iloc[-2]
            current_price = close.iloc[-1]

            has_position = code in context.positions and context.positions[code] > 0

            # 金叉买入
            if prev_dif <= prev_dea and current_dif > current_dea:
                # 零轴过滤：只在零轴上方金叉
                if not self.use_zero_filter or current_dif > 0:
                    if not has_position:
                        signals.append(
                            Signal(
                                code=code,
                                direction="buy",
                                strength=self.position_pct,
                                price=current_price,
                                reason=f"MACD金叉: DIF={current_dif:.3f}",
                            )
                        )

            # 死叉卖出
            elif prev_dif >= prev_dea and current_dif < current_dea:
                if has_position:
                    signals.append(
                        Signal(
                            code=code,
                            direction="sell",
                            strength=1.0,
                            price=current_price,
                            reason=f"MACD死叉: DIF={current_dif:.3f}",
                        )
                    )

        return signals

    def before_trading(self, context: StrategyContext) -> None:
        pass

    def after_trading(self, context: StrategyContext) -> None:
        pass
