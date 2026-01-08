"""RSI超买超卖策略"""

from typing import List, Dict, Any, Optional
import pandas as pd

from src.strategy.base import Strategy, Signal, StrategyContext
from src.strategy.registry import StrategyRegistry


@StrategyRegistry.register("rsi")
class RSIStrategy(Strategy):
    """
    RSI 超买超卖策略

    - RSI < 30: 超卖，买入信号
    - RSI > 70: 超买，卖出信号
    """

    name = "RSI"
    description = "RSI超买超卖策略"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        # 策略参数
        self.period = self.get_param("period", 14)
        self.oversold = self.get_param("oversold", 30)
        self.overbought = self.get_param("overbought", 70)
        self.position_pct = self.get_param("position_pct", 0.3)

        # 内部状态
        self._history: Dict[str, pd.DataFrame] = {}

    def init(self, context: StrategyContext) -> None:
        """策略初始化"""
        self.log(f"RSI策略初始化: period={self.period}, "
                 f"超卖<{self.oversold}, 超买>{self.overbought}")

    def _calculate_rsi(self, prices: pd.Series) -> float:
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else 50

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
                ).drop_duplicates(subset=["trade_date"]).tail(self.period + 10)

            history = self._history[code]
            if len(history) < self.period + 1:
                continue

            # 计算 RSI
            close = history["close"]
            rsi = self._calculate_rsi(close)
            current_price = close.iloc[-1]

            # 检查持仓
            has_position = code in context.positions and context.positions[code] > 0

            # 超卖买入
            if rsi < self.oversold and not has_position:
                signals.append(
                    Signal(
                        code=code,
                        direction="buy",
                        strength=self.position_pct,
                        price=current_price,
                        reason=f"RSI超卖: {rsi:.1f} < {self.oversold}",
                    )
                )

            # 超买卖出
            elif rsi > self.overbought and has_position:
                signals.append(
                    Signal(
                        code=code,
                        direction="sell",
                        strength=1.0,
                        price=current_price,
                        reason=f"RSI超买: {rsi:.1f} > {self.overbought}",
                    )
                )

        return signals

    def before_trading(self, context: StrategyContext) -> None:
        """盘前处理"""
        pass

    def after_trading(self, context: StrategyContext) -> None:
        """盘后处理"""
        pass

    def on_order_filled(self, context: StrategyContext, order: Dict) -> None:
        """订单成交回调"""
        self.log(f"订单成交: {order['code']} {order['direction']} "
                 f"{order['filled_quantity']}股 @ {order['filled_price']:.2f}")

    def on_order_rejected(self, context: StrategyContext, order: Dict, reason: str) -> None:
        """订单拒绝回调"""
        self.log(f"订单拒绝: {order['code']} - {reason}")
