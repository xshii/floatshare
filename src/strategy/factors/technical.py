"""技术因子 — 薄封装 `ta` 库，不再手写公式。"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from ta.momentum import StochasticOscillator
from ta.trend import EMAIndicator, MACD, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands

from .base import Factor


class MAFactor(Factor):
    """均线偏离度 (close - MA) / MA"""

    name = "ma"
    description = "均线偏离度"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        period = int(self.get_param("period", 20))
        close = data["close"]
        ma = SMAIndicator(close=close, window=period).sma_indicator()
        return (close - ma) / ma


class MACDFactor(Factor):
    """MACD 柱"""

    name = "macd"
    description = "MACD柱"

    def _macd(self, data: pd.DataFrame) -> MACD:
        return MACD(
            close=data["close"],
            window_fast=int(self.get_param("fast_period", 12)),
            window_slow=int(self.get_param("slow_period", 26)),
            window_sign=int(self.get_param("signal_period", 9)),
        )

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        return self._macd(data).macd_diff()

    def calculate_full(
        self, data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        m = self._macd(data)
        return m.macd(), m.macd_signal(), m.macd_diff()


class BollFactor(Factor):
    """布林带位置 (close - lower) / (upper - lower)"""

    name = "boll"
    description = "布林带位置"

    def _boll(self, data: pd.DataFrame) -> BollingerBands:
        return BollingerBands(
            close=data["close"],
            window=int(self.get_param("period", 20)),
            window_dev=int(self.get_param("std_dev", 2)),
        )

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        b = self._boll(data)
        upper = b.bollinger_hband()
        lower = b.bollinger_lband()
        return (data["close"] - lower) / (upper - lower)

    def calculate_bands(
        self, data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        b = self._boll(data)
        return b.bollinger_hband(), b.bollinger_mavg(), b.bollinger_lband()


class ATRFactor(Factor):
    """ATR 波动率"""

    name = "atr"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        return AverageTrueRange(
            high=data["high"],
            low=data["low"],
            close=data["close"],
            window=int(self.get_param("period", 14)),
        ).average_true_range()


class KDJFactor(Factor):
    """KDJ 中的 J 值（基于 `ta` 的 Stochastic 衍生）"""

    name = "kdj"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        n = int(self.get_param("n", 9))
        m1 = int(self.get_param("m1", 3))
        st = StochasticOscillator(
            high=data["high"],
            low=data["low"],
            close=data["close"],
            window=n,
            smooth_window=m1,
        )
        k = st.stoch()
        d = st.stoch_signal()
        return 3 * k - 2 * d
