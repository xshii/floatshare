"""技术因子 — `ta` 库薄封装，dataclass 风格。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd
from ta.momentum import StochasticOscillator
from ta.trend import MACD, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands


def _close(data: pd.DataFrame) -> pd.Series:
    return cast(pd.Series, data["close"])


def _high(data: pd.DataFrame) -> pd.Series:
    return cast(pd.Series, data["high"])


def _low(data: pd.DataFrame) -> pd.Series:
    return cast(pd.Series, data["low"])


@dataclass(frozen=True, slots=True)
class MAFactor:
    """均线偏离度 (close - MA) / MA"""

    name: str = "ma"
    period: int = 20

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        close = _close(data)
        ma = SMAIndicator(close=close, window=self.period).sma_indicator()
        return (close - ma) / ma


@dataclass(frozen=True, slots=True)
class MACDFactor:
    """MACD 柱"""

    name: str = "macd"
    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9

    def _macd(self, data: pd.DataFrame) -> MACD:
        return MACD(
            close=_close(data),
            window_fast=self.fast_period,
            window_slow=self.slow_period,
            window_sign=self.signal_period,
        )

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        return self._macd(data).macd_diff()

    def calculate_full(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        m = self._macd(data)
        return m.macd(), m.macd_signal(), m.macd_diff()


@dataclass(frozen=True, slots=True)
class BollFactor:
    """布林带位置 (close - lower) / (upper - lower)"""

    name: str = "boll"
    period: int = 20
    std_dev: int = 2

    def _boll(self, data: pd.DataFrame) -> BollingerBands:
        return BollingerBands(
            close=_close(data),
            window=self.period,
            window_dev=self.std_dev,
        )

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        b = self._boll(data)
        upper = b.bollinger_hband()
        lower = b.bollinger_lband()
        return (_close(data) - lower) / (upper - lower)

    def calculate_bands(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        b = self._boll(data)
        return b.bollinger_hband(), b.bollinger_mavg(), b.bollinger_lband()


@dataclass(frozen=True, slots=True)
class ATRFactor:
    """ATR 波动率"""

    name: str = "atr"
    period: int = 14

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        return AverageTrueRange(
            high=_high(data),
            low=_low(data),
            close=_close(data),
            window=self.period,
        ).average_true_range()


@dataclass(frozen=True, slots=True)
class KDJFactor:
    """KDJ 中的 J 值"""

    name: str = "kdj"
    n: int = 9
    m1: int = 3

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        st = StochasticOscillator(
            high=_high(data),
            low=_low(data),
            close=_close(data),
            window=self.n,
            smooth_window=self.m1,
        )
        k = st.stoch()
        d = st.stoch_signal()
        return 3 * k - 2 * d
