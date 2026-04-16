"""动量类因子 — RSI 走 ta，其余直接 pandas 一行。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator


def _close(data: pd.DataFrame) -> pd.Series:
    """从 OHLCV df 取 close 列；pandas stub 把单列索引建模成 Series|DataFrame union，需要 cast 收窄。"""
    return cast(pd.Series, data["close"])


def _volume(data: pd.DataFrame) -> pd.Series:
    return cast(pd.Series, data["volume"])


@dataclass(frozen=True, slots=True)
class MomentumFactor:
    name: str = "momentum"
    period: int = 20

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        return _close(data).pct_change(self.period)


@dataclass(frozen=True, slots=True)
class RSIFactor:
    name: str = "rsi"
    period: int = 14

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        return RSIIndicator(close=_close(data), window=self.period).rsi()


@dataclass(frozen=True, slots=True)
class ReversalFactor:
    name: str = "reversal"
    period: int = 5

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        return -_close(data).pct_change(self.period)


@dataclass(frozen=True, slots=True)
class VolumeMomentumFactor:
    name: str = "volume_momentum"
    period: int = 20

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        price_mom = _close(data).pct_change(self.period)
        vol_mom = _volume(data).pct_change(self.period)
        return price_mom * np.sign(vol_mom)
