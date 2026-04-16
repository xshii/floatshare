"""动量类因子 — RSI 走 `ta`，其余直接 pandas 一行。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator

from .base import Factor


class MomentumFactor(Factor):
    """N 日收益率动量"""

    name = "momentum"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        period = int(self.get_param("period", 20))
        return data["close"].pct_change(period)


class RSIFactor(Factor):
    """RSI 相对强弱"""

    name = "rsi"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        period = int(self.get_param("period", 14))
        return RSIIndicator(close=data["close"], window=period).rsi()


class ReversalFactor(Factor):
    """短期反转（负动量）"""

    name = "reversal"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        period = int(self.get_param("period", 5))
        return -data["close"].pct_change(period)


class VolumeMomentumFactor(Factor):
    """量价动量：价格动量 × 成交量动量符号"""

    name = "volume_momentum"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        period = int(self.get_param("period", 20))
        price_mom = data["close"].pct_change(period)
        vol_mom = data["volume"].pct_change(period)
        return price_mom * np.sign(vol_mom)
