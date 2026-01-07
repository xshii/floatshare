"""动量因子"""

from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from .base import Factor


class MomentumFactor(Factor):
    """动量因子"""

    name = "momentum"
    description = "N日收益率动量因子"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.period = self.get_param("period", 20)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算动量因子"""
        close = data["close"]
        return close.pct_change(self.period)


class RSIFactor(Factor):
    """RSI因子"""

    name = "rsi"
    description = "相对强弱指标"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.period = self.get_param("period", 14)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算RSI"""
        close = data["close"]
        delta = close.diff()

        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi


class ReversalFactor(Factor):
    """反转因子"""

    name = "reversal"
    description = "短期反转因子"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.period = self.get_param("period", 5)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算反转因子（负动量）"""
        close = data["close"]
        return -close.pct_change(self.period)


class VolumeMomentumFactor(Factor):
    """量价动量因子"""

    name = "volume_momentum"
    description = "量价关系因子"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.period = self.get_param("period", 20)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算量价动量"""
        close = data["close"]
        volume = data["volume"]

        # 价格动量
        price_mom = close.pct_change(self.period)

        # 成交量动量
        vol_mom = volume.pct_change(self.period)

        # 量价同向为正信号
        return price_mom * np.sign(vol_mom)
