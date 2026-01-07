"""技术指标因子"""

from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np

from src.strategy.factors.base import Factor


class MAFactor(Factor):
    """均线因子"""

    name = "ma"
    description = "移动平均线因子"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.period = self.get_param("period", 20)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算均线偏离度"""
        close = data["close"]
        ma = close.rolling(window=self.period).mean()
        return (close - ma) / ma


class MACDFactor(Factor):
    """MACD因子"""

    name = "macd"
    description = "MACD指标"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.fast_period = self.get_param("fast_period", 12)
        self.slow_period = self.get_param("slow_period", 26)
        self.signal_period = self.get_param("signal_period", 9)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算MACD柱状图"""
        close = data["close"]

        ema_fast = close.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_period, adjust=False).mean()

        dif = ema_fast - ema_slow
        dea = dif.ewm(span=self.signal_period, adjust=False).mean()
        macd = (dif - dea) * 2

        return macd

    def calculate_full(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算完整MACD（DIF, DEA, MACD柱）"""
        close = data["close"]

        ema_fast = close.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_period, adjust=False).mean()

        dif = ema_fast - ema_slow
        dea = dif.ewm(span=self.signal_period, adjust=False).mean()
        macd = (dif - dea) * 2

        return dif, dea, macd


class BollFactor(Factor):
    """布林带因子"""

    name = "boll"
    description = "布林带位置因子"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.period = self.get_param("period", 20)
        self.std_dev = self.get_param("std_dev", 2)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算布林带位置（0-1之间）"""
        close = data["close"]

        middle = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()

        upper = middle + self.std_dev * std
        lower = middle - self.std_dev * std

        # 价格在布林带中的相对位置
        position = (close - lower) / (upper - lower)
        return position

    def calculate_bands(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算布林带三轨"""
        close = data["close"]

        middle = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()

        upper = middle + self.std_dev * std
        lower = middle - self.std_dev * std

        return upper, middle, lower


class KDJFactor(Factor):
    """KDJ因子"""

    name = "kdj"
    description = "KDJ随机指标"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.n = self.get_param("n", 9)
        self.m1 = self.get_param("m1", 3)
        self.m2 = self.get_param("m2", 3)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算J值"""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        low_n = low.rolling(window=self.n).min()
        high_n = high.rolling(window=self.n).max()

        rsv = (close - low_n) / (high_n - low_n) * 100

        k = rsv.ewm(span=self.m1, adjust=False).mean()
        d = k.ewm(span=self.m2, adjust=False).mean()
        j = 3 * k - 2 * d

        return j

    def calculate_full(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算完整KDJ"""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        low_n = low.rolling(window=self.n).min()
        high_n = high.rolling(window=self.n).max()

        rsv = (close - low_n) / (high_n - low_n) * 100

        k = rsv.ewm(span=self.m1, adjust=False).mean()
        d = k.ewm(span=self.m2, adjust=False).mean()
        j = 3 * k - 2 * d

        return k, d, j


class ATRFactor(Factor):
    """ATR波动率因子"""

    name = "atr"
    description = "平均真实波幅"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.period = self.get_param("period", 14)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算ATR"""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.period).mean()

        return atr
