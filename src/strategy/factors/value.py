"""价值因子"""

from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from src.strategy.factors.base import Factor


class PEFactor(Factor):
    """市盈率因子"""

    name = "pe"
    description = "市盈率倒数（EP）因子"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算EP（市盈率倒数）"""
        if "pe" in data.columns:
            pe = data["pe"].replace(0, np.nan)
            return 1 / pe
        elif "eps" in data.columns and "close" in data.columns:
            eps = data["eps"]
            close = data["close"]
            return eps / close
        else:
            return pd.Series(np.nan, index=data.index)


class PBFactor(Factor):
    """市净率因子"""

    name = "pb"
    description = "市净率倒数（BP）因子"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算BP（市净率倒数）"""
        if "pb" in data.columns:
            pb = data["pb"].replace(0, np.nan)
            return 1 / pb
        elif "bps" in data.columns and "close" in data.columns:
            bps = data["bps"]
            close = data["close"]
            return bps / close
        else:
            return pd.Series(np.nan, index=data.index)


class DividendYieldFactor(Factor):
    """股息率因子"""

    name = "dividend_yield"
    description = "股息率因子"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算股息率"""
        if "dividend_yield" in data.columns:
            return data["dividend_yield"]
        elif "dps" in data.columns and "close" in data.columns:
            return data["dps"] / data["close"]
        else:
            return pd.Series(np.nan, index=data.index)


class ROEFactor(Factor):
    """ROE因子"""

    name = "roe"
    description = "净资产收益率因子"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """获取ROE"""
        if "roe" in data.columns:
            return data["roe"]
        elif "net_profit" in data.columns and "total_equity" in data.columns:
            equity = data["total_equity"].replace(0, np.nan)
            return data["net_profit"] / equity
        else:
            return pd.Series(np.nan, index=data.index)


class PSFactor(Factor):
    """市销率因子"""

    name = "ps"
    description = "市销率倒数因子"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算市销率倒数"""
        if "ps" in data.columns:
            ps = data["ps"].replace(0, np.nan)
            return 1 / ps
        elif "revenue" in data.columns and "market_cap" in data.columns:
            market_cap = data["market_cap"].replace(0, np.nan)
            return data["revenue"] / market_cap
        else:
            return pd.Series(np.nan, index=data.index)
