"""价值因子 — 涉及 A 股财报字段的业务逻辑，不依赖外部库。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd


def _col(data: pd.DataFrame, name: str) -> pd.Series:
    return cast(pd.Series, data[name])


@dataclass(frozen=True, slots=True)
class PEFactor:
    name: str = "pe"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        if "pe" in data.columns:
            pe = _col(data, "pe").replace(0, np.nan)
            return 1 / pe
        if "eps" in data.columns and "close" in data.columns:
            return _col(data, "eps") / _col(data, "close")
        return pd.Series(np.nan, index=data.index)


@dataclass(frozen=True, slots=True)
class PBFactor:
    name: str = "pb"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        if "pb" in data.columns:
            pb = _col(data, "pb").replace(0, np.nan)
            return 1 / pb
        if "bps" in data.columns and "close" in data.columns:
            return _col(data, "bps") / _col(data, "close")
        return pd.Series(np.nan, index=data.index)


@dataclass(frozen=True, slots=True)
class DividendYieldFactor:
    name: str = "dividend_yield"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        if "dividend_yield" in data.columns:
            return _col(data, "dividend_yield")
        if "dps" in data.columns and "close" in data.columns:
            return _col(data, "dps") / _col(data, "close")
        return pd.Series(np.nan, index=data.index)


@dataclass(frozen=True, slots=True)
class ROEFactor:
    name: str = "roe"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        if "roe" in data.columns:
            return _col(data, "roe")
        if "net_profit" in data.columns and "total_equity" in data.columns:
            equity = _col(data, "total_equity").replace(0, np.nan)
            return _col(data, "net_profit") / equity
        return pd.Series(np.nan, index=data.index)


@dataclass(frozen=True, slots=True)
class PSFactor:
    name: str = "ps"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        if "ps" in data.columns:
            ps = _col(data, "ps").replace(0, np.nan)
            return 1 / ps
        if "revenue" in data.columns and "market_cap" in data.columns:
            mc = _col(data, "market_cap").replace(0, np.nan)
            return _col(data, "revenue") / mc
        return pd.Series(np.nan, index=data.index)
