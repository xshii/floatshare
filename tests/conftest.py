"""Pytest 配置和 fixtures。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_daily_data() -> pd.DataFrame:
    """单只股票的日线数据。"""
    dates = pd.date_range("2023-01-01", periods=120, freq="B")
    n = len(dates)

    rng = np.random.default_rng(42)
    base = 10.0
    returns = rng.standard_normal(n) * 0.02
    prices = base * np.exp(np.cumsum(returns))

    return pd.DataFrame(
        {
            "code": "000001.SZ",
            "trade_date": dates,
            "open": prices * (1 + rng.standard_normal(n) * 0.005),
            "high": prices * (1 + np.abs(rng.standard_normal(n) * 0.01)),
            "low": prices * (1 - np.abs(rng.standard_normal(n) * 0.01)),
            "close": prices,
            "volume": rng.integers(100_000, 1_000_000, n),
            "amount": prices * rng.integers(100_000, 1_000_000, n),
        }
    )


@pytest.fixture
def multi_stock_data() -> pd.DataFrame:
    """多只股票的日线数据。"""
    codes = ["000001.SZ", "000002.SZ", "600000.SH"]
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=120, freq="B")

    frames = []
    for code in codes:
        n = len(dates)
        base = float(rng.uniform(5, 50))
        returns = rng.standard_normal(n) * 0.02
        prices = base * np.exp(np.cumsum(returns))
        frames.append(
            pd.DataFrame(
                {
                    "code": code,
                    "trade_date": dates,
                    "open": prices * (1 + rng.standard_normal(n) * 0.005),
                    "high": prices * (1 + np.abs(rng.standard_normal(n) * 0.01)),
                    "low": prices * (1 - np.abs(rng.standard_normal(n) * 0.01)),
                    "close": prices,
                    "volume": rng.integers(100_000, 1_000_000, n),
                    "amount": prices * rng.integers(100_000, 1_000_000, n),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def sample_returns() -> pd.Series:
    """日收益率序列 fixture。"""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    return pd.Series(
        rng.standard_normal(len(dates)) * 0.01 + 0.0003,
        index=dates,
    )
