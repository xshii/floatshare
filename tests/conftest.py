"""全局 pytest fixtures。"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from loguru import logger

# src layout：测试需要能 import floatshare 包
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def silence_loguru():
    """测试期间静默 loguru 输出，避免污染 stderr。"""
    logger.remove()
    yield
    logger.remove()


@pytest.fixture(autouse=True)
def reset_apprise_cache(monkeypatch):
    """每个测试前重置 apprise 单例 + 清 notify URL — 避免测试意外推 Bark."""
    from floatshare.observability.alert import _client

    # 清 env 防止 notify 真推到用户 Bark (pipeline runner / preflight / healthcheck 都会调)
    monkeypatch.delenv("FLOATSHARE_NOTIFY_URLS", raising=False)
    _client.cache_clear()
    yield
    _client.cache_clear()


def _ohlcv(seed: int, periods: int = 120, base: float = 10.0) -> pd.DataFrame:
    """合成单只股票的 OHLCV。"""
    dates = pd.date_range("2023-01-01", periods=periods, freq="B")
    rng = np.random.default_rng(seed)
    n = len(dates)
    returns = rng.standard_normal(n) * 0.02
    prices = base * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
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
def sample_daily_data() -> pd.DataFrame:
    df = _ohlcv(seed=42)
    df.insert(0, "code", "000001.SZ")
    return df


@pytest.fixture
def multi_stock_data() -> pd.DataFrame:
    frames = []
    for i, code in enumerate(["000001.SZ", "000002.SZ", "600000.SH"]):
        sub = _ohlcv(seed=42 + i, base=10.0 + i * 5)
        sub.insert(0, "code", code)
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def sample_returns() -> pd.Series:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    return pd.Series(
        rng.standard_normal(len(dates)) * 0.01 + 0.0003,
        index=dates,
    )


# ----------------------------------------------------------------------------
# Fake DataSource — 用于 loader 契约 / 集成测试
# ----------------------------------------------------------------------------


class FakeDataSource:
    """实现所有 DataSource Protocol 的内存假源，可控制何时抛错。"""

    def __init__(
        self,
        daily: pd.DataFrame | None = None,
        fail_modes: set[str] | None = None,
    ) -> None:
        self.daily_df = daily if daily is not None else _ohlcv(seed=1)
        self.daily_df.insert(0, "code", "TEST.SZ") if "code" not in self.daily_df.columns else None
        self.fail_modes = fail_modes or set()
        self.calls: dict[str, int] = {}

    def _record(self, op: str) -> None:
        self.calls[op] = self.calls.get(op, 0) + 1
        if op in self.fail_modes:
            from floatshare.interfaces.data_source import DataSourceError

            raise DataSourceError(f"FakeDataSource configured to fail on {op}")

    def get_daily(self, code: str, start=None, end=None, adj: Any = "qfq") -> pd.DataFrame:
        self._record("get_daily")
        return self.daily_df.copy()

    def get_minute(self, code, start=None, end=None, freq=None) -> pd.DataFrame:
        self._record("get_minute")
        return self.daily_df.copy()

    def get_index_daily(self, code, start=None, end=None) -> pd.DataFrame:
        self._record("get_index_daily")
        return self.daily_df.copy()

    def get_trade_calendar(self, start=None, end=None) -> list[date]:
        self._record("get_trade_calendar")
        return [date(2024, 1, 2), date(2024, 1, 3)]

    def get_stock_list(self) -> pd.DataFrame:
        self._record("get_stock_list")
        return pd.DataFrame({"code": ["000001.SZ"], "name": ["平安银行"]})


@pytest.fixture
def fake_source() -> FakeDataSource:
    return FakeDataSource()


@pytest.fixture
def failing_source() -> FakeDataSource:
    return FakeDataSource(fail_modes={"get_daily", "get_index_daily"})
