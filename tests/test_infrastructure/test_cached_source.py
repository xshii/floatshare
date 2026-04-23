"""CachedSource 降级链测试 — 验证 cache miss 抛 DataSourceError。"""

from __future__ import annotations

from pathlib import Path

import pytest

from floatshare.infrastructure.data_sources.cached import CachedSource, daily_key
from floatshare.infrastructure.storage.cache import CacheManager
from floatshare.interfaces.data_source import DataSourceError


class TestCachedSource:
    def test_miss_raises(self, tmp_path: Path) -> None:
        src = CachedSource(CacheManager(cache_dir=tmp_path / "cache"))
        with pytest.raises(DataSourceError, match="cache miss"):
            src.get_daily("000001.SZ")

    def test_hit_returns_data(self, tmp_path: Path, sample_daily_data) -> None:
        cache = CacheManager(cache_dir=tmp_path / "cache")
        src = CachedSource(cache)
        src.put(daily_key("000001.SZ"), sample_daily_data)
        df = src.get_daily("000001.SZ")
        assert not df.empty

    def test_stock_list_miss(self, tmp_path: Path) -> None:
        src = CachedSource(CacheManager(cache_dir=tmp_path / "cache"))
        with pytest.raises(DataSourceError, match="cache miss"):
            src.get_stock_list()

    def test_calendar_miss(self, tmp_path: Path) -> None:
        src = CachedSource(CacheManager(cache_dir=tmp_path / "cache"))
        with pytest.raises(DataSourceError, match="cache miss"):
            src.get_trade_calendar()
