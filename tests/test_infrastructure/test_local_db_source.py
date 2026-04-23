"""LocalDbSource 降级链测试 — 验证 db miss 抛 DataSourceError。"""

from __future__ import annotations

from pathlib import Path

import pytest

from floatshare.domain.records import RawDaily
from floatshare.infrastructure.data_sources.local_db import LocalDbSource
from floatshare.infrastructure.storage.database import DatabaseStorage
from floatshare.interfaces.data_source import DataSourceError


class TestLocalDbSource:
    def test_daily_miss_raises(self, tmp_path: Path) -> None:
        db = DatabaseStorage(db_path=tmp_path / "test.db")
        db.init_tables()
        src = LocalDbSource(db)
        with pytest.raises(DataSourceError, match="local db miss"):
            src.get_daily("000001.SZ")

    def test_daily_hit(self, tmp_path: Path, sample_daily_data) -> None:
        db = DatabaseStorage(db_path=tmp_path / "test.db")
        db.init_tables()
        db.save(RawDaily, sample_daily_data)
        src = LocalDbSource(db)
        df = src.get_daily("000001.SZ")
        assert not df.empty

    def test_stock_list_miss_raises(self, tmp_path: Path) -> None:
        db = DatabaseStorage(db_path=tmp_path / "test.db")
        db.init_tables()
        src = LocalDbSource(db)
        with pytest.raises(DataSourceError, match="local db miss"):
            src.get_stock_list()
