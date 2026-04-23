"""web/data.py — 批量最新价 + 持仓视图 单元测试。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from floatshare.application.treasury import Treasury
from floatshare.domain.enums import DcaFrequency
from floatshare.domain.records import RawDaily
from floatshare.infrastructure.storage.database import DatabaseStorage


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DatabaseStorage:
    """隔离 DB + 替换 web.data 模块级常量指向它。"""
    db = DatabaseStorage(db_path=tmp_path / "web.db")
    db.init_tables()
    import floatshare.web.data as d

    monkeypatch.setattr(d, "DB", db)
    monkeypatch.setattr(d, "BOOK", Treasury(db))
    return db


def _seed_klines(db: DatabaseStorage, code: str, closes: list[tuple[str, float]]) -> None:
    df = pd.DataFrame(
        [
            {
                "code": code,
                "trade_date": dt,
                "open": c,
                "high": c,
                "low": c,
                "close": c,
                "volume": 1000,
            }
            for dt, c in closes
        ]
    )
    db.save(RawDaily, df)


class TestLatestCloses:
    def test_returns_latest_per_code(self, db):
        from floatshare.web.data import _latest_closes

        _seed_klines(db, "A", [("2024-01-01", 10.0), ("2024-02-01", 12.0)])
        _seed_klines(db, "B", [("2024-01-05", 20.0)])
        prices = _latest_closes(["A", "B"])
        assert prices["A"] == pytest.approx(12.0)
        assert prices["B"] == pytest.approx(20.0)

    def test_empty_codes(self, db):
        from floatshare.web.data import _latest_closes

        assert _latest_closes([]) == {}

    def test_missing_code(self, db):
        from floatshare.web.data import _latest_closes

        _seed_klines(db, "A", [("2024-01-01", 10.0)])
        prices = _latest_closes(["A", "GHOST"])
        assert prices == {"A": 10.0}


class TestHoldingsWithPrices:
    def test_empty_when_no_dca(self, db):
        from floatshare.web.data import BOOK, holdings_with_prices

        BOOK.open_account("ACC", "test")
        assert holdings_with_prices("ACC").empty

    def test_pnl_computed_correctly(self, db):
        from floatshare.web.data import BOOK, holdings_with_prices

        _seed_klines(db, "A", [("2026-01-01", 4.0), ("2026-02-01", 5.0)])
        BOOK.open_account("ACC", "test")
        BOOK.deposit("ACC", 10_000)
        BOOK.create_plan("P", "ACC", "A", 1000, DcaFrequency.WEEKLY, date(2026, 1, 1))
        BOOK.execute_plan("P", date(2026, 1, 1), price=4.0)  # 250 股成本 1000

        df = holdings_with_prices("ACC")
        assert len(df) == 1
        row = df.iloc[0]
        assert row["last_price"] == pytest.approx(5.0)  # 最新收盘
        assert row["market_value"] == pytest.approx(250 * 5.0)
        assert row["pnl"] == pytest.approx(250 * 5.0 - 1000)


class TestTableCounts:
    def test_iterates_all_records(self, db, monkeypatch, tmp_path):
        from floatshare.application import db_snapshot
        from floatshare.domain.records import ALL_RECORDS
        from floatshare.web import data as _wd

        # 隔离 snapshot 路径 + 失效内存 cache, 强制走真实 DB 查询
        monkeypatch.setattr(db_snapshot, "COUNTS_SNAPSHOT_PATH", tmp_path / "table-counts.json")
        monkeypatch.setattr(_wd, "_counts_cache", None)
        df = _wd.table_counts()
        assert not df.empty
        expected_tables = {r.TABLE for r in ALL_RECORDS if hasattr(r, "TABLE")}
        assert expected_tables.issubset(set(df["表"]))


class TestAccountSummary:
    def test_zero_when_no_activity(self, db):
        from floatshare.web.data import BOOK, account_summary

        BOOK.open_account("ACC", "test")
        s = account_summary("ACC")
        assert s["cash"] == 0.0
        assert s["market_value"] == 0.0
        assert s["invested"] == 0.0

    def test_invested_reflects_deposits_only(self, db):
        from floatshare.web.data import BOOK, account_summary

        BOOK.open_account("ACC", "test")
        BOOK.deposit("ACC", 50_000)
        BOOK.withdraw("ACC", 10_000)
        s = account_summary("ACC")
        assert s["invested"] == pytest.approx(40_000)
        assert s["cash"] == pytest.approx(40_000)
