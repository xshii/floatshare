"""DB 完整性检查测试 — (code, trade_date) 重复行检测."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from floatshare.application.db_integrity import (
    DbIntegrityError,
    check_trade_date_duplicates,
    raise_if_any_duplicates,
)


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """最小 raw_daily/daily_basic 表."""
    db = tmp_path / "test.db"
    with sqlite3.connect(db) as conn:
        for table in ("raw_daily", "daily_basic", "moneyflow"):
            conn.execute(f"""
                CREATE TABLE {table} (
                    code TEXT, trade_date TEXT, close REAL
                )
            """)
        conn.commit()
    return db


class TestTradeDateDuplicates:
    def test_no_duplicates_passes(self, tmp_db: Path) -> None:
        with sqlite3.connect(tmp_db) as conn:
            conn.executemany(
                "INSERT INTO raw_daily VALUES (?, ?, ?)",
                [
                    ("000001.SZ", "2026-04-20", 11.0),
                    ("000001.SZ", "2026-04-21", 11.5),
                    ("600519.SH", "2026-04-20", 1500.0),
                ],
            )
            conn.commit()
        dups = check_trade_date_duplicates(str(tmp_db))
        assert dups == []
        raise_if_any_duplicates(str(tmp_db))  # 不 raise

    def test_single_duplicate_detected(self, tmp_db: Path) -> None:
        with sqlite3.connect(tmp_db) as conn:
            conn.executemany(
                "INSERT INTO raw_daily VALUES (?, ?, ?)",
                [
                    ("000001.SZ", "2026-04-20", 11.0),
                    ("000001.SZ", "2026-04-20", 11.1),  # 重复!
                ],
            )
            conn.commit()
        dups = check_trade_date_duplicates(str(tmp_db))
        assert len(dups) == 1
        assert dups[0].table == "raw_daily"
        assert dups[0].code == "000001.SZ"
        assert dups[0].trade_date == "2026-04-20"
        assert dups[0].count == 2

    def test_raise_if_any_duplicates_raises(self, tmp_db: Path) -> None:
        with sqlite3.connect(tmp_db) as conn:
            conn.execute("INSERT INTO raw_daily VALUES ('X.SZ', '2026-04-20', 1.0)")
            conn.execute("INSERT INTO raw_daily VALUES ('X.SZ', '2026-04-20', 1.1)")
            conn.commit()
        with pytest.raises(DbIntegrityError) as exc_info:
            raise_if_any_duplicates(str(tmp_db))
        issues = exc_info.value.issues
        assert isinstance(issues, list)
        assert len(issues) == 1
        assert "raw_daily:1" in str(exc_info.value)

    def test_cross_table_duplicates(self, tmp_db: Path) -> None:
        with sqlite3.connect(tmp_db) as conn:
            conn.execute("INSERT INTO raw_daily VALUES ('A.SZ', '2026-04-20', 1.0)")
            conn.execute("INSERT INTO raw_daily VALUES ('A.SZ', '2026-04-20', 1.1)")
            conn.execute("INSERT INTO daily_basic VALUES ('B.SZ', '2026-04-20', 2.0)")
            conn.execute("INSERT INTO daily_basic VALUES ('B.SZ', '2026-04-20', 2.1)")
            conn.commit()
        dups = check_trade_date_duplicates(str(tmp_db))
        tables = {d.table for d in dups}
        assert tables == {"raw_daily", "daily_basic"}

    def test_nonexistent_table_ignored(self, tmp_db: Path) -> None:
        """检测 'index_daily' 等未建表不应崩."""
        dups = check_trade_date_duplicates(str(tmp_db), tables=("raw_daily", "no_such_table"))
        assert dups == []

    def test_3_way_duplicate_counted(self, tmp_db: Path) -> None:
        """同一行重复 3 次, count=3."""
        with sqlite3.connect(tmp_db) as conn:
            for _ in range(3):
                conn.execute("INSERT INTO moneyflow VALUES ('C.SZ', '2026-04-20', 1.0)")
            conn.commit()
        dups = check_trade_date_duplicates(str(tmp_db))
        assert len(dups) == 1
        assert dups[0].count == 3
