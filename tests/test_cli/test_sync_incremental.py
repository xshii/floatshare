"""run_sync 增量同步逻辑测试 — 覆盖:

Bug 1: 历史回填 — local=[2022,2024], requested=[2020,2025] 应回填 [2020,2021]
Bug 2: 跳过未来日期 — next_day > today 不发请求
Lifecycle: 用上市/退市日收窄请求窗口
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from floatshare.cli.run_sync import (
    _effective_window,
    _missing_ranges,
    _sync_daily_table,
)
from floatshare.domain.records import RawDaily, StockLifecycle
from floatshare.infrastructure.storage.database import DatabaseStorage


def _save_raw(db: DatabaseStorage, df: pd.DataFrame) -> int:
    return db.save(RawDaily, df)


@pytest.fixture
def db(tmp_path: Path) -> DatabaseStorage:
    db = DatabaseStorage(db_path=tmp_path / "test.db")
    db.init_tables()
    return db


def _ohlcv_rows(code: str, dates: list[date]) -> pd.DataFrame:
    """构造 raw_daily 行数据，用于预填本地表。"""
    return pd.DataFrame(
        {
            "code": code,
            "trade_date": pd.to_datetime(dates),
            "open": 10.0,
            "high": 10.5,
            "low": 9.5,
            "close": 10.2,
            "volume": 1000.0,
        }
    )


# ============================================================================
# Bug 1: 历史回填
# ============================================================================


class TestMissingRangesBackfill:
    def test_empty_local_returns_full_window(self, db: DatabaseStorage) -> None:
        """本地无数据 → 拉用户给的整个窗口。"""
        ranges = _missing_ranges(db, "raw_daily", "X", date(2020, 1, 1), date(2025, 12, 31))
        assert ranges == [(date(2020, 1, 1), date(2025, 12, 31))]

    def test_backfill_history_when_local_starts_late(self, db: DatabaseStorage) -> None:
        """本地最早是 2022，用户要 2020 起 → 应回填 [2020, 2021-12-31]。"""
        _save_raw(db, _ohlcv_rows("X", [date(2022, 1, 1), date(2024, 6, 1)]))

        ranges = _missing_ranges(db, "raw_daily", "X", date(2020, 1, 1), date(2024, 6, 1))
        assert (date(2020, 1, 1), date(2021, 12, 31)) in ranges

    def test_backfill_and_forward_both_when_local_in_middle(self, db: DatabaseStorage) -> None:
        """本地 [2022, 2023], 用户要 [2020, 2024] → 两段：[2020, 2021] + [2024-1-1, 2024]。"""
        _save_raw(db, _ohlcv_rows("X", [date(2022, 1, 1), date(2023, 12, 31)]))

        ranges = _missing_ranges(db, "raw_daily", "X", date(2020, 1, 1), date(2024, 12, 31))
        assert len(ranges) == 2
        assert (date(2020, 1, 1), date(2021, 12, 31)) in ranges
        assert (date(2024, 1, 1), date(2024, 12, 31)) in ranges

    def test_no_backfill_when_local_min_earlier_than_requested(self, db: DatabaseStorage) -> None:
        """本地 [2018, 2024], 用户要 [2020, 2025] → 只前向追加。"""
        _save_raw(db, _ohlcv_rows("X", [date(2018, 1, 1), date(2024, 12, 31)]))

        today = date.today()
        # 用今天作为 end，避免未来跳过逻辑干扰
        ranges = _missing_ranges(db, "raw_daily", "X", date(2020, 1, 1), today)
        assert all(start is not None and start >= date(2025, 1, 1) for start, _ in ranges)


# ============================================================================
# Bug 2: 跳过未来日期
# ============================================================================


class TestMissingRangesSkipFuture:
    def test_skip_when_next_day_in_future(self, db: DatabaseStorage) -> None:
        """本地有今天的数据，下一个要拉的日期已是明天 → 不返回前向区间。"""
        today = date.today()
        _save_raw(db, _ohlcv_rows("X", [today]))

        ranges = _missing_ranges(db, "raw_daily", "X", None, today + timedelta(days=30))
        assert ranges == []

    def test_no_call_when_local_already_today(self, db: DatabaseStorage) -> None:
        """本地最新就是今天 → up-to-date，无任何区间。"""
        today = date.today()
        _save_raw(db, _ohlcv_rows("X", [today]))

        fetch_calls: list[tuple] = []

        def fake_fetch(code, start, end):
            fetch_calls.append((code, start, end))
            return pd.DataFrame()

        _sync_daily_table(
            db,
            "X",
            None,
            today,
            force=False,
            table="raw_daily",
            fetch=fake_fetch,
            save=lambda df: db.save(RawDaily, df),
        )
        assert fetch_calls == []  # 没发任何请求


# ============================================================================
# DatabaseStorage.date_range — 新加方法的覆盖
# ============================================================================


class TestDateRange:
    def test_empty_table_returns_none_pair(self, db: DatabaseStorage) -> None:
        assert db.date_range("raw_daily", "X") == (None, None)

    def test_returns_min_max_for_code(self, db: DatabaseStorage) -> None:
        _save_raw(
            db,
            _ohlcv_rows(
                "X",
                [
                    date(2020, 1, 1),
                    date(2022, 6, 15),
                    date(2024, 12, 31),
                ],
            ),
        )
        lo, hi = db.date_range("raw_daily", "X")
        assert lo == date(2020, 1, 1)
        assert hi == date(2024, 12, 31)

    def test_filters_by_code(self, db: DatabaseStorage) -> None:
        _save_raw(db, _ohlcv_rows("X", [date(2020, 1, 1)]))
        _save_raw(db, _ohlcv_rows("Y", [date(2024, 12, 31)]))
        assert db.date_range("raw_daily", "X") == (date(2020, 1, 1), date(2020, 1, 1))
        assert db.date_range("raw_daily", "Y") == (date(2024, 12, 31), date(2024, 12, 31))


# ============================================================================
# Lifecycle: _effective_window 收窄请求窗口
# ============================================================================


def _save_lifecycle(
    db: DatabaseStorage,
    code: str,
    list_date: str,
    delist_date: str | None = None,
    status: str = "L",
) -> None:
    df = pd.DataFrame(
        [
            {
                "code": code,
                "name": "TEST",
                "list_date": list_date,
                "delist_date": delist_date,
                "list_status": status,
                "market": "主板",
                "industry": "测试",
            }
        ]
    )
    db.save(StockLifecycle, df)


class TestEffectiveWindow:
    def test_no_lifecycle_returns_window_unchanged(self, db: DatabaseStorage) -> None:
        """lifecycle 表无该 code → 不收窄。"""
        result = _effective_window(db, "UNKNOWN", date(2020, 1, 1), date(2025, 12, 31))
        assert result == (date(2020, 1, 1), date(2025, 12, 31))

    def test_narrows_start_to_list_date(self, db: DatabaseStorage) -> None:
        """请求 2010 起，但股票 2015 上市 → start 收窄到 2015。"""
        _save_lifecycle(db, "X", "2015-06-01")
        result = _effective_window(db, "X", date(2010, 1, 1), date(2025, 12, 31))
        assert result == (date(2015, 6, 1), date(2025, 12, 31))

    def test_narrows_end_to_delist_date(self, db: DatabaseStorage) -> None:
        """请求到 2025，但股票 2020 退市 → end 收窄到 2020。"""
        _save_lifecycle(db, "X", "2010-01-01", "2020-06-30", status="D")
        result = _effective_window(db, "X", date(2015, 1, 1), date(2025, 12, 31))
        assert result == (date(2015, 1, 1), date(2020, 6, 30))

    def test_skip_when_request_after_delist(self, db: DatabaseStorage) -> None:
        """请求 2022 起，但 2020 退市 → 返回 None（应跳过）。"""
        _save_lifecycle(db, "X", "2010-01-01", "2020-06-30", status="D")
        result = _effective_window(db, "X", date(2022, 1, 1), date(2025, 12, 31))
        assert result is None

    def test_skip_when_request_before_listing(self, db: DatabaseStorage) -> None:
        """请求到 2010，但 2015 才上市 → 窗口无效，返回 None。"""
        _save_lifecycle(db, "X", "2015-01-01")
        result = _effective_window(db, "X", date(2005, 1, 1), date(2010, 12, 31))
        assert result is None

    def test_none_start_uses_list_date(self, db: DatabaseStorage) -> None:
        """请求 start=None → 用 list_date。"""
        _save_lifecycle(db, "X", "2015-06-01")
        result = _effective_window(db, "X", None, date(2025, 12, 31))
        assert result == (date(2015, 6, 1), date(2025, 12, 31))

    def test_none_end_uses_delist_date_if_delisted(self, db: DatabaseStorage) -> None:
        """请求 end=None 但已退市 → 用 delist_date。"""
        _save_lifecycle(db, "X", "2010-01-01", "2020-06-30", status="D")
        result = _effective_window(db, "X", date(2015, 1, 1), None)
        assert result == (date(2015, 1, 1), date(2020, 6, 30))

    def test_none_end_kept_if_listed(self, db: DatabaseStorage) -> None:
        """请求 end=None 且仍在市 → end 保持 None。"""
        _save_lifecycle(db, "X", "2010-01-01")
        result = _effective_window(db, "X", date(2015, 1, 1), None)
        assert result == (date(2015, 1, 1), None)


class TestSyncDailyTableWithLifecycle:
    def test_skip_fetch_when_outside_lifecycle(self, db: DatabaseStorage) -> None:
        """股票 2020 退市，请求 2022 后 → 不发任何 API 请求。"""
        _save_lifecycle(db, "X", "2010-01-01", "2020-06-30", status="D")
        fetch_calls: list = []

        def fake_fetch(code, start, end):
            fetch_calls.append((code, start, end))
            return pd.DataFrame()

        _sync_daily_table(
            db,
            "X",
            date(2022, 1, 1),
            date(2025, 12, 31),
            force=False,
            table="raw_daily",
            fetch=fake_fetch,
            save=lambda df: db.save(RawDaily, df),
        )
        assert fetch_calls == []

    def test_force_bypasses_lifecycle_check(self, db: DatabaseStorage) -> None:
        """--force 应忽略 lifecycle 收窄。"""
        _save_lifecycle(db, "X", "2010-01-01", "2020-06-30", status="D")
        fetch_calls: list = []

        def fake_fetch(code, start, end):
            fetch_calls.append((code, start, end))
            return pd.DataFrame()

        _sync_daily_table(
            db,
            "X",
            date(2022, 1, 1),
            date(2025, 12, 31),
            force=True,
            table="raw_daily",
            fetch=fake_fetch,
            save=lambda df: db.save(RawDaily, df),
        )
        assert len(fetch_calls) == 1


class TestLifecycleStorage:
    def test_save_and_load(self, db: DatabaseStorage) -> None:
        df = pd.DataFrame(
            [
                {
                    "code": "A",
                    "name": "甲",
                    "list_date": "2020-01-01",
                    "delist_date": None,
                    "list_status": "L",
                    "market": "主板",
                    "industry": "金融",
                },
                {
                    "code": "B",
                    "name": "乙",
                    "list_date": "2010-01-01",
                    "delist_date": "2022-06-30",
                    "list_status": "D",
                    "market": "主板",
                    "industry": "工业",
                },
            ]
        )
        n = db.save(StockLifecycle, df)
        assert n == 2

        listed = db.load_lifecycle(list_status="L")
        assert len(listed) == 1
        assert listed.iloc[0]["code"] == "A"

        delisted = db.load_lifecycle(list_status="D")
        assert len(delisted) == 1
        assert delisted.iloc[0]["delist_date"] == "2022-06-30"

    def test_get_returns_dataclass_or_none(self, db: DatabaseStorage) -> None:
        assert db.get_lifecycle("MISSING") is None

        _save_lifecycle(db, "X", "2020-01-01")
        lc = db.get_lifecycle("X")
        assert lc is not None
        assert lc.list_date == "2020-01-01"
        assert lc.list_status == "L"
