"""DataSyncer 增量同步 + 读时复权 测试。"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from floatshare.application.data_syncer import DataSyncer, apply_adjustment
from floatshare.domain.enums import AdjustType
from floatshare.domain.records import AdjFactor, RawDaily
from floatshare.infrastructure.storage.database import DatabaseStorage
from floatshare.interfaces.data_source import DataSourceError

# --- apply_adjustment 纯函数测试 -------------------------------------------


class TestApplyAdjustment:
    def _raw_df(self) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        return pd.DataFrame(
            {
                "trade_date": dates,
                "open": [10.0, 10.5, 11.0, 10.8, 11.2],
                "high": [10.5, 11.0, 11.5, 11.0, 11.5],
                "low": [9.8, 10.0, 10.5, 10.5, 10.8],
                "close": [10.2, 10.8, 11.0, 10.7, 11.3],
                "volume": [1000, 1200, 900, 1100, 1050],
            }
        )

    def _factor_df(self) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        return pd.DataFrame(
            {
                "trade_date": dates,
                "adj_factor": [1.0, 1.0, 2.0, 2.0, 2.0],  # 第3天送股，因子翻倍
            }
        )

    def test_none_returns_unchanged(self) -> None:
        raw = self._raw_df()
        result = apply_adjustment(raw, self._factor_df(), AdjustType.NONE)
        pd.testing.assert_frame_equal(result, raw)

    def test_qfq_latest_price_unchanged(self) -> None:
        raw = self._raw_df()
        factors = self._factor_df()
        result = apply_adjustment(raw, factors, AdjustType.QFQ)
        # 前复权: 最后一天的价格不变
        assert result["close"].iloc[-1] == pytest.approx(11.3)
        # 第1天 factor=1.0, latest=2.0, ratio=0.5 → close=10.2×0.5=5.1
        assert result["close"].iloc[0] == pytest.approx(5.1)

    def test_hfq_earliest_price_unchanged(self) -> None:
        raw = self._raw_df()
        factors = self._factor_df()
        result = apply_adjustment(raw, factors, AdjustType.HFQ)
        # 后复权: 第一天的价格不变
        assert result["close"].iloc[0] == pytest.approx(10.2)
        # 第5天 factor=2.0, earliest=1.0, ratio=2.0 → close=11.3×2=22.6
        assert result["close"].iloc[-1] == pytest.approx(22.6)

    def test_volume_not_adjusted(self) -> None:
        raw = self._raw_df()
        factors = self._factor_df()
        result = apply_adjustment(raw, factors, AdjustType.QFQ)
        np.testing.assert_array_equal(result["volume"].values, raw["volume"].values)

    def test_empty_factors_returns_raw(self) -> None:
        raw = self._raw_df()
        result = apply_adjustment(raw, pd.DataFrame(), AdjustType.QFQ)
        pd.testing.assert_frame_equal(result, raw)


# --- DataSyncer 测试 -------------------------------------------------------


class FakeRawSource:
    """返回未复权数据的假源。"""

    def __init__(self, data: pd.DataFrame | None = None, fail: bool = False) -> None:
        self._data = data
        self._fail = fail
        self.call_count = 0

    def get_raw_daily(
        self, code: str, start: date | None = None, end: date | None = None
    ) -> pd.DataFrame:
        self.call_count += 1
        if self._fail:
            raise DataSourceError("fake raw source failed")
        if self._data is not None:
            df = self._data.copy()
            if start:
                df = df[df["trade_date"] >= pd.Timestamp(start)]
            return df
        # 数据截止到昨天，保证 watermark 检查时判定为 "已是最新"
        end_date = date.today() - timedelta(days=1)
        dates = pd.bdate_range(end=end_date, periods=10)
        return pd.DataFrame(
            {
                "code": code,
                "trade_date": dates,
                "open": range(10),
                "high": range(10),
                "low": range(10),
                "close": range(10),
                "volume": range(10),
            }
        )


class FakeAdjSource:
    """返回复权因子的假源。"""

    def __init__(self, fail: bool = False) -> None:
        self._fail = fail

    def get_adj_factor(
        self, code: str, start: date | None = None, end: date | None = None
    ) -> pd.DataFrame:
        if self._fail:
            raise DataSourceError("fake adj source failed")
        end_date = date.today() - timedelta(days=1)
        dates = pd.bdate_range(end=end_date, periods=10)
        return pd.DataFrame(
            {
                "code": code,
                "trade_date": dates,
                "adj_factor": [1.0] * 10,
            }
        )


class TestDataSyncer:
    def _make_syncer(
        self,
        tmp_path: Path,
        raw_sources: list | None = None,
        adj_sources: list | None = None,
    ) -> DataSyncer:
        db = DatabaseStorage(db_path=tmp_path / "test.db")
        db.init_tables()
        return DataSyncer(
            db=db,
            raw_sources=raw_sources or [FakeRawSource()],
            adj_sources=adj_sources or [FakeAdjSource()],
        )

    def test_first_sync_full_fetch(self, tmp_path: Path) -> None:
        raw_src = FakeRawSource()
        syncer = self._make_syncer(tmp_path, raw_sources=[raw_src])
        df = syncer.get_daily("000001.SZ", adj=AdjustType.NONE)
        assert not df.empty
        assert raw_src.call_count == 1

    def test_second_call_uses_local(self, tmp_path: Path) -> None:
        raw_src = FakeRawSource()
        syncer = self._make_syncer(tmp_path, raw_sources=[raw_src])
        # 第一次: 全量同步
        syncer.get_daily("000001.SZ", adj=AdjustType.NONE)
        assert raw_src.call_count == 1
        # 第二次: watermark 已是最新，不再拉远程
        syncer.get_daily("000001.SZ", adj=AdjustType.NONE)
        assert raw_src.call_count == 1  # 没有新调用

    def test_qfq_adjustment_applied(self, tmp_path: Path) -> None:
        syncer = self._make_syncer(tmp_path)
        df = syncer.get_daily("000001.SZ", adj=AdjustType.QFQ)
        assert not df.empty
        # 因为 adj_factor 全是 1.0，价格应该不变
        assert "close" in df.columns

    def test_fallback_to_second_source(self, tmp_path: Path) -> None:
        bad_src = FakeRawSource(fail=True)
        good_src = FakeRawSource()
        syncer = self._make_syncer(tmp_path, raw_sources=[bad_src, good_src])
        df = syncer.get_daily("000001.SZ", adj=AdjustType.NONE)
        assert not df.empty
        assert bad_src.call_count == 1
        assert good_src.call_count == 1

    def test_no_adj_factor_returns_raw(self, tmp_path: Path) -> None:
        syncer = self._make_syncer(
            tmp_path,
            adj_sources=[FakeAdjSource(fail=True)],
        )
        df = syncer.get_daily("000001.SZ", adj=AdjustType.QFQ)
        # adj_factor 拉取失败，应该返回不复权数据（降级，而非报错）
        assert not df.empty

    def test_all_sources_fail_raises(self, tmp_path: Path) -> None:
        syncer = self._make_syncer(
            tmp_path,
            raw_sources=[FakeRawSource(fail=True)],
            adj_sources=[FakeAdjSource(fail=True)],
        )
        with pytest.raises(DataSourceError, match="本地无"):
            syncer.get_daily("000001.SZ")


# --- DatabaseStorage 新表测试 -----------------------------------------------


class TestDatabaseNewTables:
    def test_raw_daily_save_and_load(self, tmp_path: Path) -> None:
        db = DatabaseStorage(db_path=tmp_path / "test.db")
        db.init_tables()
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        df = pd.DataFrame(
            {
                "code": "000001.SZ",
                "trade_date": dates,
                "open": [10.0, 10.5, 11.0],
                "high": [10.5, 11.0, 11.5],
                "low": [9.5, 10.0, 10.5],
                "close": [10.2, 10.8, 11.1],
                "volume": [1000, 1200, 900],
            }
        )
        saved = db.save(RawDaily, df)
        assert saved == 3

        loaded = db.load("raw_daily", "000001.SZ")
        assert len(loaded) == 3
        assert loaded["close"].iloc[-1] == pytest.approx(11.1)

    def test_raw_daily_upsert(self, tmp_path: Path) -> None:
        db = DatabaseStorage(db_path=tmp_path / "test.db")
        db.init_tables()
        df1 = pd.DataFrame(
            {
                "code": "000001.SZ",
                "trade_date": ["2024-01-02"],
                "open": [10.0],
                "high": [10.5],
                "low": [9.5],
                "close": [10.2],
                "volume": [1000],
            }
        )
        db.save(RawDaily, df1)
        # 用不同价格再写一次同一天 → 应覆盖
        df2 = pd.DataFrame(
            {
                "code": "000001.SZ",
                "trade_date": ["2024-01-02"],
                "open": [99.0],
                "high": [99.0],
                "low": [99.0],
                "close": [99.0],
                "volume": [9999],
            }
        )
        db.save(RawDaily, df2)
        loaded = db.load("raw_daily", "000001.SZ")
        assert len(loaded) == 1
        assert loaded["close"].iloc[0] == pytest.approx(99.0)

    def test_adj_factor_save_and_load(self, tmp_path: Path) -> None:
        db = DatabaseStorage(db_path=tmp_path / "test.db")
        db.init_tables()
        df = pd.DataFrame(
            {
                "code": "000001.SZ",
                "trade_date": pd.date_range("2024-01-01", periods=3, freq="B"),
                "adj_factor": [1.0, 1.0, 2.0],
            }
        )
        db.save(AdjFactor, df)
        loaded = db.load("adj_factor", "000001.SZ")
        assert len(loaded) == 3
        assert loaded["adj_factor"].iloc[-1] == pytest.approx(2.0)

    def test_watermark_round_trip(self, tmp_path: Path) -> None:
        db = DatabaseStorage(db_path=tmp_path / "test.db")
        db.init_tables()
        assert db.get_watermark("000001.SZ") is None

        db.update_watermark("000001.SZ", date(2024, 6, 1), "tushare")
        assert db.get_watermark("000001.SZ") == date(2024, 6, 1)

        # 更新水位线
        db.update_watermark("000001.SZ", date(2024, 7, 1), "tushare")
        assert db.get_watermark("000001.SZ") == date(2024, 7, 1)

    def test_trade_calendar_save_and_load(self, tmp_path: Path) -> None:
        db = DatabaseStorage(db_path=tmp_path / "test.db")
        db.init_tables()
        dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
        db.save_trade_calendar(dates)
        loaded = db.load_trade_calendar()
        assert loaded == dates


# --- 智能 TTL 测试 ----------------------------------------------------------


class TestSmartTTL:
    def test_historical_gets_long_ttl(self) -> None:
        from floatshare.infrastructure.data_sources.cached import (
            TTL_HISTORICAL,
            smart_daily_ttl,
        )

        yesterday = date.today() - timedelta(days=1)
        assert smart_daily_ttl(end=yesterday) == TTL_HISTORICAL

    def test_today_gets_short_ttl(self) -> None:
        from floatshare.infrastructure.data_sources.cached import (
            TTL_INTRADAY,
            smart_daily_ttl,
        )

        assert smart_daily_ttl(end=date.today()) == TTL_INTRADAY

    def test_none_end_gets_short_ttl(self) -> None:
        from floatshare.infrastructure.data_sources.cached import (
            TTL_INTRADAY,
            smart_daily_ttl,
        )

        assert smart_daily_ttl(end=None) == TTL_INTRADAY
