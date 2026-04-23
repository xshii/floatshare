"""Domain 层 — 枚举和 TradingConfig 单元测试。"""

from __future__ import annotations

from floatshare.domain import (
    AdjustType,
    DataKind,
    DataSourceKind,
    Direction,
    TradingConfig,
)


class TestEnums:
    def test_adjust_type_str_compatible(self):
        assert AdjustType.QFQ == "qfq"
        assert AdjustType.HFQ.value == "hfq"

    def test_direction_compare(self):
        assert Direction.BUY != Direction.SELL


class TestDataKindNested:
    def test_nested_access(self):
        assert DataKind.DAILY.RAW_DAILY == "raw_daily"
        assert DataKind.FUNDAMENTAL.INCOME == "income"
        assert DataKind.REFERENCE.LIFECYCLE == "lifecycle"

    def test_iter_per_group(self):
        daily = list(DataKind.DAILY)
        assert DataKind.DAILY.RAW_DAILY in daily
        assert DataKind.DAILY.ADJ_FACTOR in daily
        assert DataKind.DAILY.CHIP_PERF in daily

    def test_all_returns_every_kind(self):
        all_kinds = DataKind.all()
        # 4 reference + 6 daily + 1 intraday_heavy + 6 fundamental + 6 market + 4 event = 27
        assert len(all_kinds) == 27
        assert DataKind.MARKET.INDEX_DAILY in all_kinds
        # 各分类代表都在
        assert DataKind.REFERENCE.LIFECYCLE in all_kinds
        assert DataKind.REFERENCE.INDEX_WEIGHT in all_kinds
        assert DataKind.DAILY.DAILY_BASIC in all_kinds
        assert DataKind.DAILY.MARGIN_DETAIL in all_kinds
        assert DataKind.FUNDAMENTAL.FINA_INDICATOR in all_kinds
        assert DataKind.FUNDAMENTAL.HOLDER_NUMBER in all_kinds
        assert DataKind.MARKET.MONEYFLOW_HSGT in all_kinds
        assert DataKind.MARKET.CN_CPI in all_kinds
        assert DataKind.MARKET.CN_PPI in all_kinds
        assert DataKind.MARKET.SHIBOR in all_kinds
        assert DataKind.MARKET.FX_DAILY in all_kinds
        assert DataKind.EVENT.BROKER_PICKS in all_kinds
        assert DataKind.EVENT.DIVIDEND in all_kinds
        assert DataKind.EVENT.TOP_LIST in all_kinds
        assert DataKind.EVENT.TOP_INST in all_kinds

    def test_from_value_roundtrip(self):
        assert DataKind.from_value("raw_daily") == DataKind.DAILY.RAW_DAILY
        assert DataKind.from_value("income") == DataKind.FUNDAMENTAL.INCOME


class TestDataSourceKindNested:
    def test_nested_access(self):
        assert DataSourceKind.PAID_REMOTE.TUSHARE == "tushare"
        assert DataSourceKind.FREE_REMOTE.AKSHARE == "akshare"

    def test_all_returns_5_sources(self):
        assert len(DataSourceKind.all()) == 5

    def test_from_value_roundtrip(self):
        assert DataSourceKind.from_value("tushare") == DataSourceKind.PAID_REMOTE.TUSHARE


class TestTradingConfig:
    def test_buy_no_stamp_duty(self):
        cfg = TradingConfig()
        fee = cfg.calculate_fee(amount=100_000, direction=Direction.BUY)
        # 佣金 + 过户费，无印花税
        expected = (
            max(100_000 * cfg.commission_rate, cfg.min_commission) + 100_000 * cfg.transfer_fee
        )
        assert abs(fee - expected) < 1e-6

    def test_sell_includes_stamp_duty(self):
        cfg = TradingConfig()
        fee = cfg.calculate_fee(amount=100_000, direction=Direction.SELL)
        expected = (
            max(100_000 * cfg.commission_rate, cfg.min_commission)
            + 100_000 * cfg.stamp_duty
            + 100_000 * cfg.transfer_fee
        )
        assert abs(fee - expected) < 1e-6

    def test_min_commission_floor(self):
        cfg = TradingConfig()
        # 1000 * 0.0003 = 0.3, should floor to 5.0
        fee = cfg.calculate_fee(amount=1000, direction=Direction.BUY)
        assert fee >= 5.0
