"""Domain 层 — 枚举和 TradingConfig 单元测试。"""

from __future__ import annotations

from floatshare.domain import AdjustType, Direction, TradingConfig


class TestEnums:
    def test_adjust_type_str_compatible(self):
        assert AdjustType.QFQ == "qfq"
        assert AdjustType.HFQ.value == "hfq"

    def test_direction_compare(self):
        assert Direction.BUY != Direction.SELL


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
