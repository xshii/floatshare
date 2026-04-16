"""AStockCommission 印花税逻辑测试 — 这是 P0 修复点。"""

from __future__ import annotations

from floatshare.infrastructure.broker import AStockCommission


class TestAStockCommission:
    def test_buy_no_stamp_duty(self):
        comm = AStockCommission(
            commission=0.0003,
            stamp_duty=0.0005,
            transfer_fee=0.00001,
            min_commission=5.0,
        )
        # size > 0 表示买入
        fee = comm._getcommission(size=1000, price=10.0, pseudoexec=False)
        # amount = 10000; commission = max(3.0, 5.0) = 5.0; transfer = 0.1; stamp = 0
        assert abs(fee - (5.0 + 0.1)) < 1e-6

    def test_sell_includes_stamp_duty(self):
        comm = AStockCommission(
            commission=0.0003,
            stamp_duty=0.0005,
            transfer_fee=0.00001,
            min_commission=5.0,
        )
        fee = comm._getcommission(size=-1000, price=10.0, pseudoexec=False)
        # amount = 10000; commission = 5.0; transfer = 0.1; stamp = 5.0
        assert abs(fee - (5.0 + 0.1 + 5.0)) < 1e-6

    def test_large_trade_uses_percentage_commission(self):
        comm = AStockCommission(
            commission=0.0003,
            stamp_duty=0.0005,
            transfer_fee=0.00001,
            min_commission=5.0,
        )
        fee = comm._getcommission(size=10_000, price=100.0, pseudoexec=False)
        # amount = 1_000_000; commission = 300; transfer = 10; stamp = 0
        assert abs(fee - (300.0 + 10.0)) < 1e-6
