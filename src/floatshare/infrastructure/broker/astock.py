"""A 股佣金模型 — 修复 backtrader 默认模型不区分买卖印花税的问题。

正确的 A 股费用：
- 双边：佣金（券商，万 3 左右，最低 5 元）
- 双边：过户费（万 0.1）
- 卖出单边：印花税（万 5，2023 减半后）
"""

from __future__ import annotations

import backtrader as bt


class AStockCommission(bt.CommInfoBase):
    """符合 A 股规则的佣金信息器。"""

    params = (
        ("commission", 0.0003),
        ("stamp_duty", 0.0005),
        ("transfer_fee", 0.00001),
        ("min_commission", 5.0),
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_PERC),
        # 关键：percabs=True 让 commission 直接当作绝对小数解释，
        # 否则 backtrader 会把 0.0003 当成"百分之 0.0003" = 0.000003。
        ("percabs", True),
    )

    def _getcommission(self, size: float, price: float, pseudoexec: bool) -> float:
        amount = abs(size) * price
        commission = max(amount * self.p.commission, self.p.min_commission)
        transfer = amount * self.p.transfer_fee
        # backtrader 中买入 size > 0，卖出 size < 0
        stamp = amount * self.p.stamp_duty if size < 0 else 0.0
        return commission + transfer + stamp
