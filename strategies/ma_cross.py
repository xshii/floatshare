"""均线交叉策略 — backtrader 版本。"""

from __future__ import annotations

import backtrader as bt

from floatshare.strategy import register


@register("ma_cross")
class MACrossStrategy(bt.Strategy):
    """短均线上穿长均线买入，下穿卖出。"""

    name = "MACross"
    description = "均线交叉策略"

    params = (
        ("short_period", 5),
        ("long_period", 20),
        ("position_pct", 0.9),
    )

    def __init__(self) -> None:
        self.ma_short = {
            d._name: bt.indicators.SMA(d.close, period=self.p.short_period) for d in self.datas
        }
        self.ma_long = {
            d._name: bt.indicators.SMA(d.close, period=self.p.long_period) for d in self.datas
        }
        self.crossover = {
            d._name: bt.indicators.CrossOver(self.ma_short[d._name], self.ma_long[d._name])
            for d in self.datas
        }

    def next(self) -> None:
        cash = self.broker.getcash()
        for d in self.datas:
            pos = self.getposition(d).size
            cross = self.crossover[d._name][0]
            if cross > 0 and pos == 0:
                target_value = self.broker.getvalue() * self.p.position_pct / len(self.datas)
                size = int(target_value / d.close[0] / 100) * 100
                if size > 0 and size * d.close[0] <= cash:
                    self.buy(data=d, size=size)
            elif cross < 0 and pos > 0:
                self.close(data=d)
