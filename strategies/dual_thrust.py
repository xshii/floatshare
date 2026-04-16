"""Dual Thrust 突破策略 — backtrader 版本。"""

from __future__ import annotations

import backtrader as bt

from floatshare.strategy import register


@register("dual_thrust")
class DualThrustStrategy(bt.Strategy):
    """基于前 N 日 HH/HC/LC/LL 计算 Range，价格突破上轨买入、下轨卖出。"""

    name = "DualThrust"
    description = "Dual Thrust 突破策略"

    params = (
        ("lookback", 4),
        ("k1", 0.5),
        ("k2", 0.5),
        ("position_pct", 0.8),
    )

    def __init__(self) -> None:
        self.hh = {
            d._name: bt.indicators.Highest(d.high, period=self.p.lookback) for d in self.datas
        }
        self.ll = {d._name: bt.indicators.Lowest(d.low, period=self.p.lookback) for d in self.datas}
        self.hc = {
            d._name: bt.indicators.Highest(d.close, period=self.p.lookback) for d in self.datas
        }
        self.lc = {
            d._name: bt.indicators.Lowest(d.close, period=self.p.lookback) for d in self.datas
        }

    def next(self) -> None:
        for d in self.datas:
            if len(d) <= self.p.lookback:
                continue
            hh = self.hh[d._name][-1]
            ll = self.ll[d._name][-1]
            hc = self.hc[d._name][-1]
            lc = self.lc[d._name][-1]
            rng = max(hh - lc, hc - ll)

            today_open = d.open[0]
            today_close = d.close[0]
            upper = today_open + self.p.k1 * rng
            lower = today_open - self.p.k2 * rng

            pos = self.getposition(d).size
            if today_close > upper and pos == 0:
                target_value = self.broker.getvalue() * self.p.position_pct / len(self.datas)
                size = int(target_value / today_close / 100) * 100
                if size > 0:
                    self.buy(data=d, size=size)
            elif today_close < lower and pos > 0:
                self.close(data=d)
