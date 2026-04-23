"""Dual Thrust 突破策略 — backtrader 版本。"""

from __future__ import annotations

import backtrader as bt

from floatshare import register


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

    @classmethod
    def search_space(cls, trial):  # type: ignore[no-untyped-def]
        """optuna 搜索空间 (供 walk_forward_optimize 用)。"""
        return {
            "lookback": trial.suggest_int("lookback", 2, 20),
            "k1": trial.suggest_float("k1", 0.1, 1.0),
            "k2": trial.suggest_float("k2", 0.1, 1.0),
            "position_pct": trial.suggest_float("position_pct", 0.3, 1.0),
        }

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
