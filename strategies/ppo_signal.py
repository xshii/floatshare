"""PPO 信号策略 — 读 ml/eval.py 输出的 signal CSV → 按权重选 top-K 持仓。

CSV 格式: (trade_date, token_id, weight)
    每日 ~K 条记录, weight 已 softmax (sum ≈ 1)。

调仓: 每 rebal_period 天读最新 signal, 卖非目标, 按权重买目标 (T+1 成交)。
"""

from __future__ import annotations

from pathlib import Path

import backtrader as bt
import pandas as pd

from floatshare import register


@register("ppo_signal")
class PPOSignalStrategy(bt.Strategy):
    """ML signal driven 组合策略 — 读 CSV, 按权重 top-K 持仓。"""

    name = "PPOSignal"
    description = "PPO 模型预测信号驱动的组合调仓策略"

    params = (
        ("signal_csv", "data/ml/signals/daily.csv"),
        ("top_k", 10),  # 取 top-K 持仓
        ("rebal_period", 5),  # 每 N 天调一次仓 (建议=PPO reward_horizon)
        ("position_pct", 0.95),  # 总仓位 (留 5% 现金)
    )

    def __init__(self) -> None:
        csv_path = Path(self.p.signal_csv)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"signal CSV 不存在: {csv_path}\n"
                f"先跑: floatshare-eval-ppo --ckpt data/ml/ckpts/phase1_best.pt --out {csv_path}"
            )
        self._signals = pd.read_csv(csv_path, parse_dates=["trade_date"])
        # 按日期索引以便 next() 内 O(1) 查
        self._sig_by_date: dict = {
            d: g[["token_id", "weight"]]
            for d, g in self._signals.groupby(self._signals["trade_date"].dt.date)
        }
        self._bar = 0
        # 名字映射 (data._name → backtrader data feed)
        self._name_to_data: dict[str, bt.AbstractDataBase] = {d._name: d for d in self.datas}

    def next(self) -> None:
        self._bar += 1
        if self._bar < self.p.rebal_period:
            return
        if (self._bar - 1) % self.p.rebal_period != 0:
            return

        cur_date = self.datetime.date(0)
        today_sig = self._sig_by_date.get(cur_date)
        if today_sig is None or today_sig.empty:
            return

        # Top-K (按 weight 排序)
        top = today_sig.nlargest(self.p.top_k, "weight")
        if top.empty:
            return
        target_codes: set[str] = set(top["token_id"].astype(str))

        # 卖非目标
        for d in self.datas:
            if d._name not in target_codes and self.getposition(d).size > 0:
                self.close(data=d)

        # 按 weight 比例买目标 (相对权重)
        weight_sum = float(top["weight"].sum())
        if weight_sum <= 0:
            return
        cash_avail = self.broker.getcash() * self.p.position_pct
        for _, row in top.iterrows():
            code = str(row["token_id"])
            d = self._name_to_data.get(code)
            if d is None or self.getposition(d).size > 0:
                continue
            target_value = cash_avail * (float(row["weight"]) / weight_sum)
            price = float(d.close[0])
            if price <= 0:
                continue
            size = int(target_value / price / 100) * 100  # A 股 100 股一手
            if size > 0:
                self.buy(data=d, size=size)
