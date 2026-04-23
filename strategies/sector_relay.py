"""板块接力策略 — 强势行业的"跟随者"优先买入。

逻辑:
  每 N 个交易日:
    1. 算 31 个 SW L1 行业过去 lookback 日累计涨幅
    2. 取 top-K 强势板块 (红旗手)
    3. 查 LEAD_LAG_MAP (从 EDA 沉淀的) → 候选跟随者
    4. 排除已涨过的 (避免追高)
    5. 等权买入剩余 → 持有 N 天 → 再选

约束:
  - T+1 自动 (backtrader 默认 next-bar fill)
  - 整手 100 股
  - 留 5% 现金缓冲

注意:
  LEAD_LAG_MAP 是 2018-2024 全样本算的, in-sample 回测会过拟合。
  严肃做要按时间切 train / test。
"""

from __future__ import annotations

import json
from pathlib import Path

import backtrader as bt

from floatshare import register

# SW L1 代码 ↔ 中文名 (回测时数据 feed 用代码, LEAD_LAG_MAP 用中文名)
SW_L1: dict[str, str] = {
    "801010.SI": "农林牧渔",
    "801030.SI": "基础化工",
    "801040.SI": "钢铁",
    "801050.SI": "有色金属",
    "801080.SI": "电子",
    "801110.SI": "家用电器",
    "801120.SI": "食品饮料",
    "801130.SI": "纺织服饰",
    "801140.SI": "轻工制造",
    "801150.SI": "医药生物",
    "801160.SI": "公用事业",
    "801170.SI": "交通运输",
    "801180.SI": "房地产",
    "801200.SI": "商贸零售",
    "801210.SI": "社会服务",
    "801230.SI": "综合",
    "801710.SI": "建筑材料",
    "801720.SI": "建筑装饰",
    "801730.SI": "电力设备",
    "801740.SI": "国防军工",
    "801750.SI": "计算机",
    "801760.SI": "传媒",
    "801770.SI": "通信",
    "801780.SI": "银行",
    "801790.SI": "非银金融",
    "801880.SI": "汽车",
    "801890.SI": "机械设备",
    "801950.SI": "煤炭",
    "801960.SI": "石油石化",
    "801970.SI": "环保",
    "801980.SI": "美容护理",
}
NAME_TO_CODE: dict[str, str] = {v: k for k, v in SW_L1.items()}

LEAD_LAG_PATH = Path(__file__).parent.parent / "notebooks/output/lead_lag_map_lag5.json"


def _load_lead_lag() -> dict[str, list[str]]:
    """从 EDA 输出加载 lookup 表 — 文件不存在时返回空 dict (策略退化为纯动量)。"""
    if not LEAD_LAG_PATH.exists():
        return {}
    return json.loads(LEAD_LAG_PATH.read_text())


@register("sector_relay")
class SectorRelay(bt.Strategy):
    """板块接力 — 见模块 docstring。"""

    name = "SectorRelay"
    params = (
        ("lookback", 5),  # 算强势板块的回看窗口
        ("rebal_period", 5),  # 多少天重选一次
        ("n_leaders", 3),  # 取 top-N 红旗手
        ("n_holdings", 3),  # 最多持仓数
        ("cash_buffer", 0.05),  # 留 5% 现金不投
    )

    @classmethod
    def search_space(cls, trial):  # type: ignore[no-untyped-def]
        """optuna 搜索空间 — 给 walk_forward_optimize 用。"""
        return {
            "lookback": trial.suggest_int("lookback", 3, 20),
            "rebal_period": trial.suggest_int("rebal_period", 5, 60),
            "n_leaders": trial.suggest_int("n_leaders", 2, 6),
            "n_holdings": trial.suggest_int("n_holdings", 2, 6),
            "cash_buffer": trial.suggest_float("cash_buffer", 0.0, 0.15),
        }

    def __init__(self) -> None:
        self._lookup = _load_lead_lag()
        self._bar = 0

    def next(self) -> None:
        self._bar += 1
        # 第 1 个 bar 不操作; 之后每 rebal_period 调仓一次
        if self._bar < self.p.lookback + 1:
            return
        if self._bar % self.p.rebal_period != 0:
            return

        # 1. 计算过去 lookback 日各板块涨幅 {code: ret}
        returns: dict[str, float] = {}
        for d in self.datas:
            if len(d) <= self.p.lookback:
                continue
            past = d.close[-self.p.lookback]
            now = d.close[0]
            if past and past > 0:
                returns[d._name] = (now - past) / past
        if not returns:
            return

        # 2. top-K 红旗手 (按 5 日涨幅)
        ranked = sorted(returns, key=lambda c: returns[c], reverse=True)
        leader_codes = ranked[: self.p.n_leaders]

        # 3. 红旗手 → 跟随者 (查 lookup, 中文名 → 代码)
        follower_codes: list[str] = []
        seen: set[str] = set(leader_codes)  # 排除红旗手本身
        for code in leader_codes:
            name = SW_L1.get(code)
            if not name:
                continue
            for follower_name in self._lookup.get(name, []):
                follower_code = NAME_TO_CODE.get(follower_name)
                if follower_code and follower_code not in seen:
                    follower_codes.append(follower_code)
                    seen.add(follower_code)

        # 4. 候选过滤: 不要已经涨过的 (corr 5 日涨幅 > 中位数)
        if not follower_codes:
            # lookup 为空 (没跑 EDA)，退化为直接买红旗手
            target = leader_codes[: self.p.n_holdings]
        else:
            median_ret = sorted(returns.values())[len(returns) // 2]
            cold = [c for c in follower_codes if returns.get(c, 0) < median_ret]
            target = cold[: self.p.n_holdings] or follower_codes[: self.p.n_holdings]

        # 5. 卖非目标
        for d in self.datas:
            if d._name not in target and self.getposition(d).size > 0:
                self.close(data=d)

        # 6. 买目标 (等权 + 100 股一手)
        # 价格已被入口归一化为 ETF 量级 (~3-5 元/股), 100 股一手在 10w 资金下可成交
        cash_avail = self.broker.getcash() * (1 - self.p.cash_buffer)
        per_holding = cash_avail / max(len(target), 1)
        for code in target:
            d = next((x for x in self.datas if x._name == code), None)
            if d is None or self.getposition(d).size > 0:
                continue
            price = d.close[0]
            if price <= 0:
                continue
            size = int(per_holding / price / 100) * 100  # A 股 100 股一手
            if size > 0:
                self.buy(data=d, size=size)
