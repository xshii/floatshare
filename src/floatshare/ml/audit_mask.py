"""Per-stock trading mask 生成 — 标记当日不可参与训练/推理的股.

触发 mask=False 的三种情况:

    1. ST 股 (`stock_lifecycle.name` 含 'ST' / '*ST'):
       涨跌停 5% 而非 10%, is_limit 判定与主流逻辑不兼容.
       正常情况下 universe 选择会过滤, 但可能混入.

    2. 除权日 (`dividend.ex_date == trade_date`):
       raw_daily.close 不复权, 除权日会产生伪"大跌" ret_1d,
       污染 ma_dev / vola / limit_up_history 等.
       建议当日 mask 或用复权价重算.

    3. 停牌日 (该股当日不在 panel 里):
       pandas rolling 按行算, 停牌股 rolling 窗口跨越更多自然日,
       语义上模糊. Mask 当日避免模型看错位的 "最近 20 天".

用途:
    audit 调用 `compute_trading_mask(panel, trade_date, db_path)` →
    返回 `{code: reasons}` 字典. 调用方把 mask[code]=False 传给模型
    (ActorCritic.forward(mask=...) 或 cube.traded).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import pandas as pd

from floatshare.domain.enums import MaskReason


@dataclass(frozen=True, slots=True)
class TradingMaskReason:
    """某股某日被 mask 的原因 (可多个)."""

    code: str
    reasons: tuple[str, ...]  # ('ST', 'ex_dividend', 'suspended')


@dataclass(slots=True)
class TradingMaskReport:
    """全市场当日 mask 汇总."""

    trade_date: str
    masked: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def is_masked(self, code: str) -> bool:
        return code in self.masked

    def summary(self) -> str:
        n_st = sum(1 for rs in self.masked.values() if MaskReason.ST in rs)
        n_div = sum(1 for rs in self.masked.values() if MaskReason.EX_DIVIDEND in rs)
        n_sus = sum(1 for rs in self.masked.values() if MaskReason.SUSPENDED in rs)
        return (
            f"TradingMaskReport T={self.trade_date} masked={len(self.masked)} "
            f"(ST={n_st} ex_dividend={n_div} suspended={n_sus})"
        )


def compute_trading_mask(
    panel: pd.DataFrame,
    trade_date: str,
    db_path: str,
    universe: list[str] | None = None,
) -> TradingMaskReport:
    """计算 trade_date 当日每只股的 mask. 返回哪些股被屏蔽 + 原因.

    Args:
        panel: compute_features 的输入 panel (含 code + trade_date)
        trade_date: 目标交易日 YYYY-MM-DD
        db_path: floatshare.db 路径
        universe: 要检查的股列表, None=取 panel 里所有 code

    Returns:
        TradingMaskReport. masked[code] = ('ST',) / ('ex_dividend',) / ('suspended',) / 组合.
    """
    report = TradingMaskReport(trade_date=trade_date)
    target_codes = set(universe) if universe is not None else set(panel["code"].unique())
    if not target_codes:
        return report

    st_codes = _fetch_st_codes(db_path, target_codes)
    ex_div_codes = _fetch_ex_dividend_codes(db_path, trade_date, target_codes)
    suspended_codes = _find_suspended_codes(panel, trade_date, target_codes)

    for code in target_codes:
        reasons: list[str] = []
        if code in st_codes:
            reasons.append(MaskReason.ST)
        if code in ex_div_codes:
            reasons.append(MaskReason.EX_DIVIDEND)
        if code in suspended_codes:
            reasons.append(MaskReason.SUSPENDED)
        if reasons:
            report.masked[code] = tuple(reasons)
    return report


# --- DB / panel queries ------------------------------------------------------


def _fetch_st_codes(db_path: str, target_codes: set[str]) -> set[str]:
    """从 stock_lifecycle 查名字含 ST / *ST 的股."""
    if not target_codes:
        return set()
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("""
            SELECT code FROM stock_lifecycle
            WHERE name LIKE 'ST%' OR name LIKE '*ST%'
        """).fetchall()
    return {r[0] for r in rows if r[0] in target_codes}


def _fetch_ex_dividend_codes(
    db_path: str,
    trade_date: str,
    target_codes: set[str],
) -> set[str]:
    """查 trade_date 当日的除权股. dividend.ex_date 存 YYYY-MM-DD 或 ISO8601."""
    if not target_codes:
        return set()
    # DB 里 trade_date 有时带 'T00:00:00', 有时裸日期; 用 LIKE 兼容
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT code FROM dividend
            WHERE (ex_date = ? OR ex_date LIKE ?)
              AND div_proc IN ('实施', '已实施')
        """,
            (trade_date, f"{trade_date}T%"),
        ).fetchall()
    return {r[0] for r in rows if r[0] in target_codes}


def _find_suspended_codes(
    panel: pd.DataFrame,
    trade_date: str,
    target_codes: set[str],
) -> set[str]:
    """panel 里 trade_date 当日没出现的 code (相对 target_codes) = 停牌."""
    td = pd.Timestamp(trade_date)
    active_today = set(panel.loc[panel["trade_date"] == td, "code"].unique())
    return target_codes - active_today
