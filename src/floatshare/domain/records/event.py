"""事件类 — 分红、龙虎榜。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class Dividend:
    """分红送股明细 (Tushare dividend，按公告事件)。"""

    TABLE: ClassVar[str] = "dividend"
    PK: ClassVar[tuple[str, ...]] = ("code", "end_date", "div_proc")

    code: str
    end_date: str
    div_proc: str  # 预案 / 决案 / 实施
    ann_date: str | None = None
    stk_div: float | None = None  # 每 10 股送红股
    stk_bo_rate: float | None = None
    stk_co_rate: float | None = None
    cash_div: float | None = None
    cash_div_tax: float | None = None
    record_date: str | None = None
    ex_date: str | None = None
    pay_date: str | None = None
    div_listdate: str | None = None
    imp_ann_date: str | None = None
    base_date: str | None = None
    base_share: float | None = None


@dataclass(frozen=True, slots=True)
class TopList:
    """龙虎榜每日个股 (Tushare top_list)。"""

    TABLE: ClassVar[str] = "top_list"
    PK: ClassVar[tuple[str, ...]] = ("trade_date", "code", "reason")

    trade_date: str
    code: str
    reason: str
    name: str | None = None
    close: float | None = None
    pct_change: float | None = None
    turnover_rate: float | None = None
    amount: float | None = None
    l_sell: float | None = None
    l_buy: float | None = None
    l_amount: float | None = None
    net_amount: float | None = None
    net_rate: float | None = None
    amount_rate: float | None = None
    float_values: float | None = None


@dataclass(frozen=True, slots=True)
class TopInst:
    """龙虎榜机构席位明细 (Tushare top_inst)。"""

    TABLE: ClassVar[str] = "top_inst"
    PK: ClassVar[tuple[str, ...]] = ("trade_date", "code", "exalter", "side", "reason")

    trade_date: str
    code: str
    exalter: str
    side: str  # '0' 买 / '1' 卖
    reason: str
    buy: float | None = None
    buy_rate: float | None = None
    sell: float | None = None
    sell_rate: float | None = None
    net_buy: float | None = None
