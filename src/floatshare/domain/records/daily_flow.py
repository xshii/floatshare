"""日频 · 资金流类 — 个股大单、沪深港通、融资融券。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from floatshare.domain.records._meta import FieldSpec


@dataclass(frozen=True, slots=True)
class Moneyflow:
    """个股资金流向 (Tushare moneyflow，仅保留 amount，省 50% 存储)。"""

    TABLE: ClassVar[str] = "moneyflow"
    PK: ClassVar[tuple[str, ...]] = ("code", "trade_date")
    FIELDS: ClassVar[dict[str, FieldSpec]] = {
        "net_mf_amount": FieldSpec("主力净流入", primary=True, unit="万元"),
        "buy_elg_amount": FieldSpec("特大单买入", unit="万元"),
        "sell_elg_amount": FieldSpec("特大单卖出", unit="万元"),
        "buy_lg_amount": FieldSpec("大单买入", unit="万元"),
        "sell_lg_amount": FieldSpec("大单卖出", unit="万元"),
    }

    code: str
    trade_date: str
    buy_sm_amount: float | None = None
    sell_sm_amount: float | None = None
    buy_md_amount: float | None = None
    sell_md_amount: float | None = None
    buy_lg_amount: float | None = None
    sell_lg_amount: float | None = None
    buy_elg_amount: float | None = None
    sell_elg_amount: float | None = None
    net_mf_amount: float | None = None


@dataclass(frozen=True, slots=True)
class MoneyflowHsgt:
    """沪深港通北向/南向资金 (市场级别) — Tushare moneyflow_hsgt。"""

    TABLE: ClassVar[str] = "moneyflow_hsgt"
    PK: ClassVar[tuple[str, ...]] = ("trade_date",)
    FIELDS: ClassVar[dict[str, FieldSpec]] = {
        "north_money": FieldSpec("北向资金", primary=True, unit="百万元"),
        "south_money": FieldSpec("南向资金", primary=True, unit="百万元"),
        "hgt": FieldSpec("沪股通", unit="百万元"),
        "sgt": FieldSpec("深股通", unit="百万元"),
    }

    trade_date: str
    hgt: float | None = None
    sgt: float | None = None
    north_money: float | None = None
    south_money: float | None = None
    ggt_ss: float | None = None
    ggt_sz: float | None = None


@dataclass(frozen=True, slots=True)
class MarginDetail:
    """个股融资融券明细 (Tushare margin_detail)。"""

    TABLE: ClassVar[str] = "margin_detail"
    PK: ClassVar[tuple[str, ...]] = ("code", "trade_date")
    FIELDS: ClassVar[dict[str, FieldSpec]] = {
        "rzrqye": FieldSpec("两融余额合计", primary=True, unit="元"),
        "rzye": FieldSpec("融资余额", primary=True, unit="元"),
        "rqye": FieldSpec("融券余额", unit="元"),
        "rzmre": FieldSpec("融资买入额", unit="元"),
    }

    code: str
    trade_date: str
    rzye: float | None = None
    rqye: float | None = None
    rzmre: float | None = None
    rqyl: float | None = None
    rzche: float | None = None
    rqchl: float | None = None
    rqmcl: float | None = None
    rzrqye: float | None = None
