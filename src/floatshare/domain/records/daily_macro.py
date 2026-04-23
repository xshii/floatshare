"""日频 · 宏观类 — SHIBOR 利率、外汇日行情。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from floatshare.domain.records._meta import FieldSpec


@dataclass(frozen=True, slots=True)
class Shibor:
    """上海银行间同业拆借利率 (Tushare shibor，日频)。"""

    TABLE: ClassVar[str] = "shibor"
    PK: ClassVar[tuple[str, ...]] = ("date",)
    FIELDS: ClassVar[dict[str, FieldSpec]] = {
        "overnight": FieldSpec("隔夜", primary=True, unit="%"),
        "m1": FieldSpec("1 月", primary=True, unit="%"),
        "m3": FieldSpec("3 月", unit="%"),
        "y1": FieldSpec("1 年", unit="%"),
        "w1": FieldSpec("1 周", unit="%"),
    }

    date: str
    overnight: float | None = None
    w1: float | None = None
    w2: float | None = None
    m1: float | None = None
    m3: float | None = None
    m6: float | None = None
    m9: float | None = None
    y1: float | None = None


@dataclass(frozen=True, slots=True)
class FxDaily:
    """外汇日行情 (Tushare fx_daily，多币对支持)。"""

    TABLE: ClassVar[str] = "fx_daily"
    PK: ClassVar[tuple[str, ...]] = ("code", "trade_date")
    FIELDS: ClassVar[dict[str, FieldSpec]] = {
        "bid_close": FieldSpec("买价收盘", primary=True),
        "ask_close": FieldSpec("卖价收盘", primary=True),
        "bid_open": FieldSpec("买价开盘"),
        "bid_high": FieldSpec("买价最高"),
        "bid_low": FieldSpec("买价最低"),
    }

    code: str
    trade_date: str
    bid_open: float | None = None
    bid_close: float | None = None
    bid_high: float | None = None
    bid_low: float | None = None
    ask_open: float | None = None
    ask_close: float | None = None
    ask_high: float | None = None
    ask_low: float | None = None
    tick_qty: float | None = None
    exchange: str | None = None
