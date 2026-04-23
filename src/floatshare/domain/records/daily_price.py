"""日频 · 价格类 — 原始 OHLCV、复权因子、每日基本面代理 (PE/PB/市值)。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from floatshare.domain.records._meta import FieldSpec


@dataclass(frozen=True, slots=True)
class RawDaily:
    """原始未复权日线 OHLCV。"""

    TABLE: ClassVar[str] = "raw_daily"
    PK: ClassVar[tuple[str, ...]] = ("code", "trade_date")
    FIELDS: ClassVar[dict[str, FieldSpec]] = {
        "close": FieldSpec("收盘", primary=True, unit="元"),
        "volume": FieldSpec("成交量", primary=True, unit="股"),
        "open": FieldSpec("开盘", unit="元"),
        "high": FieldSpec("最高", unit="元"),
        "low": FieldSpec("最低", unit="元"),
        "amount": FieldSpec("成交额", unit="元"),
        "pct_change": FieldSpec("涨跌幅", unit="%"),
        "turnover": FieldSpec("换手率", unit="%"),
    }

    code: str
    trade_date: str
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float | None = None
    amount: float | None = None
    pre_close: float | None = None
    pct_change: float | None = None
    turnover: float | None = None


@dataclass(frozen=True, slots=True)
class AdjFactor:
    """复权因子 (可追溯修正)。"""

    TABLE: ClassVar[str] = "adj_factor"
    PK: ClassVar[tuple[str, ...]] = ("code", "trade_date")
    FIELDS: ClassVar[dict[str, FieldSpec]] = {
        "adj_factor": FieldSpec("复权因子", primary=True),
    }

    code: str
    trade_date: str
    adj_factor: float | None = None


@dataclass(frozen=True, slots=True)
class IndexDaily:
    """指数日线 OHLCV — 沪深300/中证500/红利等风格代理。"""

    TABLE: ClassVar[str] = "index_daily"
    PK: ClassVar[tuple[str, ...]] = ("code", "trade_date")
    FIELDS: ClassVar[dict[str, FieldSpec]] = {
        "close": FieldSpec("收盘", primary=True, unit="点"),
        "volume": FieldSpec("成交量", primary=True),
        "open": FieldSpec("开盘", unit="点"),
        "high": FieldSpec("最高", unit="点"),
        "low": FieldSpec("最低", unit="点"),
        "amount": FieldSpec("成交额", unit="千元"),
        "pct_change": FieldSpec("涨跌幅", unit="%"),
    }

    code: str  # 指数代码 (如 000300.SH)
    trade_date: str
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float | None = None
    amount: float | None = None
    pre_close: float | None = None
    pct_change: float | None = None


@dataclass(frozen=True, slots=True)
class DailyBasic:
    """每日基本面 (Tushare daily_basic): PE/PB/PS/股息/总市值/换手率。"""

    TABLE: ClassVar[str] = "daily_basic"
    PK: ClassVar[tuple[str, ...]] = ("code", "trade_date")
    FIELDS: ClassVar[dict[str, FieldSpec]] = {
        "pe": FieldSpec("PE", primary=True),
        "pb": FieldSpec("PB", primary=True),
        "total_mv": FieldSpec("总市值", primary=True, unit="万元"),
        "turnover_rate": FieldSpec("换手率", unit="%"),
        "pe_ttm": FieldSpec("PE-TTM"),
        "ps": FieldSpec("PS"),
        "dv_ratio": FieldSpec("股息率", unit="%"),
        "circ_mv": FieldSpec("流通市值", unit="万元"),
        "close": FieldSpec("收盘", unit="元"),
    }

    code: str
    trade_date: str
    close: float | None = None
    turnover_rate: float | None = None
    turnover_rate_f: float | None = None
    volume_ratio: float | None = None
    pe: float | None = None
    pe_ttm: float | None = None
    pb: float | None = None
    ps: float | None = None
    ps_ttm: float | None = None
    dv_ratio: float | None = None
    dv_ttm: float | None = None
    total_share: float | None = None
    float_share: float | None = None
    free_share: float | None = None
    total_mv: float | None = None
    circ_mv: float | None = None
