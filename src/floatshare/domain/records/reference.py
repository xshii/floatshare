"""参考类 — 指数成分、行业、生命周期、交易日历、水位线、股票基础。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class IndexWeight:
    """指数成分股权重 (Tushare index_weight)。"""

    TABLE: ClassVar[str] = "index_weight"
    PK: ClassVar[tuple[str, ...]] = ("index_code", "trade_date", "con_code")

    index_code: str
    trade_date: str
    con_code: str
    weight: float | None = None


@dataclass(frozen=True, slots=True)
class Industry:
    """申万行业分类映射 (个股 → L1/L2/L3 行业)。"""

    TABLE: ClassVar[str] = "industry"
    PK: ClassVar[tuple[str, ...]] = ("code",)

    code: str
    l1_code: str | None = None
    l1_name: str | None = None
    l2_code: str | None = None
    l2_name: str | None = None
    l3_code: str | None = None
    l3_name: str | None = None


@dataclass(frozen=True, slots=True)
class StockLifecycle:
    """A 股个股生命周期 (上市/退市/暂停 三状态)。"""

    TABLE: ClassVar[str] = "stock_lifecycle"
    PK: ClassVar[tuple[str, ...]] = ("code",)

    code: str
    list_status: str
    name: str | None = None
    list_date: str | None = None
    delist_date: str | None = None
    market: str | None = None
    industry: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True, slots=True)
class TradeCalendar:
    """交易日历。"""

    TABLE: ClassVar[str] = "trade_calendar"
    PK: ClassVar[tuple[str, ...]] = ("trade_date",)

    trade_date: str
    exchange: str | None = None  # 默认 'SSE'


@dataclass(frozen=True, slots=True)
class SyncWatermark:
    """每只股票的最后同步日期 (增量同步水位线)。"""

    TABLE: ClassVar[str] = "sync_watermark"
    PK: ClassVar[tuple[str, ...]] = ("code",)

    code: str
    last_date: str
    source: str
    updated_at: str | None = None


@dataclass(frozen=True, slots=True)
class ConceptBoard:
    """概念板块清单 (默认同花顺 src='ths', 后备 Tushare src='ts')。

    和 SW 行业 (Industry 表) 的差别:
      - 行业是一对一 (一只股归属唯一 L1/L2/L3)
      - 概念是多对多 (同一只股可属多个概念, 如 CPO + AI 算力 + 半导体)
    """

    TABLE: ClassVar[str] = "concept_board"
    PK: ClassVar[tuple[str, ...]] = ("board_code",)

    board_code: str  # ths='885883.TI' / ts='TS123'
    board_name: str
    src: str | None = None  # 'ths' | 'ts'
    member_count: int | None = None
    list_date: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True, slots=True)
class ConceptMember:
    """概念板块成分股映射 (board ↔ code 多对多)。"""

    TABLE: ClassVar[str] = "concept_member"
    PK: ClassVar[tuple[str, ...]] = ("board_code", "code")

    board_code: str
    code: str
    name: str | None = None
    weight: float | None = None
    in_date: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True, slots=True)
class StockInfo:
    """股票基本信息 (legacy — stock_lifecycle 是其超集)。"""

    TABLE: ClassVar[str] = "stock_info"
    PK: ClassVar[tuple[str, ...]] = ("code",)

    code: str
    ticker: str | None = None
    name: str | None = None
    market: str | None = None
    industry: str | None = None
    list_date: str | None = None
    delist_date: str | None = None
