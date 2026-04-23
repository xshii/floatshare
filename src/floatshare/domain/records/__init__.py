"""数据记录 dataclass 包 — 表 schema 的单一真相来源 (Single Source of Truth)。

Python Cookbook 风格 (Recipe 8.10/9.20): 用 @dataclass + ClassVar(TABLE/PK)
表达表结构，DDL / INSERT SQL / 列名都从它自动生成
(`infrastructure/storage/schema_sql.py`)。

约定:
- ClassVar TABLE: SQL 表名
- ClassVar PK:    主键列名元组
- ClassVar FIELDS (可选): 字段 → FieldSpec (label/primary/unit)，驱动前端渐进式披露
- 字段类型注解 → SQLite 类型:
    str / str | None       → TEXT
    int / int | None       → INTEGER
    float / float | None   → REAL
- PK 列自动 NOT NULL；非 Optional 字段也 NOT NULL

子模块按主题/频率组织 (单文件 < 200 行):
    daily_price / daily_flow / daily_chip / daily_macro
    monthly / quarterly / event / reference / book / ops

新加表:
    1. 找到对应子模块加 dataclass
    2. 在本文件 import 一行
    → ALL_RECORDS 自动收集 (无需手动追加)
    → DDL/CRUD 自动可用 (db.save(MyRecord, df))
"""

from __future__ import annotations

import dataclasses as _dc
import sys as _sys

from floatshare.domain.records._meta import FieldSpec, RecordSchema
from floatshare.domain.records.book import (
    CashAccount,
    CashTxn,
    DcaExecution,
    DcaPlan,
)
from floatshare.domain.records.cctv_news import CctvNewsMention, CctvNewsRaw
from floatshare.domain.records.daily_chip import ChipDist, ChipPerf
from floatshare.domain.records.daily_flow import (
    MarginDetail,
    Moneyflow,
    MoneyflowHsgt,
)
from floatshare.domain.records.daily_macro import FxDaily, Shibor
from floatshare.domain.records.daily_price import (
    AdjFactor,
    DailyBasic,
    IndexDaily,
    RawDaily,
)
from floatshare.domain.records.event import Dividend, TopInst, TopList
from floatshare.domain.records.monthly import BrokerPicks, CnCpi, CnPpi
from floatshare.domain.records.ops import CounterEvent, KpiSnapshot, SyncKpis
from floatshare.domain.records.quarterly import (
    Balancesheet,
    Cashflow,
    EarningsForecast,
    FinaIndicator,
    Income,
    StkHolderNumber,
)
from floatshare.domain.records.reference import (
    ConceptBoard,
    ConceptMember,
    IndexWeight,
    Industry,
    StockInfo,
    StockLifecycle,
    SyncWatermark,
    TradeCalendar,
)


def _discover_records() -> tuple[type, ...]:
    """自动收集本包导入的所有带 TABLE ClassVar 的 dataclass。

    Cookbook Recipe 9.16 + 8.10 思想 — 让标记 (TABLE) 充当注册指标，
    而不是另开一份手写注册表。新加表只需 import 一行。
    """
    return tuple(
        obj
        for obj in vars(_sys.modules[__name__]).values()
        if isinstance(obj, type) and _dc.is_dataclass(obj) and hasattr(obj, "TABLE")
    )


ALL_RECORDS: tuple[type, ...] = _discover_records()

# __all__ 派生自 ALL_RECORDS，单一真相来源
# (ruff 静态扫描看不出 c.__name__ 是 str → noqa)
__all__ = [  # noqa: PLE0604
    "ALL_RECORDS",
    "FieldSpec",
    *sorted(c.__name__ for c in ALL_RECORDS),
]
