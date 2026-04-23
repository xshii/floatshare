"""日频 · 筹码类 — 筹码胜率、筹码分布。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from floatshare.domain.records._meta import FieldSpec


@dataclass(frozen=True, slots=True)
class ChipPerf:
    """每日筹码胜率 (Tushare cyq_perf)。"""

    TABLE: ClassVar[str] = "chip_perf"
    PK: ClassVar[tuple[str, ...]] = ("code", "trade_date")
    FIELDS: ClassVar[dict[str, FieldSpec]] = {
        "winner_rate": FieldSpec("获利盘比例", primary=True, unit="%"),
        "weight_avg": FieldSpec("加权平均价", primary=True, unit="元"),
        "cost_50pct": FieldSpec("50% 成本线", unit="元"),
        "cost_5pct": FieldSpec("5% 成本线", unit="元"),
        "cost_95pct": FieldSpec("95% 成本线", unit="元"),
    }

    code: str
    trade_date: str
    his_low: float | None = None
    his_high: float | None = None
    cost_5pct: float | None = None
    cost_15pct: float | None = None
    cost_50pct: float | None = None
    cost_85pct: float | None = None
    cost_95pct: float | None = None
    weight_avg: float | None = None
    winner_rate: float | None = None


@dataclass(frozen=True, slots=True)
class ChipDist:
    """每日筹码分布 (每股每天 ~50 价位) — Tushare cyq_chips。"""

    # 多行每股每日，不走 primary 字段展示
    TABLE: ClassVar[str] = "chip_dist"
    PK: ClassVar[tuple[str, ...]] = ("code", "trade_date", "price")

    code: str
    trade_date: str
    price: float
    percent: float | None = None
