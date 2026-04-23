"""运维观测 — counter (原始计数) + KPI (组合分析) 两层时序存储.

设计:
    counter_event  — 追加即走的原始时间序列. scope/name/value/tags 维度自由.
    kpi_snapshot   — 周期性分析快照. 由 counter_event 聚合而来, 带分析含义.

用法 (详见 floatshare.observability.metrics):
    with run_context() as run_id:
        with time_scope("pipeline/S1_sync"):
            ...                                          # → counter: duration_s
        record_counter("audit/rsi12", "nan_count", 14)   # → counter: 原始计数
        record_kpi("backtest", "v9-2026-04-21",
                   "excess_ratio", 0.128)                # → KPI: 分析级
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class CounterEvent:
    """原始计数事件 — 追加即走, 高频, schema 自由.

    scope/name 约定 (便于聚合):
        scope = '<domain>/<subject>'
            e.g. 'pipeline/S1_sync', 'audit/rsi12', 'backtest/v9-2026-04-21'
        name  = 纯计数维度
            e.g. 'duration_s', 'nan_count', 'mismatches', 'daily_alpha'

    tags_json: 额外维度 JSON (indexable 粒度不需要), e.g.
        {"trade_date": "2026-04-21", "feature": "rsi12"}

    PK 说明: 用 uuid4 hex (event_id) 做 PK — 省掉 INTEGER AUTOINCREMENT 对
    schema_sql.ddl 的特例处理, 同时保持 INSERT OR REPLACE 幂等 (uuid 不会撞).
    """

    TABLE: ClassVar[str] = "counter_event"
    PK: ClassVar[tuple[str, ...]] = ("event_id",)

    event_id: str  # uuid4 hex — record_counter 自动生成
    ts: str  # ISO8601 'YYYY-MM-DDTHH:MM:SS'
    scope: str
    name: str
    value: float
    tags_json: str | None = None
    run_id: str | None = None


@dataclass(frozen=True, slots=True)
class SyncKpis:
    """sync 一次执行的 KPI 聚合 — 字段名 = KPI 名 (单一真相来源).

    使用方式 (在 cli/run_sync.py 末尾):
        k = SyncKpis(tables_synced=len(selected), error_count=len(errors), codes_total=len(codes))
        for f in dataclasses.fields(k):
            record_kpi(Metric.Domain.SYNC, today, f.name, getattr(k, f.name))
    """

    tables_synced: int
    error_count: int
    codes_total: int


@dataclass(frozen=True, slots=True)
class KpiSnapshot:
    """分析级指标快照 — counter 的组合, 带分析含义.

    domain/subject/kpi_name 约定:
        domain    = 'pipeline' | 'backtest' | 'audit' | 'train' | 'db_integrity'
        subject   = 具体对象标识, e.g. '2026-04-21', 'v9-ckpt123', 'rsi12-drift'
        kpi_name  = 分析指标, e.g. 'excess_ratio', 'sharpe', 'total_duration_s'

    details_json 可回溯参与计算的 counter_event id / 公式参数.
    """

    TABLE: ClassVar[str] = "kpi_snapshot"
    PK: ClassVar[tuple[str, ...]] = ("snapshot_id",)

    snapshot_id: str  # uuid4 hex — record_kpi 自动生成
    ts: str
    domain: str
    subject: str
    kpi_name: str
    value: float
    details_json: str | None = None
    run_id: str | None = None
