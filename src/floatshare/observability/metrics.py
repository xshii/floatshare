"""Counter + KPI 时序指标 API — 进程内 API, 通过可插拔 sink 持久化.

设计契约:
    - counter = 原始计数事件 (append-only stream)
    - kpi     = 由 counter 组合出的分析级指标
    - run_id  = 同一次 CLI 执行共享的 UUID, 通过 ContextVar 自动传递

分层: 本模块只依赖 domain.records + stdlib, 不依赖 infrastructure.
     持久化通过 _MetricsSink Protocol 注入 (SqliteMetricsSink 等).

用法:
    # 入口处一次性设置 (application 组合根负责)
    from floatshare.infrastructure.storage.metrics_sink import SqliteMetricsSink
    set_sink(SqliteMetricsSink(db))

    # 业务代码
    with run_context() as rid:
        with time_scope("pipeline/S1_sync"):           # 自动 record_counter(duration_s)
            sync_data()
        record_counter("audit/rsi12", "nan_count", 14)
        record_kpi("backtest", "v9-2026-04-21", "excess_ratio", 0.128)

Sink 未注入时 record_* 仅打 debug 日志 (测试 / 无 DB 环境下无副作用).
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from floatshare.domain.records import CounterEvent, KpiSnapshot
from floatshare.observability import logger

# === 指标命名空间 — 统一 Metric.Domain/Counter/Kpi (CLAUDE.md 全局规范第一条) ===
#     与 domain.enums.DataSourceKind / DataKind 同款嵌套结构 (Cookbook 9.16)


class Metric:
    """指标命名空间 — 通过嵌套 StrEnum 表达 "类别 + 名字" 两级, 消灭魔法字符串.

    访问:
        Metric.Domain.BACKTEST → "backtest"
        Metric.Counter.LATENCY_MS → "latency_ms"
        Metric.Kpi.EXCESS_RATIO → "excess_ratio"

    设计原则:
        - 凡是 dataclass 字段就是 KPI 名的场景 (SignalMetrics / SuspensionReport /
          SyncKpis), 直接 `for f in fields(cls): record_kpi(..., f.name, ...)`,
          不要在 Metric.Kpi 里重复登记 — 单一真相来源 (CLAUDE.md 函数设计第 1 条).
        - 只有 **派生/聚合** KPI (非 dataclass 原生字段) 进 Metric.Kpi.
    """

    class Domain(StrEnum):
        """kpi_snapshot.domain 的闭合集合. 新加领域 → append 一条."""

        PIPELINE = "pipeline"  # S1/S2/S3/S4/S5 编排时间
        BACKTEST = "backtest"  # 回测/eval 指标
        AUDIT = "audit"  # feature 数据质量
        SYNC = "sync"  # daily-sync 任务级
        TRAIN = "train"  # 训练曲线 (per-epoch)
        DB_INTEGRITY = "db_integrity"  # DB 完整性检查结果
        HEALTHCHECK = "healthcheck"  # 数据源 API 健康

    class Counter(StrEnum):
        """counter_event.name 的常用维度. 加新维度追加一条."""

        DURATION_S = "duration_s"  # time_scope 默认写的耗时
        LATENCY_MS = "latency_ms"  # API probe 延迟
        OK = "ok"  # 布尔 0/1 (probe/任务成功)
        DAILY_LOG_RETURN = "daily_log_return"  # 每日 log return (回测曲线)
        DAILY_ALPHA = "daily_alpha"  # 每日 alpha (vs bench)
        NAN_COUNT = "nan_count"  # audit: 某特征 NaN 行数
        WINSORIZE_LOW = "winsorize_low_count"
        WINSORIZE_HIGH = "winsorize_high_count"
        MISMATCHES = "mismatches"  # tushare 对拍 / spot-check 不一致数
        RAW_DAILY_ROWS = "raw_daily_rows_today"
        DUPLICATES = "duplicates"
        CROSS_MISSING_PAIRS = "cross_missing_pairs"
        ROW_COUNT_JUMPS = "row_count_jumps"

    class Kpi(StrEnum):
        """派生 / 聚合 KPI 名 — 不登记 dataclass 字段派生的 (那些走 fields())."""

        EXCESS_RATIO = "excess_ratio"  # (cum - bench) / |bench|
        SUSPENDED_BLINDSPOT_RATE = "suspended_blindspot_rate"  # T→T+1 停牌命中率
        TOTAL_DURATION_S = "total_duration_s"
        SUCCESS_RATE_7D = "success_rate_7d"
        FEATURE_QUALITY_SCORE = "feature_quality_score"
        TUSHARE_CONSISTENCY = "tushare_consistency"


def scope(*parts: str) -> str:
    """拼接 scope 字符串 (e.g. scope(Metric.Domain.HEALTHCHECK, 'tushare', 'get_raw_daily')
    → 'healthcheck/tushare/get_raw_daily'). 消除散落的 f-string 拼接."""
    return "/".join(str(p) for p in parts)


if TYPE_CHECKING:
    from collections.abc import Sequence


class MetricsSink(Protocol):
    """持久化 sink 契约 — infrastructure 层提供 SQLite/内存实现."""

    def write_counter(self, events: Sequence[CounterEvent]) -> None: ...
    def write_kpi(self, snapshots: Sequence[KpiSnapshot]) -> None: ...


# === 全局状态 =========================================================

_sink: MetricsSink | None = None
_current_run_id: ContextVar[str | None] = ContextVar("floatshare_run_id", default=None)


def set_sink(sink: MetricsSink | None) -> None:
    """注入 / 重置 sink. 测试里用 None 关闭持久化."""
    global _sink
    _sink = sink


def get_run_id() -> str | None:
    """读当前 run_id (ContextVar). 没 run_context 就是 None."""
    return _current_run_id.get()


# === 核心 API =========================================================


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _json_or_none(d: dict[str, Any], **dumps_kwargs: Any) -> str | None:
    """非空 dict → JSON, 空 dict → None. 统一处理 tags/details."""
    return json.dumps(d, ensure_ascii=False, **dumps_kwargs) if d else None


def _make_counter(scope: str, name: str, value: float, tags: dict[str, Any]) -> CounterEvent:
    return CounterEvent(
        event_id=uuid4().hex,
        ts=_now_iso(),
        scope=scope,
        name=name,
        value=float(value),
        tags_json=_json_or_none(tags),
        run_id=get_run_id(),
    )


def _make_kpi(
    domain: str, subject: str, kpi_name: str, value: float, details: dict[str, Any]
) -> KpiSnapshot:
    return KpiSnapshot(
        snapshot_id=uuid4().hex,
        ts=_now_iso(),
        domain=domain,
        subject=subject,
        kpi_name=kpi_name,
        value=float(value),
        details_json=_json_or_none(details, default=str),
        run_id=get_run_id(),
    )


def record_counter(scope: str, name: str, value: float, **tags: Any) -> None:
    """记录一次计数事件 (原始计数).

    Args:
        scope: '<domain>/<subject>' 形式, e.g. 'pipeline/S1_sync'
        name:  纯计数维度, e.g. 'duration_s', 'nan_count'
        value: 数值 (int/float 自动转 float)
        **tags: 额外维度, 自动 JSON 序列化
    """
    evt = _make_counter(scope, name, value, tags)
    if _sink is None:
        logger.debug(f"[metrics] counter (no sink): {scope}.{name}={value}")
        return
    _sink.write_counter([evt])


def record_kpi(
    domain: str,
    subject: str,
    kpi_name: str,
    value: float,
    **details: Any,
) -> None:
    """记录一条 KPI 快照 (分析级指标).

    Args:
        domain:   'pipeline' | 'backtest' | 'audit' | 'train' | 'db_integrity'
        subject:  具体对象, e.g. '2026-04-21', 'v9-ckpt123'
        kpi_name: 分析指标名, e.g. 'excess_ratio', 'sharpe', 'total_duration_s'
        value:    数值
        **details: 计算参数/上下文, 自动 JSON 序列化
    """
    snap = _make_kpi(domain, subject, kpi_name, value, details)
    if _sink is None:
        logger.debug(f"[metrics] kpi (no sink): {domain}/{subject}.{kpi_name}={value}")
        return
    _sink.write_kpi([snap])


# === Context managers ================================================


@contextmanager
def time_scope(scope: str, name: str = "duration_s", **tags: Any) -> Iterator[None]:
    """with time_scope('pipeline/S1_sync'): ... 自动 record_counter 耗时 (秒).

    异常不吞, 但耗时依然落库 (便于事后分析失败路径).
    """
    t0 = time.monotonic()
    try:
        yield
    finally:
        record_counter(scope, name, time.monotonic() - t0, **tags)


@contextmanager
def run_context(run_id: str | None = None) -> Iterator[str]:
    """为一次 CLI 执行分配 run_id, 所有嵌套 record_* 自动带这个 id.

    Args:
        run_id: 显式指定, None = 生成 uuid4 的前 12 位 hex
    """
    rid = run_id or uuid4().hex[:12]
    tok = _current_run_id.set(rid)
    try:
        yield rid
    finally:
        _current_run_id.reset(tok)


# === 批处理 helper (大量 counter 时减少 IO) ============================


class CounterBatch:
    """批量记录 counter, 退出时一次性 flush. 适合 audit 循环 (39 特征 × 5 counter)."""

    def __init__(self) -> None:
        self._buf: list[CounterEvent] = []

    def record(self, scope: str, name: str, value: float, **tags: Any) -> None:
        self._buf.append(_make_counter(scope, name, value, tags))

    def flush(self) -> None:
        if self._buf and _sink is not None:
            _sink.write_counter(self._buf)
        self._buf.clear()


@contextmanager
def counter_batch() -> Iterator[CounterBatch]:
    """with counter_batch() as b: b.record(...); b.record(...) — 退出时批量 flush."""
    batch = CounterBatch()
    try:
        yield batch
    finally:
        batch.flush()
