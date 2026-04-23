"""observability.metrics 单元测试 — 纯 in-memory sink, 验证 API 契约.

覆盖:
    record_counter / record_kpi 基本路径
    time_scope 自动写 duration_s
    run_context 注入 run_id 到 counter + kpi
    counter_batch 批写入
    set_sink(None) 时 record_* 不崩 (只落 debug 日志)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from floatshare.domain.records import CounterEvent, KpiSnapshot
from floatshare.observability.metrics import (
    Metric,
    counter_batch,
    get_run_id,
    record_counter,
    record_kpi,
    run_context,
    scope,
    set_sink,
    time_scope,
)


@dataclass
class _InMemorySink:
    """in-memory 实现 MetricsSink Protocol, 收集所有写入用于断言."""

    counters: list[CounterEvent] = field(default_factory=list)
    kpis: list[KpiSnapshot] = field(default_factory=list)

    def write_counter(self, events) -> None:
        self.counters.extend(events)

    def write_kpi(self, snapshots) -> None:
        self.kpis.extend(snapshots)


def _with_sink() -> _InMemorySink:
    sink = _InMemorySink()
    set_sink(sink)
    return sink


def test_record_counter_basic() -> None:
    sink = _with_sink()
    try:
        record_counter("test/scope", Metric.Counter.LATENCY_MS, 123.4, tag="foo")
        assert len(sink.counters) == 1
        c = sink.counters[0]
        assert c.scope == "test/scope"
        assert c.name == "latency_ms"
        assert c.value == 123.4
        assert c.tags_json is not None
        assert "foo" in c.tags_json
        assert c.run_id is None  # 无 run_context
    finally:
        set_sink(None)


def test_record_kpi_basic() -> None:
    sink = _with_sink()
    try:
        record_kpi(Metric.Domain.BACKTEST, "v9-ckpt1", "sharpe", 1.234, note="demo")
        assert len(sink.kpis) == 1
        k = sink.kpis[0]
        assert k.domain == "backtest"
        assert k.subject == "v9-ckpt1"
        assert k.kpi_name == "sharpe"
        assert k.value == 1.234
        assert k.details_json is not None
        assert "demo" in k.details_json
    finally:
        set_sink(None)


def test_time_scope_writes_duration() -> None:
    sink = _with_sink()
    try:
        with time_scope("test/slow"):
            pass
        assert len(sink.counters) == 1
        c = sink.counters[0]
        assert c.name == Metric.Counter.DURATION_S
        assert c.value >= 0  # monotonic 不会倒流
    finally:
        set_sink(None)


def test_time_scope_records_on_exception() -> None:
    """异常路径也落耗时 (便于诊断失败 stage)."""
    sink = _with_sink()
    try:
        import pytest

        with pytest.raises(ValueError, match="boom"), time_scope("test/fail"):
            raise ValueError("boom")
        assert len(sink.counters) == 1
    finally:
        set_sink(None)


def test_run_context_injects_run_id() -> None:
    sink = _with_sink()
    try:
        assert get_run_id() is None
        with run_context() as rid:
            assert get_run_id() == rid
            assert len(rid) == 12  # uuid4 hex[:12]
            record_counter("test/inctx", "x", 1.0)
            record_kpi("test", "s", "k", 1.0)
        assert get_run_id() is None  # 上下文退出后清空
        assert sink.counters[0].run_id == rid
        assert sink.kpis[0].run_id == rid
    finally:
        set_sink(None)


def test_run_context_explicit_id() -> None:
    sink = _with_sink()
    try:
        with run_context(run_id="deadbeef") as rid:
            assert rid == "deadbeef"
            record_counter("test", "x", 1.0)
        assert sink.counters[0].run_id == "deadbeef"
    finally:
        set_sink(None)


def test_counter_batch_flushes_on_exit() -> None:
    sink = _with_sink()
    try:
        with counter_batch() as batch:
            batch.record("test", "a", 1.0)
            batch.record("test", "b", 2.0)
            assert sink.counters == []  # not flushed yet
        assert len(sink.counters) == 2
    finally:
        set_sink(None)


def test_no_sink_is_noop() -> None:
    """未注入 sink 时 record_* 仅打日志, 不崩."""
    set_sink(None)
    record_counter("test", "x", 1.0)
    record_kpi("test", "s", "k", 1.0)
    with time_scope("test"):
        pass
    # 到这里没抛异常即成功


def test_scope_builder() -> None:
    assert scope("a", "b", "c") == "a/b/c"
    assert scope(Metric.Domain.BACKTEST, "v9") == "backtest/v9"


def test_metric_enum_string_value() -> None:
    """StrEnum 成员 IS str, 可直接做 scope/name 传参."""
    assert Metric.Domain.BACKTEST == "backtest"
    assert Metric.Counter.LATENCY_MS == "latency_ms"
    assert Metric.Kpi.EXCESS_RATIO == "excess_ratio"
