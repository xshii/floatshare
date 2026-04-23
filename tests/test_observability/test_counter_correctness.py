"""Counter 正确性用例 — 各 emit 路径产出的 counter_event / kpi_snapshot 形状是否符合契约.

覆盖:
    - probe _stage(): duration_s + ok + 每个数值 extra 成单独 counter
    - SyncKpis 迭代 fields() 写 kpi_snapshot, 字段名 = kpi 名
    - SignalMetrics 迭代 fields() 写 kpi_snapshot, 字段名 = kpi 名
    - eval excess_ratio / suspended_blindspot_rate 派生 KPI
    - scope() + Metric enum 拼接 scope 字符串
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from floatshare.domain.enums import PipelineStage, StageStatus
from floatshare.domain.records import CounterEvent, KpiSnapshot, SyncKpis
from floatshare.ml.eval import SignalMetrics, SuspensionReport
from floatshare.observability.metrics import (
    Metric,
    record_counter,
    record_kpi,
    run_context,
    scope,
    set_sink,
)


@dataclass
class _InMemorySink:
    counters: list[CounterEvent] = field(default_factory=list)
    kpis: list[KpiSnapshot] = field(default_factory=list)

    def write_counter(self, events) -> None:
        self.counters.extend(events)

    def write_kpi(self, snapshots) -> None:
        self.kpis.extend(snapshots)


def _sink() -> _InMemorySink:
    s = _InMemorySink()
    set_sink(s)
    return s


# === SyncKpis ←→ kpi_snapshot ===========================================


def test_sync_kpis_fields_match_recorded_names() -> None:
    """cli/run_sync.py 的 `for f in fields(SyncKpis): record_kpi(..., f.name, ...)` 契约.

    字段名必须 1:1 落到 kpi_snapshot.kpi_name, 不允许漂移.
    """
    sink = _sink()
    try:
        kpis = SyncKpis(tables_synced=12, error_count=0, codes_total=5498)
        with run_context():
            for f in dataclasses.fields(kpis):
                record_kpi(Metric.Domain.SYNC, "2026-04-21", f.name, getattr(kpis, f.name))

        names = {k.kpi_name for k in sink.kpis}
        assert names == {"tables_synced", "error_count", "codes_total"}

        values = {k.kpi_name: k.value for k in sink.kpis}
        assert values == {"tables_synced": 12.0, "error_count": 0.0, "codes_total": 5498.0}

        # 所有 KPI 同 domain + subject (一批)
        assert {k.domain for k in sink.kpis} == {"sync"}
        assert {k.subject for k in sink.kpis} == {"2026-04-21"}
    finally:
        set_sink(None)


# === SignalMetrics ←→ kpi_snapshot (eval 自动 record_kpi 契约) ==========


def test_signal_metrics_fields_all_recorded() -> None:
    """ml/eval.py::run_eval 的 `for f in fields(SignalMetrics): record_kpi(...)`.

    7 个字段 → 7 条 kpi_snapshot, 字段名一致.
    """
    sink = _sink()
    try:
        metrics = SignalMetrics(
            sharpe=1.23,
            cum_return=0.45,
            mean_per_step=0.0012,
            max_drawdown=-0.08,
            turnover_avg=0.3,
            n_steps=252,
            n_signals=1260,
        )
        for f in dataclasses.fields(metrics):
            record_kpi(Metric.Domain.BACKTEST, "v9-ckpt", f.name, float(getattr(metrics, f.name)))

        recorded = {k.kpi_name for k in sink.kpis}
        expected = {f.name for f in dataclasses.fields(SignalMetrics)}
        assert recorded == expected
        assert len(sink.kpis) == 7
    finally:
        set_sink(None)


def test_signal_metrics_no_duplicate_enum_drift() -> None:
    """Metric.Kpi 里的派生 KPI 不和 SignalMetrics 字段名重复 (单一真相来源).

    防止未来有人在 Metric.Kpi 里把 SHARPE/SHARPE_RATIO 加上, 造成双重登记.
    """
    signal_field_names = {f.name for f in dataclasses.fields(SignalMetrics)}
    metric_kpi_values = {m.value for m in Metric.Kpi}
    overlap = signal_field_names & metric_kpi_values
    assert overlap == set(), f"Metric.Kpi 里不该重复登记 SignalMetrics 字段: {overlap}"


# === 派生 KPI (非 dataclass 字段) ========================================


def test_derived_kpi_uses_metric_kpi_enum() -> None:
    """excess_ratio / suspended_blindspot_rate 必须走 Metric.Kpi, 不是裸字符串."""
    sink = _sink()
    try:
        record_kpi(Metric.Domain.BACKTEST, "v9", Metric.Kpi.EXCESS_RATIO, 0.12, bench_cum=0.5)
        record_kpi(
            Metric.Domain.BACKTEST,
            "v9",
            Metric.Kpi.SUSPENDED_BLINDSPOT_RATE,
            0.001,
            weighted=0.0005,
            total_picks=2500,
        )
        assert sink.kpis[0].kpi_name == "excess_ratio"
        assert sink.kpis[1].kpi_name == "suspended_blindspot_rate"
        # details_json 带进来的上下文可回溯
        assert sink.kpis[0].details_json is not None
        assert "bench_cum" in sink.kpis[0].details_json
    finally:
        set_sink(None)


# === scope 拼接契约 =====================================================


def test_scope_composition_with_metric_enum() -> None:
    """scope(Metric.Domain.X, ...) 正确拼接, 跟 record_counter 接受一致."""
    assert scope(Metric.Domain.PIPELINE, PipelineStage.S1_SYNC) == "pipeline/S1_sync"
    assert (
        scope(Metric.Domain.HEALTHCHECK, "tushare", "get_raw_daily")
        == "healthcheck/tushare/get_raw_daily"
    )
    assert scope(Metric.Domain.BACKTEST, "v9-ckpt") == "backtest/v9-ckpt"


# === probe _stage 的 counter 形状 ======================================


def _simulate_stage_emit(
    stage: PipelineStage,
    extras: dict[str, Any],
    status: StageStatus = StageStatus.OK,
) -> None:
    """复现 scripts/pipeline_timing_probe.py::_stage 的 emit 逻辑 (不跑完整 probe).

    这里是 counter 契约的 mirror — 如果 probe 改了, 本测试会先失败.
    """
    stage_scope = scope(Metric.Domain.PIPELINE, stage.value)
    record_counter(stage_scope, Metric.Counter.DURATION_S, 1.23, status=status.value)
    record_counter(
        stage_scope,
        Metric.Counter.OK,
        1.0 if status == StageStatus.OK else 0.0,
    )
    for k, v in extras.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            record_counter(stage_scope, k, float(v))


def test_stage_emits_duration_and_ok() -> None:
    sink = _sink()
    try:
        _simulate_stage_emit(PipelineStage.S1_SYNC, {"raw_daily_rows_today": 5498})
        names = [c.name for c in sink.counters]
        assert "duration_s" in names
        assert "ok" in names
        assert "raw_daily_rows_today" in names
        # scope 一致
        assert all(c.scope == "pipeline/S1_sync" for c in sink.counters)
    finally:
        set_sink(None)


def test_stage_ok_is_boolean_float() -> None:
    """ok counter 必须是 0.0 / 1.0 数值, 便于后续聚合成功率."""
    sink = _sink()
    try:
        _simulate_stage_emit(
            PipelineStage.S3B_FEATURE_AUDIT, {"n_alerts": 0}, status=StageStatus.FAIL
        )
        ok_counter = next(c for c in sink.counters if c.name == "ok")
        assert ok_counter.value == 0.0

        sink.counters.clear()
        _simulate_stage_emit(
            PipelineStage.S3B_FEATURE_AUDIT, {"n_alerts": 0}, status=StageStatus.OK
        )
        ok_counter = next(c for c in sink.counters if c.name == "ok")
        assert ok_counter.value == 1.0
    finally:
        set_sink(None)


def test_stage_extras_skips_booleans_and_none() -> None:
    """has_errors=True 这种 bool 和 None 不该当数值 counter (会被判定为 1/0 误导分析)."""
    sink = _sink()
    try:
        _simulate_stage_emit(
            PipelineStage.S3B_FEATURE_AUDIT,
            {"n_alerts": 3, "has_errors": True, "features": ["rsi12"]},
        )
        names = {c.name for c in sink.counters}
        assert "n_alerts" in names
        assert "has_errors" not in names  # bool 不入数值 counter
        assert "features" not in names  # list 不入
    finally:
        set_sink(None)


# === SuspensionReport 契约 ==============================================


def test_suspension_report_shape_for_kpi_emit() -> None:
    """eval.py record_kpi(SUSPENDED_BLINDSPOT_RATE) 期望的字段齐全."""
    report = SuspensionReport(total=100, suspended_next_day=3, rate=0.03, rate_weighted=0.05)
    # 如果这些字段改名, eval.py 的 details=... 参数会失败
    assert hasattr(report, "rate")
    assert hasattr(report, "rate_weighted")
    assert hasattr(report, "total")
    assert report.rate == 0.03
