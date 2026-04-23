"""Runner 行为单测 — fake stage fn 验证 fail-fast / fail-soft / gate skip / KPI 聚合."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from floatshare.application.bootstrap import setup_metrics
from floatshare.application.pipeline.runner import (
    StageOutcome,
    StagePolicy,
    _run_stage,
    run_pipeline,
)
from floatshare.application.pipeline.stages import StageContext
from floatshare.domain.enums import FailPolicy, PipelineStage, StageStatus
from floatshare.domain.records import CounterEvent, KpiSnapshot
from floatshare.observability.metrics import set_sink


@dataclass
class _Sink:
    counters: list[CounterEvent] = field(default_factory=list)
    kpis: list[KpiSnapshot] = field(default_factory=list)

    def write_counter(self, events) -> None:
        self.counters.extend(events)

    def write_kpi(self, snapshots) -> None:
        self.kpis.extend(snapshots)


@pytest.fixture
def sink():
    s = _Sink()
    set_sink(s)
    yield s
    set_sink(None)


@pytest.fixture
def clean_db(tmp_path: Path) -> Path:
    """空 DB, 带 counter_event / raw_daily (无前置 stage OK, 无 raw_daily 数据)."""
    db = tmp_path / "pipeline.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE counter_event (
            event_id TEXT PRIMARY KEY, ts TEXT, scope TEXT, name TEXT,
            value REAL, tags_json TEXT, run_id TEXT
        );
        CREATE TABLE kpi_snapshot (
            snapshot_id TEXT PRIMARY KEY, ts TEXT, domain TEXT, subject TEXT,
            kpi_name TEXT, value REAL, details_json TEXT, run_id TEXT
        );
        CREATE TABLE raw_daily (
            code TEXT, trade_date TEXT, open REAL, high REAL, low REAL,
            close REAL, volume REAL, amount REAL, PRIMARY KEY(code, trade_date)
        );
        """
    )
    conn.commit()
    conn.close()
    return db


def _fake_ok(extras: dict[str, Any] | None = None):
    def fn(ctx: StageContext) -> dict[str, Any]:
        return extras or {}

    return fn


def _fake_fail(msg: str = "boom"):
    def fn(ctx: StageContext) -> dict[str, Any]:
        raise RuntimeError(msg)

    return fn


# === _run_stage 直接测 =====================================================


def test_run_stage_ok_emits_duration_and_extras(sink, clean_db) -> None:
    policy = StagePolicy(PipelineStage.S1_SYNC, FailPolicy.FAST, _fake_ok({"rows": 5498}))
    outcome = _run_stage(policy, StageContext(trade_date="2026-04-21", db_path=clean_db))
    assert outcome.status == StageStatus.OK
    assert outcome.duration_s >= 0
    assert outcome.extras == {"rows": 5498}

    # emit 了 duration_s + ok + rows 三条 counter
    names = [c.name for c in sink.counters]
    assert "duration_s" in names
    assert "ok" in names
    assert "rows" in names
    ok_counter = next(c for c in sink.counters if c.name == "ok")
    assert ok_counter.value == 1.0


def test_run_stage_fail_captures_exception(sink, clean_db) -> None:
    policy = StagePolicy(PipelineStage.S1_SYNC, FailPolicy.FAST, _fake_fail("DB sync 崩了"))
    outcome = _run_stage(policy, StageContext(trade_date="2026-04-21", db_path=clean_db))
    assert outcome.status == StageStatus.FAIL
    assert outcome.error is not None
    assert "DB sync 崩了" in outcome.error

    ok_counter = next(c for c in sink.counters if c.name == "ok")
    assert ok_counter.value == 0.0


# === run_pipeline 编排分支 =================================================


def _ok_today_counter(conn: sqlite3.Connection, stage: PipelineStage) -> None:
    """人为插入 S1 的 ok counter, 让 S2+ 的 prior_stage gate 通过.

    必须带 tags_json trade_date — gate 按 tag 查不按 ts.
    """
    conn.execute(
        "INSERT INTO counter_event VALUES (?, ?, ?, 'ok', 1.0, ?, 'rid')",
        (
            f"eid-{stage.value}",
            "2026-04-22T17:00:00",
            f"pipeline/{stage.value}",
            '{"trade_date": "2026-04-21"}',
        ),
    )


def _seed_raw_daily(conn: sqlite3.Connection, n: int = 5000) -> None:
    for i in range(n):
        conn.execute(
            "INSERT INTO raw_daily VALUES (?, '2026-04-21T00:00:00', 0,0,0,0,0,0)",
            (f"C{i:06d}.SZ",),
        )


def test_run_pipeline_fail_fast_aborts_on_s1(sink, clean_db) -> None:
    """S1 FAST 失败 → 整条 pipeline abort, 后续 stage 不跑."""
    stages = (
        StagePolicy(PipelineStage.S1_SYNC, FailPolicy.FAST, _fake_fail("sync 失败")),
        StagePolicy(PipelineStage.S2A_DB_INTEGRITY, FailPolicy.FAST, _fake_ok()),
    )
    summary = run_pipeline("2026-04-21", stages, db_path=clean_db, skip_preflight=True)
    assert summary.aborted is True
    # 只跑了 S1, S2 完全没跑
    assert len(summary.stages) == 1
    assert summary.stages[0].status == StageStatus.FAIL


def test_run_pipeline_fail_soft_continues(sink, clean_db) -> None:
    """SOFT 失败 → 记 FAIL 继续; aborted=False."""
    setup_metrics(existing_db=None)  # no-op (sink 已 in-memory)
    set_sink(sink)
    # 先让 S1/S2a/prep 都 OK (通过插 counter + raw_daily 造环境)
    conn = sqlite3.connect(str(clean_db))
    _seed_raw_daily(conn, 5000)
    _ok_today_counter(conn, PipelineStage.S1_SYNC)
    _ok_today_counter(conn, PipelineStage.S2B_PREP_FEATURES)
    conn.commit()
    conn.close()

    stages = (
        StagePolicy(PipelineStage.S3A_TUSHARE_CHECK, FailPolicy.SOFT, _fake_fail("tushare 空")),
        StagePolicy(PipelineStage.S3B_FEATURE_AUDIT, FailPolicy.SOFT, _fake_ok({"n_alerts": 0})),
    )
    summary = run_pipeline("2026-04-21", stages, db_path=clean_db, skip_preflight=True)
    assert summary.aborted is False
    statuses = [s.status for s in summary.stages]
    assert StageStatus.FAIL in statuses  # S2b fail
    assert StageStatus.OK in statuses  # S3 continued


def test_run_pipeline_gate_blocks_stage(sink, clean_db) -> None:
    """S2a 要求 S1 今日 OK — S1 未跑时, S2a 进入 GATE_BLOCKED."""
    stages = (StagePolicy(PipelineStage.S2A_DB_INTEGRITY, FailPolicy.FAST, _fake_ok()),)
    summary = run_pipeline("2026-04-21", stages, db_path=clean_db, skip_preflight=True)
    # S2a gate 前置 S1 未就绪 → GATE_BLOCKED + FAST → abort
    assert summary.aborted is True
    assert summary.stages[0].status == StageStatus.GATE_BLOCKED
    assert "前置" in summary.stages[0].gate_block_reason


def test_run_pipeline_emits_kpis(sink, clean_db) -> None:
    """无论成功失败, 末尾都要 record_kpi(TOTAL_DURATION_S + SUCCESS_RATE_7D)."""
    stages = (StagePolicy(PipelineStage.S1_SYNC, FailPolicy.FAST, _fake_ok({"rows": 100})),)
    summary = run_pipeline("2026-04-21", stages, db_path=clean_db, skip_preflight=True)
    assert summary.aborted is False

    kpi_names = {k.kpi_name for k in sink.kpis}
    assert "total_duration_s" in kpi_names
    assert "success_rate_7d" in kpi_names


def test_run_pipeline_success_rate_calculation(sink, clean_db) -> None:
    """成功率 = n_ok / n_real (不算 SKIPPED). 2 OK + 1 FAIL → 0.666..."""
    conn = sqlite3.connect(str(clean_db))
    _seed_raw_daily(conn, 5000)
    _ok_today_counter(conn, PipelineStage.S1_SYNC)
    _ok_today_counter(conn, PipelineStage.S2B_PREP_FEATURES)
    conn.commit()
    conn.close()

    stages = (
        StagePolicy(PipelineStage.S3A_TUSHARE_CHECK, FailPolicy.SOFT, _fake_ok()),
        StagePolicy(PipelineStage.S3B_FEATURE_AUDIT, FailPolicy.SOFT, _fake_fail("audit 崩")),
    )
    summary = run_pipeline("2026-04-21", stages, db_path=clean_db, skip_preflight=True)
    # 2 stage 跑了, 1 OK 1 FAIL → 0.5
    assert abs(summary.success_rate - 0.5) < 1e-9


def test_stage_outcome_is_proper_dataclass() -> None:
    """StageOutcome 必须是结构化 dataclass, 不是 dict (防 dict 回归)."""
    o = StageOutcome(stage=PipelineStage.S1_SYNC, status=StageStatus.OK, duration_s=1.0)
    assert hasattr(o, "stage")
    assert hasattr(o, "status")
    assert hasattr(o, "duration_s")
    assert hasattr(o, "extras")
