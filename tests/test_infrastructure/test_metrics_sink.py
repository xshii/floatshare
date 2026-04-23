"""SqliteMetricsSink 写入 + 读回验证 — 真 SQLite 文件, 跨 record_counter/record_kpi 闭环."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from floatshare.application.bootstrap import cli_metrics_run, setup_metrics
from floatshare.observability.metrics import (
    Metric,
    record_counter,
    record_kpi,
    run_context,
    set_sink,
    time_scope,
)


@pytest.fixture
def tmp_db(tmp_path: Path) -> str:
    """每个测试一个隔离 DB, 初始化 counter_event + kpi_snapshot 表."""
    db_path = tmp_path / "metrics.db"
    yield str(db_path)
    set_sink(None)  # 清理全局 sink, 避免污染下一用例


def _count_rows(db_path: str, table: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


def _fetch(db_path: str, table: str, cols: str) -> list[tuple]:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(f"SELECT {cols} FROM {table}").fetchall()
    finally:
        conn.close()


def test_setup_metrics_creates_tables(tmp_db: str) -> None:
    setup_metrics(tmp_db)
    assert _count_rows(tmp_db, "counter_event") == 0
    assert _count_rows(tmp_db, "kpi_snapshot") == 0


def test_record_counter_writes_to_sqlite(tmp_db: str) -> None:
    setup_metrics(tmp_db)
    with run_context():
        record_counter("pipeline/S1", Metric.Counter.DURATION_S, 60.5, status="OK")
    rows = _fetch(tmp_db, "counter_event", "scope, name, value, run_id")
    assert len(rows) == 1
    scope_v, name, value, run_id = rows[0]
    assert scope_v == "pipeline/S1"
    assert name == "duration_s"
    assert value == 60.5
    assert run_id is not None
    assert len(run_id) == 12


def test_record_kpi_writes_to_sqlite(tmp_db: str) -> None:
    setup_metrics(tmp_db)
    record_kpi(Metric.Domain.BACKTEST, "v9-ckpt1", "sharpe", 1.234, note="demo")
    rows = _fetch(tmp_db, "kpi_snapshot", "domain, subject, kpi_name, value")
    assert rows == [("backtest", "v9-ckpt1", "sharpe", 1.234)]


def test_cli_metrics_run_injects_id_and_flushes(tmp_db: str) -> None:
    """cli_metrics_run CM 整体测试: setup + run_context + 子项写入 + 退出."""
    with cli_metrics_run(db_path=tmp_db) as (db, run_id):
        assert db.db_path == Path(tmp_db)
        assert len(run_id) == 12
        with time_scope("pipeline/S1"):
            pass
        record_kpi(Metric.Domain.SYNC, "2026-04-21", "tables_synced", 12)

    # 退出后, run_id 应清空
    from floatshare.observability.metrics import get_run_id

    assert get_run_id() is None

    counters = _fetch(tmp_db, "counter_event", "run_id")
    kpis = _fetch(tmp_db, "kpi_snapshot", "run_id")
    assert counters[0][0] == run_id
    assert kpis[0][0] == run_id


def test_multiple_writes_same_run(tmp_db: str) -> None:
    """同一 run_context 内多条写入共享 run_id."""
    with cli_metrics_run(db_path=tmp_db) as (_db, run_id):
        for i in range(5):
            record_counter("test", "x", float(i))
    rows = _fetch(tmp_db, "counter_event", "run_id, value")
    assert len(rows) == 5
    assert all(r[0] == run_id for r in rows)
