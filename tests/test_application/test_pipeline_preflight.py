"""preflight.py + gates.py 单测 — 不触 网络 / 外部 API."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from floatshare.application.pipeline.gates import (
    already_ran_today,
    ckpt_exists,
    prior_stage_succeeded,
    raw_daily_has_today,
)
from floatshare.application.pipeline.preflight import (
    PreflightCheck,
    PreflightResult,
    _check_db_writable,
    _check_disk_space,
    _check_trade_calendar_fresh,
    _check_tushare_token,
    has_fatal_failure,
    run_preflight,
)
from floatshare.domain.enums import PipelineStage

# === preflight 单项 =========================================================


def test_db_writable_new_dir(tmp_path: Path) -> None:
    r = _check_db_writable(tmp_path / "sub" / "new.db")
    assert r.ok
    assert "可写" in r.message


def test_db_writable_readonly_parent_fails(tmp_path: Path) -> None:
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o555)  # 只读
    try:
        sub = ro / "new.db"
        r = _check_db_writable(sub)
        assert not r.ok
        assert r.severity == "fatal"
    finally:
        ro.chmod(0o755)  # 恢复, 以便 pytest 清理


def test_disk_space_ok(tmp_path: Path) -> None:
    r = _check_disk_space(tmp_path, min_free_gb=0.001)
    assert r.ok
    assert "剩余" in r.message


def test_disk_space_threshold_too_high(tmp_path: Path) -> None:
    r = _check_disk_space(tmp_path, min_free_gb=99999.0)  # 绝对撑不住
    assert not r.ok
    assert r.severity == "fatal"


def test_trade_calendar_missing_db(tmp_path: Path) -> None:
    """DB 文件不存在 → warn, 不 fatal (允许首次 sync)."""
    r = _check_trade_calendar_fresh(tmp_path / "missing.db")
    assert not r.ok
    assert r.severity == "warn"


def test_trade_calendar_missing_table(tmp_path: Path) -> None:
    db = tmp_path / "empty.db"
    sqlite3.connect(str(db)).close()
    r = _check_trade_calendar_fresh(db)
    assert not r.ok
    assert r.severity == "warn"


def test_trade_calendar_fresh_passes(tmp_path: Path) -> None:
    """trade_calendar 有近期数据 → ok=True."""
    from datetime import date

    db = tmp_path / "cal.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE trade_calendar (trade_date TEXT PRIMARY KEY, is_open INTEGER)")
    conn.execute("INSERT INTO trade_calendar VALUES (?, 1)", (date.today().isoformat(),))
    conn.commit()
    conn.close()
    r = _check_trade_calendar_fresh(db)
    assert r.ok
    assert "距今 0 天" in r.message


def test_trade_calendar_stale_fails(tmp_path: Path) -> None:
    """trade_calendar 最新日期超过阈值 → fatal."""
    db = tmp_path / "stale.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE trade_calendar (trade_date TEXT PRIMARY KEY, is_open INTEGER)")
    conn.execute("INSERT INTO trade_calendar VALUES ('2020-01-01', 1)")
    conn.commit()
    conn.close()
    r = _check_trade_calendar_fresh(db, max_stale_days=30)
    assert not r.ok
    assert r.severity == "fatal"


# === tushare token ==========================================================


def test_tushare_token_missing(monkeypatch) -> None:
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    r = _check_tushare_token()
    assert not r.ok
    assert r.severity == "fatal"
    assert "缺失" in r.message or "过短" in r.message


def test_tushare_token_short(monkeypatch) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "short")
    r = _check_tushare_token()
    assert not r.ok


def test_tushare_token_ok(monkeypatch) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "a" * 40)
    r = _check_tushare_token()
    assert r.ok
    assert "前缀" in r.message
    assert "aaaa" in r.message


# === run_preflight 聚合 =====================================================


def test_run_preflight_aggregates_mixed() -> None:
    """混合 ok + fail, run_preflight 返回全量 list 且 has_fatal_failure 正确."""

    def _ok() -> PreflightResult:
        return PreflightResult("ok1", ok=True, severity="fatal", message="pass")

    def _fail_fatal() -> PreflightResult:
        return PreflightResult("f1", ok=False, severity="fatal", message="boom")

    def _fail_warn() -> PreflightResult:
        return PreflightResult("w1", ok=False, severity="warn", message="minor")

    checks = (
        PreflightCheck("c_ok", _ok),
        PreflightCheck("c_fatal", _fail_fatal),
        PreflightCheck("c_warn", _fail_warn),
    )
    results = run_preflight(checks, notify_on_fail=False)
    assert len(results) == 3
    assert has_fatal_failure(results) is True

    # 只 warn 失败 → 不算 fatal
    no_fatal = run_preflight(
        (PreflightCheck("c_ok", _ok), PreflightCheck("c_warn", _fail_warn)),
        notify_on_fail=False,
    )
    assert has_fatal_failure(no_fatal) is False


def test_run_preflight_catches_exception() -> None:
    """某个 check 抛异常 → 不该让整个 preflight 崩, 转成 fatal fail."""

    def _boom() -> PreflightResult:
        raise RuntimeError("unexpected")

    results = run_preflight((PreflightCheck("c_boom", _boom),), notify_on_fail=False)
    assert len(results) == 1
    assert not results[0].ok
    assert results[0].severity == "fatal"
    assert "unexpected" in results[0].message


# === gates =================================================================


def _make_pipeline_db(tmp_path: Path) -> Path:
    """造一个带 counter_event + raw_daily 表的迷你 DB."""
    db = tmp_path / "pipeline.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE counter_event (
            event_id TEXT PRIMARY KEY, ts TEXT, scope TEXT, name TEXT,
            value REAL, tags_json TEXT, run_id TEXT
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


def test_already_ran_today_no_prior(tmp_path: Path) -> None:
    db = _make_pipeline_db(tmp_path)
    r = already_ran_today(PipelineStage.S1_SYNC, "2026-04-21", db_path=db)
    assert r.ok
    assert not r.skip


def test_already_ran_today_detects_prior_ok(tmp_path: Path) -> None:
    db = _make_pipeline_db(tmp_path)
    conn = sqlite3.connect(str(db))
    conn.execute(
        "INSERT INTO counter_event VALUES ('e1', '2026-04-22T17:00:00',"
        " 'pipeline/S1_sync', 'ok', 1.0, '{\"trade_date\": \"2026-04-21\"}', 'run1')"
    )
    conn.commit()
    conn.close()
    r = already_ran_today(PipelineStage.S1_SYNC, "2026-04-21", db_path=db)
    assert r.ok
    assert r.skip  # 已跑过, 建议跳过
    assert "已成功跑过" in r.message


def test_prior_stage_succeeded_blocks_when_missing(tmp_path: Path) -> None:
    db = _make_pipeline_db(tmp_path)
    r = prior_stage_succeeded(PipelineStage.S1_SYNC, "2026-04-21", db_path=db)
    assert not r.ok
    assert "前置" in r.message


def test_prior_stage_succeeded_passes(tmp_path: Path) -> None:
    db = _make_pipeline_db(tmp_path)
    conn = sqlite3.connect(str(db))
    conn.execute(
        "INSERT INTO counter_event VALUES ('e1', '2026-04-22T17:00:00',"
        " 'pipeline/S1_sync', 'ok', 1.0, '{\"trade_date\": \"2026-04-21\"}', 'run1')"
    )
    conn.commit()
    conn.close()
    r = prior_stage_succeeded(PipelineStage.S1_SYNC, "2026-04-21", db_path=db)
    assert r.ok


def test_raw_daily_has_today_threshold(tmp_path: Path) -> None:
    db = _make_pipeline_db(tmp_path)
    # 空: 不达标
    r = raw_daily_has_today("2026-04-21", db_path=db, min_rows=100)
    assert not r.ok

    # 插 150 行
    conn = sqlite3.connect(str(db))
    for i in range(150):
        conn.execute(
            "INSERT INTO raw_daily VALUES (?, '2026-04-21T00:00:00', 0,0,0,0,0,0)",
            (f"C{i:06d}.SZ",),
        )
    conn.commit()
    conn.close()
    r = raw_daily_has_today("2026-04-21", db_path=db, min_rows=100)
    assert r.ok
    assert "150 行" in r.message


def test_ckpt_exists_missing(tmp_path: Path) -> None:
    r = ckpt_exists(tmp_path / "missing.pt")
    assert not r.ok


def test_ckpt_exists_too_small(tmp_path: Path) -> None:
    p = tmp_path / "tiny.pt"
    p.write_bytes(b"x" * 100)  # < 1024
    r = ckpt_exists(p)
    assert not r.ok


def test_ckpt_exists_ok(tmp_path: Path) -> None:
    p = tmp_path / "good.pt"
    p.write_bytes(b"x" * 5000)
    r = ckpt_exists(p)
    assert r.ok
    assert "就绪" in r.message


# === STAGE_GATES dispatch 结构 ==============================================


def test_stage_gates_are_callables_not_strings() -> None:
    """防回归: STAGE_GATES value 必须是 callable tuple, 不能回到字符串 dispatch."""
    from floatshare.application.pipeline.gates import STAGE_GATES

    for stage, gates in STAGE_GATES.items():
        assert isinstance(gates, tuple), f"{stage} gates 应是 tuple"
        for g in gates:
            assert callable(g), f"{stage} gate {g!r} 不是 callable (魔法字符串回归?)"


def test_stage_gates_invocable_with_standard_signature(tmp_path: Path) -> None:
    """STAGE_GATES 的每个 gate 都能 (trade_date, db_path) 调用返回 GateResult."""
    from floatshare.application.pipeline.gates import STAGE_GATES, GateResult

    db = _make_pipeline_db(tmp_path)
    for gates in STAGE_GATES.values():
        for g in gates:
            result = g("2026-04-21", db)
            assert isinstance(result, GateResult)
