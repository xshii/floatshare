"""MetricsTracker 单测 — 覆盖 RunHandle / context manager / query helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from floatshare.ml.tracking import (
    MetricsTracker,
    RunStatus,
    delete_run,
    get_metrics,
    get_run,
    list_runs,
    set_note,
)


def test_run_context_manager_end_to_end(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    t = MetricsTracker(db_path=db)

    with t.run(
        "PopTrainer",
        config={"lr": 1e-3, "seq_len": 60},
        note="smoke",
        n_params=207047,
        metric_key="auc",
    ) as handle:
        assert handle.run_id.startswith("PopTrainer_")
        for epoch in range(3):
            handle.log_epoch(
                epoch,
                train_metrics={"train_loss": 0.7 - epoch * 0.01},
                val_metrics={"val_auc": 0.55 + epoch * 0.02, "auc": 0.55 + epoch * 0.02},
                lr=1e-3 * (1 - epoch * 0.1),
                train_time_s=280.0,
                eval_time_s=5.0,
            )
        handle.update_best(0.59, 2)

    runs = list_runs(db_path=db)
    assert len(runs) == 1
    assert runs[0]["run_id"] == handle.run_id
    assert runs[0]["best_metric"] == 0.59
    assert runs[0]["best_epoch"] == 2
    assert runs[0]["status"] == RunStatus.DONE.value  # 自动 DONE on clean exit
    assert runs[0]["note"] == "smoke"
    assert runs[0]["finished_at"] is not None

    metrics = get_metrics(handle.run_id, db_path=db)
    assert len(metrics) == 3
    assert metrics[0]["epoch"] == 0
    assert metrics[2]["epoch"] == 2


def test_exception_in_run_block_marks_crashed(tmp_path: Path) -> None:
    """context manager 对 raise 自动 status=CRASHED, 仍向上传 exception."""
    db = tmp_path / "m.db"
    t = MetricsTracker(db_path=db)

    def _crash() -> None:
        with t.run("PopTrainer", config={}) as handle:
            handle.log_epoch(
                0,
                train_metrics={"train_loss": 0.9},
                lr=1e-3,
                train_time_s=1.0,
            )
            raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        _crash()

    runs = list_runs(db_path=db)
    assert len(runs) == 1
    assert runs[0]["status"] == RunStatus.CRASHED.value
    assert runs[0]["finished_at"] is not None


def test_set_note_updates_existing_run(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    t = MetricsTracker(db_path=db)
    with t.run("PopTrainer", config={}, note="old") as handle:
        run_id = handle.run_id

    assert set_note(run_id, "new note", db_path=db)
    runs = list_runs(db_path=db)
    assert runs[0]["note"] == "new note"
    assert not set_note("nonexistent_id", "x", db_path=db)


def test_delete_run_removes_metrics(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    t = MetricsTracker(db_path=db)
    with t.run("PopTrainer", config={}) as handle:
        handle.log_epoch(0, train_metrics={"loss": 1}, lr=1e-3, train_time_s=1.0)
        run_id = handle.run_id

    assert delete_run(run_id, db_path=db)
    assert list_runs(db_path=db) == []
    assert get_metrics(run_id, db_path=db) == []
    assert not delete_run(run_id, db_path=db)


def test_list_runs_filter_by_trainer(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    t = MetricsTracker(db_path=db)
    with t.run("PopTrainer", config={}):
        pass
    with t.run("GRPOTrainer", config={}):
        pass

    assert len(list_runs(db_path=db)) == 2
    assert len(list_runs(db_path=db, trainer="PopTrainer")) == 1
    assert list_runs(db_path=db, trainer="PopTrainer")[0]["trainer"] == "PopTrainer"


def test_list_runs_empty_db_returns_empty(tmp_path: Path) -> None:
    assert list_runs(db_path=tmp_path / "nonexistent.db") == []


def test_get_run_returns_none_when_missing(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    assert get_run("anything", db_path=db) is None

    t = MetricsTracker(db_path=db)
    with t.run("PopTrainer", config={}, note="hi") as handle:
        run_id = handle.run_id

    r = get_run(run_id, db_path=db)
    assert r is not None
    assert r["run_id"] == run_id
    assert r["note"] == "hi"
    assert get_run("does_not_exist", db_path=db) is None


def test_multiple_sequential_runs_share_tracker(tmp_path: Path) -> None:
    """同一 tracker 多次 run(): 每个独立 run_id, 无 state 污染."""
    db = tmp_path / "m.db"
    t = MetricsTracker(db_path=db)

    run_ids = []
    for i in range(3):
        with t.run("PopTrainer", config={"lr": 1e-3}, note=f"run-{i}") as handle:
            handle.log_epoch(0, train_metrics={"loss": 0.5}, lr=1e-3, train_time_s=1.0)
            run_ids.append(handle.run_id)

    assert len(set(run_ids)) == 3  # 唯一
    runs = list_runs(db_path=db)
    assert len(runs) == 3
    assert {r["status"] for r in runs} == {RunStatus.DONE.value}


def test_list_runs_on_zero_byte_db_does_not_crash(tmp_path: Path) -> None:
    """Regression: 0-byte sqlite file (半拉创建) 过去会炸 OperationalError."""
    db = tmp_path / "m.db"
    db.touch()
    assert db.stat().st_size == 0

    assert list_runs(db_path=db) == []
    assert get_metrics("anything", db_path=db) == []
    assert get_run("anything", db_path=db) is None
    assert not set_note("anything", "x", db_path=db)
    assert not delete_run("anything", db_path=db)


def test_many_writes_do_not_leak_connections(tmp_path: Path) -> None:
    """Regression: sqlite3 的 `with c:` 不 close, 必须 @contextmanager 才不 leak fd."""
    import resource

    soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    iters = 400
    assert iters < soft, "测试环境 fd limit 太小"

    db = tmp_path / "m.db"
    t = MetricsTracker(db_path=db)
    with t.run("PopTrainer", config={}) as handle:
        for epoch in range(iters):
            handle.log_epoch(
                epoch,
                train_metrics={"train_loss": 0.5},
                lr=1e-3,
                train_time_s=0.1,
            )
    for _ in range(iters):
        list_runs(db_path=db)


def test_zero_auc_not_masked_by_or_fallback(tmp_path: Path) -> None:
    """Regression: CLI val_auc=0.0 时不应被 `or` falsy 跳过到 auc 字段."""
    import json as _json

    db = tmp_path / "m.db"
    t = MetricsTracker(db_path=db)
    with t.run("PopTrainer", config={}, metric_key="auc") as handle:
        handle.log_epoch(
            0,
            train_metrics={"train_loss": 0.7},
            val_metrics={"val_auc": 0.0, "auc": 0.0, "val_p@10": 0.0},
            lr=1e-3,
            train_time_s=1.0,
        )
        run_id = handle.run_id

    metrics = get_metrics(run_id, db_path=db)
    assert len(metrics) == 1
    va = _json.loads(metrics[0]["val_metrics_json"])
    auc = va.get("val_auc", va.get("auc"))
    assert auc == 0.0  # 不是 None (避免 `or` fallback 把 0.0 吞了)
