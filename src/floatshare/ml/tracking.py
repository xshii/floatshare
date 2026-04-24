"""训练指标持久化 — SQLite-backed MetricsTracker.

目的: 日志文件易丢, 跨 run 对比不便. 把每 epoch 的 val_auc / train_loss / lr
+ 配置快照 + 用户备注存进 data/ml/metrics.db, 后续 Dash 画曲线 / CLI 列表.

Schema:
    training_runs(run_id PK, trainer, started_at, finished_at, status,
                  n_params, metric_key, best_metric, best_epoch,
                  config_json, note, git_sha, host)
    training_metrics(run_id, epoch, train_metrics_json, val_metrics_json,
                     lr, train_time_s, eval_time_s, ts)

run_id 格式: {trainer}_{YYYYMMDD-HHMMSS}_{6hex}, e.g. pop_20260423-064515_a1b2c3
"""

from __future__ import annotations

import json
import secrets
import socket
import sqlite3
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

_DEFAULT_DB = Path("data/ml/metrics.db")


class RunStatus(StrEnum):
    """training_runs.status 的取值 — state machine."""

    RUNNING = "running"
    DONE = "done"
    CRASHED = "crashed"


# 表名常量 — 避免散落在 SQL 字符串里, 重命名时一处改
_T_RUNS: Final[str] = "training_runs"
_T_METRICS: Final[str] = "training_metrics"

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {_T_RUNS} (
    run_id       TEXT PRIMARY KEY,
    trainer      TEXT NOT NULL,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    status       TEXT NOT NULL DEFAULT '{RunStatus.RUNNING.value}',
    n_params     INTEGER,
    metric_key   TEXT,
    best_metric  REAL,
    best_epoch   INTEGER,
    config_json  TEXT,
    note         TEXT,
    git_sha      TEXT,
    host         TEXT
);

CREATE TABLE IF NOT EXISTS {_T_METRICS} (
    run_id              TEXT NOT NULL,
    epoch               INTEGER NOT NULL,
    train_metrics_json  TEXT,
    val_metrics_json    TEXT,
    lr                  REAL,
    train_time_s        REAL,
    eval_time_s         REAL,
    ts                  TEXT NOT NULL,
    PRIMARY KEY (run_id, epoch),
    FOREIGN KEY (run_id) REFERENCES {_T_RUNS}(run_id)
);

CREATE INDEX IF NOT EXISTS idx_runs_trainer_time
    ON {_T_RUNS}(trainer, started_at DESC);
"""


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
        return out or None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


@contextmanager
def _open_rw(db_path: Path) -> Iterator[sqlite3.Connection]:
    """读写端 context manager: 父目录 + schema idempotent + commit + close.

    `sqlite3.Connection` 本身的 `with` 只 commit/rollback, 不 close — 所以必须
    自己做 context manager 保证 `c.close()`, 否则每次调用都 leak 一个 fd.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(db_path)
    try:
        c.executescript(_SCHEMA)
        yield c
        c.commit()
    finally:
        c.close()


@contextmanager
def _open_ro(db_path: Path) -> Iterator[sqlite3.Connection | None]:
    """只读端 context manager: db 不存在 yield None, 存在 yield conn + 保证 close.

    db 文件不存在 → yield None, 不副作用地创 db. 既存文件但表没建 (0-byte
    / 上次崩溃残留) → executescript(_SCHEMA) 幂等补齐后 yield conn.
    """
    if not db_path.exists():
        yield None
        return
    c = sqlite3.connect(db_path)
    try:
        c.executescript(_SCHEMA)
        yield c
        c.commit()
    finally:
        c.close()


@dataclass(slots=True)
class RunHandle:
    """单次 run 的生命周期 owner — 所有 epoch 写入只能通过它.

    由 `MetricsTracker.run()` context manager 创建并管理: 进入 with-block
    时 INSERT 一行 training_runs, 退出时自动 UPDATE status=done/crashed.
    RunHandle 是不可重入的 one-shot 对象.
    """

    run_id: str
    _db_path: Path = field(repr=False)

    def log_epoch(
        self,
        epoch: int,
        *,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None = None,
        lr: float,
        train_time_s: float,
        eval_time_s: float = 0.0,
    ) -> None:
        with _open_rw(self._db_path) as c:
            c.execute(
                f"INSERT OR REPLACE INTO {_T_METRICS} "
                "(run_id, epoch, train_metrics_json, val_metrics_json, "
                " lr, train_time_s, eval_time_s, ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self.run_id,
                    epoch,
                    json.dumps(train_metrics),
                    json.dumps(val_metrics) if val_metrics is not None else None,
                    lr,
                    train_time_s,
                    eval_time_s,
                    _now_iso(),
                ),
            )

    def update_best(self, best_metric: float, best_epoch: int) -> None:
        with _open_rw(self._db_path) as c:
            c.execute(
                f"UPDATE {_T_RUNS} SET best_metric=?, best_epoch=? WHERE run_id=?",
                (best_metric, best_epoch, self.run_id),
            )

    def _set_status(self, status: RunStatus) -> None:
        """By MetricsTracker.run() context manager only."""
        with _open_rw(self._db_path) as c:
            c.execute(
                f"UPDATE {_T_RUNS} SET finished_at=?, status=? WHERE run_id=?",
                (_now_iso(), status.value, self.run_id),
            )


class MetricsTracker:
    """训练 run 的工厂 — `run()` context manager 负责整条生命周期.

    用法:
        tracker = MetricsTracker()
        with tracker.run('PopTrainer', config={...}, note='xxx') as handle:
            for epoch in range(N):
                handle.log_epoch(epoch, ...)
                if better: handle.update_best(auc, epoch)
        # 退出 with-block: status=DONE 自动; 若抛异常 = CRASHED
    """

    def __init__(self, db_path: str | Path = _DEFAULT_DB) -> None:
        self.db_path = Path(db_path)
        # _open_rw 内 executescript 已建表; with-empty-body 确保创完即关连接
        with _open_rw(self.db_path):
            pass

    @contextmanager
    def run(
        self,
        trainer: str,
        *,
        config: dict[str, Any],
        note: str | None = None,
        n_params: int | None = None,
        metric_key: str | None = None,
    ) -> Iterator[RunHandle]:
        """创建一个 run, 产出 RunHandle. 退出时自动 finish(DONE/CRASHED).

        run_id 格式: `{trainer}_{YYYYMMDD-HHMMSS}_{6hex}`.
        """
        ts_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = f"{trainer}_{ts_tag}_{secrets.token_hex(3)}"
        with _open_rw(self.db_path) as c:
            c.execute(
                f"INSERT INTO {_T_RUNS} "
                "(run_id, trainer, started_at, status, n_params, metric_key, "
                " config_json, note, git_sha, host) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    trainer,
                    _now_iso(),
                    RunStatus.RUNNING.value,
                    n_params,
                    metric_key,
                    json.dumps(config, default=str),
                    note,
                    _git_sha(),
                    socket.gethostname(),
                ),
            )
        handle = RunHandle(run_id=run_id, _db_path=self.db_path)
        status = RunStatus.DONE
        try:
            yield handle
        except BaseException:
            status = RunStatus.CRASHED
            raise
        finally:
            handle._set_status(status)


# --- Query helpers (CLI 用) ---------------------------------------------------


def list_runs(
    db_path: str | Path = _DEFAULT_DB,
    *,
    trainer: str | None = None,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """列出最近 N 个 run — 仅 runs 表 (不拉 metrics). db 不存在 → 空."""
    with _open_ro(Path(db_path)) as c:
        if c is None:
            return []
        q = (
            "SELECT run_id, trainer, started_at, finished_at, status, "
            "n_params, metric_key, best_metric, best_epoch, note, git_sha "
            f"FROM {_T_RUNS}"
        )
        params: tuple[Any, ...] = ()
        if trainer:
            q += " WHERE trainer=?"
            params = (trainer,)
        q += " ORDER BY started_at DESC LIMIT ?"
        params += (limit,)
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute(q, params).fetchall()]


def get_run(run_id: str, db_path: str | Path = _DEFAULT_DB) -> dict[str, Any] | None:
    """按 run_id 查 run 头 (不含 epoch metrics). None = 不存在."""
    with _open_ro(Path(db_path)) as c:
        if c is None:
            return None
        c.row_factory = sqlite3.Row
        r = c.execute(
            f"SELECT * FROM {_T_RUNS} WHERE run_id=?",
            (run_id,),
        ).fetchone()
        return dict(r) if r is not None else None


def get_metrics(
    run_id: str,
    db_path: str | Path = _DEFAULT_DB,
) -> list[dict[str, Any]]:
    """拉某 run 的所有 epoch 指标. db 不存在 → 空."""
    with _open_ro(Path(db_path)) as c:
        if c is None:
            return []
        c.row_factory = sqlite3.Row
        rows = c.execute(
            f"SELECT * FROM {_T_METRICS} WHERE run_id=? ORDER BY epoch",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def set_note(run_id: str, note: str, db_path: str | Path = _DEFAULT_DB) -> bool:
    with _open_ro(Path(db_path)) as c:
        if c is None:
            return False
        cur = c.execute(
            f"UPDATE {_T_RUNS} SET note=? WHERE run_id=?",
            (note, run_id),
        )
        return cur.rowcount > 0


def delete_run(run_id: str, db_path: str | Path = _DEFAULT_DB) -> bool:
    """删除 run + 其 metrics (cascade 手动)."""
    with _open_ro(Path(db_path)) as c:
        if c is None:
            return False
        c.execute(f"DELETE FROM {_T_METRICS} WHERE run_id=?", (run_id,))
        cur = c.execute(f"DELETE FROM {_T_RUNS} WHERE run_id=?", (run_id,))
        return cur.rowcount > 0
