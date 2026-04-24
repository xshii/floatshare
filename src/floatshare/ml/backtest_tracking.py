"""回测记录持久化 — 跟 ml/tracking.py 并列, 共享 metrics.db.

目的: 回测结果 (Sharpe / 总收益 / MaxDD / 策略参数) 入库, 后续能:
    - `floatshare-bt-runs list` 横向对比不同策略 / 时段的表现
    - linked_run_id 把 backtest 和产生 ckpt 的 training_run 串起来

Schema:
    backtest_runs(
        run_id PK, started_at, finished_at, status,
        strategy, codes_count, window_start, window_end, capital,
        total_return, cagr, sharpe, max_drawdown, volatility, win_rate,
        metrics_json, strategy_params_json,
        linked_run_id, note, git_sha, host,
    )

run_id 格式: `bt_{strategy}_{YYYYMMDD-HHMMSS}_{6hex}`.
"""

from __future__ import annotations

import json
import secrets
import socket
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from floatshare.ml.tracking import (
    _DEFAULT_DB,
    RunStatus,
    _git_sha,
    _now_iso,
)
from floatshare.ml.tracking import (
    _open_ro as _tracking_open_ro,
)
from floatshare.ml.tracking import (
    _open_rw as _tracking_open_rw,
)

# 表名常量 — 避免散落 SQL 里; rename 时一处改
_T_BACKTESTS: Final[str] = "backtest_runs"

_BT_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {_T_BACKTESTS} (
    run_id                 TEXT PRIMARY KEY,
    started_at             TEXT NOT NULL,
    finished_at            TEXT,
    status                 TEXT NOT NULL DEFAULT '{RunStatus.RUNNING.value}',
    strategy               TEXT NOT NULL,
    codes_count            INTEGER,
    window_start           TEXT,
    window_end             TEXT,
    capital                REAL,
    total_return           REAL,
    cagr                   REAL,
    sharpe                 REAL,
    max_drawdown           REAL,
    volatility             REAL,
    win_rate               REAL,
    metrics_json           TEXT,
    strategy_params_json   TEXT,
    linked_run_id          TEXT,
    note                   TEXT,
    git_sha                TEXT,
    host                   TEXT
);

CREATE INDEX IF NOT EXISTS idx_bt_strategy_time
    ON {_T_BACKTESTS}(strategy, started_at DESC);
"""


@contextmanager
def _open_rw(db_path: Path) -> Iterator[sqlite3.Connection]:
    """在 ml/tracking._open_rw 基础上叠加 BT schema — 两类表共库共连接."""
    with _tracking_open_rw(db_path) as c:
        c.executescript(_BT_SCHEMA)
        yield c


@contextmanager
def _open_ro(db_path: Path) -> Iterator[sqlite3.Connection | None]:
    with _tracking_open_ro(db_path) as c:
        if c is not None:
            c.executescript(_BT_SCHEMA)  # 幂等补表 (旧 db 无 BT 表时)
        yield c


@dataclass(slots=True)
class BacktestHandle:
    """单次 backtest 的 run 句柄 — 通过 `BacktestTracker.run()` context 取得.

    `log_result(metrics_dict)` 写入指标; 退出 with-block 自动 DONE/CRASHED.
    """

    run_id: str
    _db_path: Path = field(repr=False)

    def log_result(
        self,
        *,
        total_return: float,
        cagr: float,
        sharpe: float,
        max_drawdown: float,
        volatility: float,
        win_rate: float,
        all_metrics: dict[str, Any],
    ) -> None:
        """写入 6 个 promote 列 + 整份 metrics JSON (供未来扩展查询)."""
        with _open_rw(self._db_path) as c:
            c.execute(
                f"UPDATE {_T_BACKTESTS} SET "
                "total_return=?, cagr=?, sharpe=?, max_drawdown=?, "
                "volatility=?, win_rate=?, metrics_json=? "
                "WHERE run_id=?",
                (
                    total_return,
                    cagr,
                    sharpe,
                    max_drawdown,
                    volatility,
                    win_rate,
                    json.dumps(all_metrics, default=str),
                    self.run_id,
                ),
            )

    def _set_status(self, status: RunStatus) -> None:
        """By BacktestTracker.run() context manager only."""
        with _open_rw(self._db_path) as c:
            c.execute(
                f"UPDATE {_T_BACKTESTS} SET finished_at=?, status=? WHERE run_id=?",
                (_now_iso(), status.value, self.run_id),
            )


class BacktestTracker:
    """回测 run 工厂 — 跟 MetricsTracker 一个模式."""

    def __init__(self, db_path: str | Path = _DEFAULT_DB) -> None:
        self.db_path = Path(db_path)
        with _open_rw(self.db_path):  # 建表
            pass

    @contextmanager
    def run(
        self,
        strategy: str,
        *,
        codes_count: int,
        window_start: str,
        window_end: str,
        capital: float,
        strategy_params: dict[str, Any] | None = None,
        note: str | None = None,
        linked_run_id: str | None = None,
    ) -> Iterator[BacktestHandle]:
        ts_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = f"bt_{strategy}_{ts_tag}_{secrets.token_hex(3)}"
        with _open_rw(self.db_path) as c:
            c.execute(
                f"INSERT INTO {_T_BACKTESTS} "
                "(run_id, started_at, status, strategy, codes_count, "
                " window_start, window_end, capital, strategy_params_json, "
                " linked_run_id, note, git_sha, host) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    _now_iso(),
                    RunStatus.RUNNING.value,
                    strategy,
                    codes_count,
                    window_start,
                    window_end,
                    capital,
                    json.dumps(strategy_params, default=str) if strategy_params else None,
                    linked_run_id,
                    note,
                    _git_sha(),
                    socket.gethostname(),
                ),
            )
        handle = BacktestHandle(run_id=run_id, _db_path=self.db_path)
        status = RunStatus.DONE
        try:
            yield handle
        except BaseException:
            status = RunStatus.CRASHED
            raise
        finally:
            handle._set_status(status)


# --- Query helpers (CLI 用) ---------------------------------------------------


def list_backtests(
    db_path: str | Path = _DEFAULT_DB,
    *,
    strategy: str | None = None,
    limit: int = 30,
) -> list[dict[str, Any]]:
    with _open_ro(Path(db_path)) as c:
        if c is None:
            return []
        q = f"SELECT * FROM {_T_BACKTESTS}"
        params: tuple[Any, ...] = ()
        if strategy:
            q += " WHERE strategy=?"
            params = (strategy,)
        q += " ORDER BY started_at DESC LIMIT ?"
        params += (limit,)
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute(q, params).fetchall()]


def get_backtest(run_id: str, db_path: str | Path = _DEFAULT_DB) -> dict[str, Any] | None:
    with _open_ro(Path(db_path)) as c:
        if c is None:
            return None
        c.row_factory = sqlite3.Row
        r = c.execute(f"SELECT * FROM {_T_BACKTESTS} WHERE run_id=?", (run_id,)).fetchone()
        return dict(r) if r is not None else None


def delete_backtest(run_id: str, db_path: str | Path = _DEFAULT_DB) -> bool:
    with _open_ro(Path(db_path)) as c:
        if c is None:
            return False
        cur = c.execute(f"DELETE FROM {_T_BACKTESTS} WHERE run_id=?", (run_id,))
        return cur.rowcount > 0
