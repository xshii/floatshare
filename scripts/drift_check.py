"""读 metrics.db 对比 daily warm 跟 anchor 的 best AUC, 掉 >5% 就 Bark 推告警.

约定:
    - Anchor run 的 note 必须含 "anchor" 字样 (e.g. `--note "cold anchor"`)
    - Daily warm run 的 note 必须含 "daily warm" 字样 (daily-train.sh 自动加)
    - 只看 PopTrainer, status=done, 按 started_at DESC 各取最新 1 条

退出码:
    0 = 成功 (可能推了告警, 也可能没掉)
    1 = anchor / warm 缺 (首次没基线, daily-train 继续别崩)
    2 = 数据库异常
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Final

from floatshare.observability import logger, notify

_DB: Final = Path("data/ml/metrics.db")
_DRIFT_PCT_THRESHOLD: Final = 5.0  # 掉 >5% 就告警


def _load_latest(db: sqlite3.Connection, note_like: str) -> sqlite3.Row | None:
    return db.execute(
        "SELECT run_id, best_metric, started_at FROM training_runs "
        "WHERE trainer='PopTrainer' AND status='done' AND note LIKE ? "
        "ORDER BY started_at DESC LIMIT 1",
        (f"%{note_like}%",),
    ).fetchone()


def main() -> int:
    if not _DB.exists():
        logger.warning(f"[drift] {_DB} 不存在, 跳过 (首次 training 没启动过)")
        return 1

    with sqlite3.connect(_DB) as db:
        db.row_factory = sqlite3.Row
        anchor = _load_latest(db, "anchor")
        warm = _load_latest(db, "daily warm")

    if anchor is None or warm is None:
        logger.info(f"[drift] anchor={anchor} warm={warm}, 缺一跳过 (正常首次无基线)")
        return 1

    anchor_auc = float(anchor["best_metric"] or 0.0)
    warm_auc = float(warm["best_metric"] or 0.0)
    if anchor_auc <= 0:
        logger.warning("[drift] anchor best_metric=0, 异常 anchor 跳过")
        return 2

    drop_pct = (anchor_auc - warm_auc) / anchor_auc * 100
    logger.info(f"[drift] anchor={anchor_auc:.3f} warm={warm_auc:.3f} drop={drop_pct:+.1f}%")
    if drop_pct > _DRIFT_PCT_THRESHOLD:
        notify(
            title="⚠ daily warm drift",
            body=(
                f"anchor AUC={anchor_auc:.3f} warm AUC={warm_auc:.3f} "
                f"(-{drop_pct:.1f}% > {_DRIFT_PCT_THRESHOLD}%) · 考虑早 cold-start · "
                f"run={warm['run_id']}"
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
