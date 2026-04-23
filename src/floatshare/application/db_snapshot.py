"""DB 状态 snapshot — 把昂贵的 DB 聚合查询 (table COUNT(*)) 写到文件供 web 秒读。

为什么独立模块:
- cli/run_sync.py 在 sync 完成时调 `refresh_counts_snapshot()` 写文件
- web/data.py 的 `table_counts()` 优先读文件，避免 25s 阻塞
- 两者都依赖 application 层 (cli 和 web 不能互相 import)
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from floatshare.domain.records import ALL_RECORDS

if TYPE_CHECKING:
    from floatshare.infrastructure.storage.database import DatabaseStorage


COUNTS_SNAPSHOT_PATH = Path("logs/table-counts.json")


def refresh_counts_snapshot(db: DatabaseStorage) -> list[dict]:
    """重算所有 ALL_RECORDS 表的行数，写到 JSON snapshot。返回 rows。

    阻塞执行 (大表 ~30s)，调用方应在合适时机调 (如 sync 完成后)。
    """
    from sqlalchemy import text

    tables = sorted({r.TABLE for r in ALL_RECORDS if hasattr(r, "TABLE")})
    rows: list[dict] = []
    with db.engine.connect() as conn:
        for t in tables:
            try:
                n = conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar() or 0
            except Exception:
                n = 0
            rows.append({"表": t, "行数": f"{n:,}"})

    with contextlib.suppress(Exception):
        COUNTS_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = COUNTS_SNAPSHOT_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        tmp.replace(COUNTS_SNAPSHOT_PATH)
    return rows


def read_counts_snapshot() -> list[dict] | None:
    """读 snapshot — 文件不存在或损坏返回 None。"""
    if not COUNTS_SNAPSHOT_PATH.exists():
        return None
    try:
        return json.loads(COUNTS_SNAPSHOT_PATH.read_text())
    except Exception:
        return None
