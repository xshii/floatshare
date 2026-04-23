#!/usr/bin/env python
"""一次性数据治理: trade_date 字符串格式归一为 ISO 8601 ('YYYY-MM-DDTHH:MM:SS')。

# Why
早期 sync 把 pd.Timestamp 直接作 SQL bind, sqlalchemy `str()` 出空格分隔
('YYYY-MM-DD HH:MM:SS'); 后来代码改用 `.isoformat()` 出带 T 格式。两种格式
被 SQLite 当作不同字符串 PK, 同一天可能存两行，pandas 解析后变重复键。

# How
per-table 在单事务内:
  1. INSERT OR IGNORE 把 with-space 行用 with-T 的 PK 重新插入 (已存在则跳过)
  2. DELETE 所有 with-space 行
  3. ANALYZE 让优化器更新统计

# Safety
- 全程在 SQLite 显式事务内, 失败 rollback, 单表原子
- **不做物理备份**: db 21G + 磁盘紧张, copy 不现实; 事务回滚已是足够保护
- 跑前必须确保 sync 不在写这 4 个表
- 跑完不会自动 VACUUM (需要 ~2x 临时空间), 物理收紧请手动 `VACUUM INTO`
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

DB_PATH = Path("data/floatshare.db")
DIRTY_TABLES = ("raw_daily", "adj_factor", "chip_perf", "chip_dist")


def normalize_table(con: sqlite3.Connection, table: str) -> tuple[int, int]:
    """单表事务内 normalize, 返回 (插入的新 with-T 行数, 删除的 with-space 行数)。"""
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()]
    if "trade_date" not in cols:
        return (0, 0)
    cols_csv = ", ".join(cols)
    select_csv = ", ".join(
        "REPLACE(trade_date, ' ', 'T') AS trade_date" if c == "trade_date" else c for c in cols
    )
    inserted = con.execute(
        f"INSERT OR IGNORE INTO {table} ({cols_csv}) "
        f"SELECT {select_csv} FROM {table} WHERE trade_date LIKE '% %'"
    ).rowcount
    deleted = con.execute(f"DELETE FROM {table} WHERE trade_date LIKE '% %'").rowcount
    return (inserted, deleted)


def main() -> None:
    if not DB_PATH.exists():
        sys.exit(f"DB not found: {DB_PATH}")

    con = sqlite3.connect(DB_PATH, timeout=60, isolation_level=None)
    con.execute("PRAGMA journal_mode = WAL")
    con.execute("PRAGMA busy_timeout = 60000")

    print(f"{'TABLE':<14} {'插入(with-T)':>14}  {'删除(with-space)':>18}  耗时")
    print("-" * 60)
    grand_total_ins, grand_total_del = 0, 0
    for table in DIRTY_TABLES:
        t0 = time.time()
        con.execute("BEGIN IMMEDIATE")
        try:
            ins, dele = normalize_table(con, table)
            con.execute("COMMIT")
        except Exception as exc:
            con.execute("ROLLBACK")
            print(f"❌ {table}: {exc}")
            continue
        elapsed = time.time() - t0
        grand_total_ins += ins
        grand_total_del += dele
        print(f"{table:<14} {ins:>14,}  {dele:>18,}  {elapsed:5.1f}s")

    print("-" * 60)
    print(f"{'TOTAL':<14} {grand_total_ins:>14,}  {grand_total_del:>18,}")
    print("\n🔧 ANALYZE 更新统计…")
    con.execute("ANALYZE")
    con.close()
    print("✅ 完成。物理收紧用: sqlite3 data/floatshare.db 'VACUUM'  (需 ~2x 空间)")


if __name__ == "__main__":
    main()
