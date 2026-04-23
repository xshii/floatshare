#!/usr/bin/env python
"""为 web 同步健康热图建 (trade_date, code) 覆盖索引。

为什么:
    daily_status_cells 需要按 trade_date 范围统计每天 distinct code 数。
    现有 PK autoindex 是 (code, trade_date) — code 在前, 帮不上日期范围查询。
    新建 (trade_date, code) 覆盖索引后:
        - 索引-only 扫描 (不回表)
        - 6 表 × 7 天 ≈ 38K 条索引项
        - 单表 <50ms, 6 表并行 <150ms (从 30s+ 降到 <1s)

代价:
    每表 ~300-500MB 索引, 6 表合计 ~2GB。
    创建过程占用 2x 临时空间, 60-180s 完成 (按表大小)。

幂等: CREATE INDEX IF NOT EXISTS, 重复跑无副作用。
"""

from __future__ import annotations

import time
from pathlib import Path

from sqlalchemy import create_engine, text

DB_PATH = Path("data/floatshare.db")

TABLES: tuple[str, ...] = (
    "raw_daily",
    "adj_factor",
    "daily_basic",
    "moneyflow",
    "margin_detail",
    "chip_perf",
)


def main() -> None:
    if not DB_PATH.exists():
        print(f"❌ 数据库不存在: {DB_PATH}")
        return

    engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"timeout": 60.0})
    print(f"📦 DB: {DB_PATH}  ({DB_PATH.stat().st_size / 1024**3:.1f}G)\n")

    with engine.connect() as conn:
        # WAL + 大缓存, 加速 CREATE INDEX
        conn.execute(text("PRAGMA journal_mode=WAL"))
        conn.execute(text("PRAGMA cache_size=-200000"))  # 200MB page cache
        conn.execute(text("PRAGMA temp_store=MEMORY"))

        for table in TABLES:
            idx_name = f"idx_{table}_date_code"
            print(f"⏳ {idx_name} ON {table}(trade_date, code) ...", end=" ", flush=True)
            t0 = time.time()
            try:
                conn.execute(
                    text(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}(trade_date, code)")
                )
                conn.commit()
                print(f"✅ {time.time() - t0:5.1f}s")
            except Exception as exc:
                print(f"❌ {exc}")

        # ANALYZE 让查询规划器知道新索引
        print("\n⏳ ANALYZE ...", end=" ", flush=True)
        t0 = time.time()
        conn.execute(text("ANALYZE"))
        conn.commit()
        print(f"✅ {time.time() - t0:5.1f}s")

    print(f"\n📦 完成。新 DB 大小: {DB_PATH.stat().st_size / 1024**3:.1f}G")


if __name__ == "__main__":
    main()
