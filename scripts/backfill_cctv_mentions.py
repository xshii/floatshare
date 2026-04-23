"""回填 cctv_news_raw + cctv_news_mentions — 用当前 v3 关键词 + baseline 重算.

场景:
    - 关键词库升级后, 历史 mentions 表的 match_score / weighted_score / matched_keywords
      需要按新词库 / 新 baseline 重算 (否则训练时混用旧 NLP 结果)
    - 新接入训练, 需要从训练起点 (e.g. 2023-01-01) 把 cctv_news 全部拉回来

行为:
    1. 扫 [start..end] 日历天, 跳周末
    2. 对每一天:
         a. 若 cctv_news_raw 已有 → 直接用库里原始文稿 (省 tushare API)
         b. 若缺 → 从 tushare cctv_news 拉, 写 raw
         c. 用当前 keywords + baseline 重算 mentions, 写 mentions (DELETE+INSERT)
    3. 进度条 + 每日 summary

用法:
    python scripts/backfill_cctv_mentions.py --start 2023-01-01           # 从训练起点到今天
    python scripts/backfill_cctv_mentions.py --start 2026-01-01 --end 2026-04-22
    python scripts/backfill_cctv_mentions.py --start 2026-04-22 --dry-run  # 只看, 不写
    python scripts/backfill_cctv_mentions.py --rebuild-mentions-only       # 不拉 tushare, 仅用库里 raw 重算

每月 NLP review 工作流:
    1. 采样 1 周 corpus → 人眼看 / 跑 keyword audit
    2. 改 data/news/industry_keywords.json (词库 v(n+1))
    3. python scripts/build_news_baseline.py --days 90           # 重算 baseline
    4. python scripts/backfill_cctv_mentions.py --rebuild-mentions-only  # 回填 mentions
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

from floatshare.domain.records import CctvNewsMention, CctvNewsRaw
from floatshare.infrastructure.data_sources.tushare import TushareSource
from floatshare.infrastructure.nlp import (
    extract_industry_mentions,
    load_industry_baseline,
    load_industry_keywords,
)
from floatshare.infrastructure.storage import schema_sql
from floatshare.observability import logger

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True, slots=True)
class DayResult:
    """单日回填结果."""

    trade_date: str
    source: str  # 'db_cached' | 'tushare_fetch' | 'empty' | 'skip_weekend'
    raw_rows: int
    mentions: int


def _date_range(start: date, end: date) -> list[date]:
    """返回 [start..end] 区间所有日历日 — 联播周末也播, tushare 周末也入库 (实测 90/90)."""
    days = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def _load_raw_from_db(conn: sqlite3.Connection, trade_date: str) -> pd.DataFrame | None:
    """从 cctv_news_raw 读当日文稿 → DataFrame. 无数据返回 None."""
    import pandas as pd

    rows = conn.execute(
        "SELECT seq, title, content FROM cctv_news_raw WHERE trade_date = ? ORDER BY seq",
        (trade_date,),
    ).fetchall()
    if not rows:
        return None
    return pd.DataFrame(rows, columns=["seq", "title", "content"])


def _upsert_raw(conn: sqlite3.Connection, trade_date: str, raw_df: pd.DataFrame, now: str) -> None:
    """把 tushare 拉到的 raw_df 写 cctv_news_raw (DELETE + INSERT)."""
    upsert = schema_sql.upsert_sql(CctvNewsRaw)  # type: ignore[arg-type]
    conn.execute("DELETE FROM cctv_news_raw WHERE trade_date = ?", (trade_date,))
    for seq, row in enumerate(raw_df.itertuples(index=False), start=1):
        conn.execute(
            upsert,
            {
                "trade_date": trade_date,
                "seq": seq,
                "title": getattr(row, "title", None),
                "content": getattr(row, "content", None),
                "ingested_at": now,
            },
        )


def _upsert_mentions(conn: sqlite3.Connection, trade_date: str, mentions: list, now: str) -> None:
    """DELETE + INSERT 当日 mentions (带 weighted_score)."""
    upsert = schema_sql.upsert_sql(CctvNewsMention)  # type: ignore[arg-type]
    conn.execute("DELETE FROM cctv_news_mentions WHERE trade_date = ?", (trade_date,))
    for m in mentions:
        conn.execute(
            upsert,
            {
                "trade_date": trade_date,
                "l1_code": m.l1_code,
                "mentioned": 1,
                "match_score": m.score,
                "weighted_score": m.weighted_score,
                "matched_keywords": json.dumps(m.matched_keywords, ensure_ascii=False),
                "news_source": "tushare_cctv",
                "ingested_at": now,
            },
        )


def _process_day(
    conn: sqlite3.Connection,
    trade_date: date,
    ts: TushareSource | None,
    *,
    rebuild_only: bool,
    dry_run: bool,
) -> DayResult:
    td_str = trade_date.isoformat()
    now = datetime.now().isoformat()

    raw_df = _load_raw_from_db(conn, td_str)
    source = "db_cached"

    if raw_df is None:
        if rebuild_only:
            return DayResult(td_str, "empty", 0, 0)
        # 拉 tushare
        assert ts is not None
        try:
            raw_df = ts.get_cctv_news_by_date(trade_date)
        except Exception as exc:
            logger.warning(f"tushare cctv_news T={td_str} 拉取失败: {exc}")
            return DayResult(td_str, "empty", 0, 0)
        if raw_df.empty:
            return DayResult(td_str, "empty", 0, 0)
        source = "tushare_fetch"
        if not dry_run:
            _upsert_raw(conn, td_str, raw_df, now)

    # 重算 mentions — 传 for_date 触发 PIT baseline (避免 look-ahead)
    full_text = "\n".join(str(c) for c in raw_df["content"].tolist() if c)
    mentions = extract_industry_mentions(full_text, for_date=td_str)
    if not dry_run:
        _upsert_mentions(conn, td_str, mentions, now)

    return DayResult(td_str, source, len(raw_df), len(mentions))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=str, required=True, help="起始日期 YYYY-MM-DD")
    p.add_argument("--end", type=str, default=date.today().isoformat(), help="终止日期 (含)")
    p.add_argument(
        "--db",
        type=str,
        default="data/floatshare.db",
        help="DB path, 默认 data/floatshare.db",
    )
    p.add_argument(
        "--rebuild-mentions-only",
        action="store_true",
        help="只用库里 cctv_news_raw 重算 mentions (不碰 tushare)",
    )
    p.add_argument("--dry-run", action="store_true", help="只打印, 不写 DB")
    args = p.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    days = _date_range(start, end)

    print("=== CCTV news mentions backfill ===")
    print(f"range     : {args.start} .. {args.end} ({len(days)} days)")
    print(f"db        : {args.db}")
    print(f"mode      : {'rebuild-only' if args.rebuild_mentions_only else 'fetch+rebuild'}")
    print(f"dry-run   : {args.dry_run}")
    print(f"keywords  : {len(load_industry_keywords().entries)} L1 词库")
    print(f"baseline  : {len(load_industry_baseline())} L1 idf (全局 fallback)")
    print("           PIT 查 data/news/baselines/YYYY-MM.json 按月匹配 for_date")
    print()

    ts = None if args.rebuild_mentions_only else TushareSource()

    # 建表幂等 + 列迁移
    raw_ddl = schema_sql.ddl(CctvNewsRaw)  # type: ignore[arg-type]
    mention_ddl = schema_sql.ddl(CctvNewsMention)  # type: ignore[arg-type]
    with sqlite3.connect(args.db) as conn:
        conn.execute(raw_ddl)
        conn.execute(mention_ddl)
        import contextlib

        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute("ALTER TABLE cctv_news_mentions ADD COLUMN weighted_score REAL")

        # 逐天处理
        results: list[DayResult] = []
        for i, d in enumerate(days, start=1):
            r = _process_day(
                conn,
                d,
                ts,
                rebuild_only=args.rebuild_mentions_only,
                dry_run=args.dry_run,
            )
            results.append(r)
            if i % 20 == 0 or i == len(days):
                print(
                    f"  [{i:>4}/{len(days)}] T={r.trade_date} "
                    f"{r.source:<14} raw={r.raw_rows:>2} mentions={r.mentions}"
                )
        if not args.dry_run:
            conn.commit()

    # summary
    by_source: dict[str, int] = {}
    total_mentions = 0
    for r in results:
        by_source[r.source] = by_source.get(r.source, 0) + 1
        total_mentions += r.mentions
    print()
    print("=== Summary ===")
    for src, n in sorted(by_source.items()):
        print(f"  {src:<14} {n:>4} days")
    print(f"  total mentions written: {total_mentions}")
    if args.dry_run:
        print("  (dry-run: 未实际写 DB)")


if __name__ == "__main__":
    main()
