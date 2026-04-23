"""生成行业 IDF baseline — 支持全局单文件 + 月度 PIT (point-in-time) 两种模式.

用途:
    - extract_industry_mentions 读 baseline 算 IDF 权重, 冷门行业被提及时加大权重
    - PIT 版本避免 look-ahead: 2018 年的 mention 用 2018 年之前的 baseline, 不偷看 2026 的行业热度
    - 版本化 + 每月重算, 跟踪信号基线漂移

两种模式:

  [默认] 全局单文件 (向后兼容):
    python scripts/build_news_baseline.py --days 90
    → data/news/industry_baseline.json  (一个文件, 用最近 90 天)

  [PIT 推荐] 月度 baseline:
    python scripts/build_news_baseline.py --per-month --start 2018-01 --end 2026-04
    → data/news/baselines/YYYY-MM.json × N
    每个月的 baseline 用该月前 `--days` 天数据 (月前 90 天), 完全 PIT

  数据源:
    - 默认读 DB cctv_news_raw (快, 需要先跑 backfill_cctv_mentions.py)
    - --no-db 强制走 tushare API (慢, 用于首次 bootstrap)
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from floatshare.infrastructure.nlp import load_industry_keywords

if TYPE_CHECKING:
    pass


def _db_text_loader(db_path: str) -> Callable[[date], str | None]:
    """返回一个 (date) → text | None 的读函数, 从 cctv_news_raw concat 当日 content."""

    def _load(d: date) -> str | None:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT content FROM cctv_news_raw WHERE trade_date = ? ORDER BY seq",
                (d.isoformat(),),
            ).fetchall()
        if not rows:
            return None
        return "\n".join(str(r[0]) for r in rows if r[0])

    return _load


def _tushare_text_loader() -> Callable[[date], str | None]:
    """fallback: 直拉 tushare (慢, 仅 bootstrap 时用)."""
    from floatshare.infrastructure.data_sources.tushare import TushareSource

    ts = TushareSource()

    def _load(d: date) -> str | None:
        df = ts.get_cctv_news_by_date(d)
        if df.empty:
            return None
        return "\n".join(str(c) for c in df["content"].tolist() if c)

    return _load


def _compute_baseline_over_window(
    start: date,
    end: date,
    load_text: Callable[[date], str | None],
) -> dict[str, object]:
    """在 [start..end] 区间逐日统计 L1 命中, 返回 baseline dict (含 _meta + industries)."""
    kd = load_industry_keywords()

    days_with_data = 0
    days_mentioned: Counter[str] = Counter()
    mentions_total: Counter[str] = Counter()
    empty_days: list[str] = []

    cur = start
    while cur <= end:
        full = load_text(cur)
        if full is None:
            empty_days.append(cur.isoformat())
            cur += timedelta(days=1)
            continue
        days_with_data += 1
        for l1_code, (_name, kws) in kd.entries.items():
            hits_this_day = sum(1 for kw in kws if kw in full)
            if hits_this_day > 0:
                days_mentioned[l1_code] += 1
                mentions_total[l1_code] += hits_this_day
        cur += timedelta(days=1)

    industries: dict[str, dict[str, object]] = {}
    for l1_code, (name, _kws) in kd.entries.items():
        dm = days_mentioned[l1_code]
        mt = mentions_total[l1_code]
        # 平滑 IDF: +1 避免 log(0), +1 提底. 数据极少 (days_with_data < 10) 时退化为 1.0
        idf = 1.0 if days_with_data < 10 else math.log(days_with_data / (dm + 1)) + 1.0
        industries[l1_code] = {
            "name": name,
            "days_mentioned": dm,
            "mentions_total": mt,
            "mentions_per_day": round(mt / max(days_with_data, 1), 3),
            "idf": round(idf, 4),
        }

    return {
        "_meta": {
            "source_start": start.isoformat(),
            "source_end": end.isoformat(),
            "source_days_total": (end - start).days + 1,
            "source_days_with_data": days_with_data,
            "source_empty_days_count": len(empty_days),
            "generated_at": date.today().isoformat(),
            "keyword_dict_version": "loaded_from_industry_keywords.json",
        },
        "industries": industries,
    }


def _month_range(start_ym: str, end_ym: str) -> list[date]:
    """返回 [start_ym, end_ym] 闭区间所有月份首日 (YYYY-MM → date(y, m, 1))."""
    ys, ms = start_ym.split("-")
    ye, me = end_ym.split("-")
    y, m = int(ys), int(ms)
    end_y, end_m = int(ye), int(me)
    out: list[date] = []
    while (y, m) <= (end_y, end_m):
        out.append(date(y, m, 1))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def _write_baseline(baseline: dict[str, object], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")


def _print_baseline_summary(baseline: dict[str, object], path: Path) -> None:
    meta = baseline["_meta"]
    ind = baseline["industries"]
    assert isinstance(ind, dict)
    assert isinstance(meta, dict)
    covered = sum(1 for v in ind.values() if v["days_mentioned"] > 0)
    print(
        f"  ✓ {path.name}: {meta['source_start']}..{meta['source_end']} "
        f"days={meta['source_days_with_data']}/{meta['source_days_total']} "
        f"L1_covered={covered}/{len(ind)}"
    )


def _run_global(args, load_text: Callable[[date], str | None]) -> None:
    end = date.fromisoformat(args.end)
    start = end - timedelta(days=args.days - 1)
    baseline = _compute_baseline_over_window(start, end, load_text)
    out = Path(args.output)
    _write_baseline(baseline, out)
    _print_baseline_summary(baseline, out)


def _run_per_month(args, load_text: Callable[[date], str | None]) -> None:
    months = _month_range(args.start, args.end)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== PIT baseline per-month: {len(months)} 个月份 ===")
    for m_start in months:
        # 窗口: [m_start - days, m_start - 1day] — 严格在本月之前, 避免 look-ahead
        window_end = m_start - timedelta(days=1)
        window_start = m_start - timedelta(days=args.days)
        baseline = _compute_baseline_over_window(window_start, window_end, load_text)
        out = out_dir / f"{m_start.strftime('%Y-%m')}.json"
        _write_baseline(baseline, out)
        _print_baseline_summary(baseline, out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=90, help="回看窗口长度 (默认 90 天)")
    p.add_argument("--end", type=str, default=date.today().isoformat(), help="[全局模式] 窗口末端")
    p.add_argument(
        "--output",
        type=str,
        default="data/news/industry_baseline.json",
        help="[全局模式] 输出路径",
    )
    p.add_argument("--per-month", action="store_true", help="启用月度 PIT baseline 模式")
    p.add_argument("--start", type=str, default=None, help="[月度模式] 起始月 YYYY-MM")
    p.add_argument(
        "--out-dir",
        type=str,
        default="data/news/baselines",
        help="[月度模式] 输出目录 (每月一个 JSON)",
    )
    p.add_argument("--db", type=str, default="data/floatshare.db", help="DB 路径")
    p.add_argument("--no-db", action="store_true", help="强制走 tushare API (慢, 首次 bootstrap)")
    args = p.parse_args()

    if args.per_month and not args.start:
        p.error("--per-month 必须提供 --start YYYY-MM")

    load_text = _tushare_text_loader() if args.no_db else _db_text_loader(args.db)

    if args.per_month:
        # 月度模式默认 end = 本月
        if not args.end or args.end == date.today().isoformat():
            today = date.today()
            args.end = f"{today.year:04d}-{today.month:02d}"
        _run_per_month(args, load_text)
    else:
        _run_global(args, load_text)


if __name__ == "__main__":
    main()
