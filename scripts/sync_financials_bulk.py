"""按 ann_date 批量同步 6 个财务表 — 替换 daily-sync 里 per-code 的慢路径.

旧 daily-sync: 5500 股 × 6 表 = ~27000 次 per-code API call ≈ 2h
新 bulk:       1 call 拉当天全市场公告 × 6 表 × N 天 ≈ 4s/天

用法:
    # 默认同步最近 7 天 (幂等, 重复跑无害)
    python scripts/sync_financials_bulk.py

    # 显式窗口
    python scripts/sync_financials_bulk.py --start 2026-04-16 --end 2026-04-23

    # 首次回补 (如 2026 初从 2023 一路补, 按 365 天 chunk 走)
    python scripts/sync_financials_bulk.py --start 2023-01-01 --end 2026-04-23

    # 只拉一部分表
    python scripts/sync_financials_bulk.py --include income balancesheet

需要 Tushare VIP (≥5000 积分), 否则 _vip 系列接口会抛 "token 不对" 错误.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd

from floatshare.domain.records import (
    Balancesheet,
    Cashflow,
    Dividend,
    FinaIndicator,
    Income,
    StkHolderNumber,
)
from floatshare.infrastructure.data_sources.tushare import TushareSource
from floatshare.infrastructure.storage.database import DatabaseStorage
from floatshare.observability import logger


@dataclass(frozen=True, slots=True)
class _BulkJob:
    """一个 bulk 同步 job 的全部配置."""

    name: str  # 日志名
    fetch: Callable[[TushareSource, date, date], pd.DataFrame]
    record_cls: type  # 落库 schema


# 6 个 job: 每个 job = 名字 + unbound method 引用 + DB record 类.
# 用 `TushareSource.get_*_bulk` 取未绑定方法, 调用时 fetch(ts, s, e) → ts 作 self.
# 不写 lambda 更干净; 重命名时 IDE 能自动跟. fetch 签名 = (ts, start, end) → DataFrame.
_JOBS: tuple[_BulkJob, ...] = (
    _BulkJob("income", TushareSource.get_income_bulk, Income),
    _BulkJob("balancesheet", TushareSource.get_balancesheet_bulk, Balancesheet),
    _BulkJob("cashflow", TushareSource.get_cashflow_bulk, Cashflow),
    _BulkJob("fina_indicator", TushareSource.get_fina_indicator_bulk, FinaIndicator),
    _BulkJob("stk_holder_number", TushareSource.get_holder_number_bulk, StkHolderNumber),
    _BulkJob("dividend", TushareSource.get_dividend_bulk, Dividend),
)


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    today = date.today()
    p.add_argument("--start", type=_parse_date, default=today - timedelta(days=7))
    p.add_argument("--end", type=_parse_date, default=today)
    p.add_argument(
        "--include",
        nargs="+",
        choices=[j.name for j in _JOBS],
        default=None,
        help=f"只同步指定表 (默认全部 6 个: {', '.join(j.name for j in _JOBS)})",
    )
    args = p.parse_args()

    jobs = _JOBS if args.include is None else tuple(j for j in _JOBS if j.name in args.include)

    ts = TushareSource()
    db = DatabaseStorage()
    logger.info(
        f"[bulk-financials] window=[{args.start}, {args.end}] ({(args.end - args.start).days + 1} 天), "
        f"{len(jobs)} 个表"
    )

    total_rows = 0
    total_time = 0.0
    for job in jobs:
        t0 = time.perf_counter()
        df = job.fetch(ts, args.start, args.end)
        dt_fetch = time.perf_counter() - t0
        if df.empty:
            logger.info(f"[{job.name}] 空 ({dt_fetch:.2f}s)")
            continue
        t0 = time.perf_counter()
        n_saved = db.save(job.record_cls, df)
        dt_save = time.perf_counter() - t0
        logger.info(
            f"[{job.name}] fetched {len(df)} → saved {n_saved} rows "
            f"(fetch {dt_fetch:.2f}s + save {dt_save:.2f}s)"
        )
        total_rows += n_saved
        total_time += dt_fetch + dt_save

    logger.info(f"[bulk-financials] 总计 {total_rows} rows in {total_time:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
