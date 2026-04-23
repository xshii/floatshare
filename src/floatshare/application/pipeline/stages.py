"""Application 层的 stage fn 集合 — 仅数据层 stages (不依赖 ml/).

每个 stage fn 签名统一: (StageContext) → dict[str, Any] (counter extras)
- ctx.shared 存跨 stage 对象 (feats / panel / codes) — ml 层的 stage fn 会填
- fn 只读 DB 或走 subprocess, 不做 record_counter (由 runner 统一埋点)

ML 相关 stages (S_prep / S2b / S3) 在 floatshare.ml.pipeline_stages — CLI 层组装.
"""

from __future__ import annotations

import sqlite3
import subprocess
import time
from datetime import date
from typing import Any

from floatshare.domain.pipeline import StageContext

# 兼容导出: 老代码 `from floatshare.application.pipeline.stages import StageContext`
__all__ = [
    "StageContext",
    "stage_s1_sync",
    "stage_s1c_news_ingest",
    "stage_s2a_db_integrity",
]


# === S1 sync =================================================================


def stage_s1_sync(ctx: StageContext) -> dict[str, Any]:
    """subprocess 拉 floatshare-sync 把 T 日 4 张主表 sync 到本地 DB.

    幂等 (watermark 自动识别); 跑失败时 returncode != 0, 由 runner 判 fail.
    """
    cmd = [
        "floatshare-sync",
        "--by-date",
        "--start",
        ctx.trade_date,
        "--end",
        ctx.trade_date,
        "--include",
        "raw_daily",
        "daily_basic",
        "moneyflow",
        "adj_factor",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, check=False)
    with sqlite3.connect(str(ctx.db_path)) as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM raw_daily WHERE trade_date LIKE ?",
            (f"{ctx.trade_date}%",),
        ).fetchone()[0]
    if r.returncode != 0:
        raise RuntimeError(f"floatshare-sync 失败 rc={r.returncode}: {r.stderr[-500:]}")
    return {"returncode": r.returncode, "raw_daily_rows_today": n}


# === S1c cctv_news ingest ====================================================


def stage_s1c_news_ingest(
    ctx: StageContext,
    max_attempts: int = 3,
    retry_interval_sec: int = 60,
) -> dict[str, Any]:
    """S1C: cctv_news ingest + NLP industry mentions.

    调度假设 pipeline 在 T 日 20:00 触发 (launchd plist), tushare cctv_news
    通常此时已发布. 留 3 次 * 60s retry 作缓冲 (实测 2026-04-22 21:00 已 8 条齐备).
    全失败 → RuntimeError, runner 按 SOFT 记 FAIL, 不阻断 S4 (news 特征全 0 = 今日无新闻).

    测试用 monkeypatch 替换 ingest_cctv_news_for_date / time.sleep (避免网络 + 真 sleep).
    """
    from floatshare.application.news_ingest import ingest_cctv_news_for_date
    from floatshare.infrastructure.data_sources.tushare import TushareSource

    source = TushareSource()
    td = date.fromisoformat(ctx.trade_date)

    last_err = "no attempt"
    for attempt in range(1, max_attempts + 1):
        result = ingest_cctv_news_for_date(td, source, str(ctx.db_path))
        if result.success:
            return {
                "attempts": attempt,
                "raw_rows": result.raw_rows,
                "mentions": result.mentions,
                "text_length": result.text_length,
            }
        last_err = result.error or "unknown"
        if attempt < max_attempts:
            time.sleep(retry_interval_sec)
    raise RuntimeError(
        f"cctv_news T={ctx.trade_date} {max_attempts} 次尝试均失败, last_err={last_err}"
    )


# === S2a DB integrity ========================================================


def stage_s2a_db_integrity(ctx: StageContext) -> dict[str, Any]:
    from floatshare.application.db_integrity import (
        DbIntegrityError,
        check_cross_table_alignment,
        check_daily_row_count_stability,
        check_trade_date_duplicates,
    )

    dups = check_trade_date_duplicates(str(ctx.db_path))
    cross = check_cross_table_alignment(str(ctx.db_path))
    jumps = check_daily_row_count_stability(str(ctx.db_path))
    extras = {
        "duplicates": len(dups),
        "cross_missing_pairs": sum(m.missing_pairs for m in cross),
        "row_count_jumps": len(jumps),
    }
    if dups or cross or jumps:
        # 让 runner fail 本 stage (FAST 策略会中止 pipeline)
        raise DbIntegrityError(dups or cross or jumps)
    return extras
