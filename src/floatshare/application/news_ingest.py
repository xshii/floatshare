"""新闻联播 ingest orchestration — T 日 20:00 触发.

流程:
    1. 调 tushare cctv_news(date=T) 拉 T 日联播文字稿
    2. concat 所有 content → full_text
    3. infrastructure.nlp.extract_industry_mentions(full_text) → list[IndustryMention]
    4. UPSERT cctv_news_raw (原始) + cctv_news_mentions (NLP 结果)

兜底 (Q5=b): tushare 拿不到 → 不填任何行 (模型当 mentioned=0). 不用 T-1 顶替.

retry 策略: caller (pipeline stage1) 应在 20:00 / 20:15 / 20:30 / 20:45 依次重试,
22:00 (stage2 开始) 前必须成功; 失败则告警但 pipeline 继续 (news 特征全 0 等价于
"今天没新闻", 模型依然能跑).
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING

from floatshare.domain.records import CctvNewsMention, CctvNewsRaw
from floatshare.infrastructure.nlp import extract_industry_mentions
from floatshare.infrastructure.storage import schema_sql
from floatshare.observability import logger

if TYPE_CHECKING:
    import pandas as pd

    from floatshare.infrastructure.data_sources.tushare import TushareSource


@dataclass(frozen=True, slots=True)
class IngestResult:
    """单次 ingest 的返回 — 用于 pipeline 判断是否成功 + 监控."""

    trade_date: str
    raw_rows: int  # 原始 cctv_news 行数 (章节数)
    mentions: int  # NLP 匹配出的 (l1_code) 数
    text_length: int  # 全文字符数 (观察异常)
    success: bool  # tushare 拉到 + NLP 执行完
    error: str | None = None


def ingest_cctv_news_for_date(
    trade_date: date,
    source: TushareSource,
    db_path: str,
) -> IngestResult:
    """执行一次 T 日新闻联播 ingest. 失败返回 success=False + error 信息."""
    td_str = trade_date.strftime("%Y-%m-%d")
    try:
        raw_df = source.get_cctv_news_by_date(trade_date)
    except Exception as exc:
        logger.warning(f"cctv_news T={td_str} 拉取失败: {exc}")
        return IngestResult(td_str, 0, 0, 0, success=False, error=str(exc))

    if raw_df.empty:
        # tushare 当日未入库 (常见于 20:00-20:15 过早调用). Caller 应 retry.
        return IngestResult(td_str, 0, 0, 0, success=False, error="empty_response")

    full_text = _concat_content(raw_df)
    mentions = extract_industry_mentions(full_text)

    _persist(db_path, trade_date=td_str, raw_df=raw_df, mentions=mentions)
    logger.info(
        f"cctv_news T={td_str}: raw_rows={len(raw_df)} text_len={len(full_text)} "
        f"mentions={len(mentions)} l1s={[m.l1_code for m in mentions]}",
    )
    return IngestResult(
        trade_date=td_str,
        raw_rows=len(raw_df),
        mentions=len(mentions),
        text_length=len(full_text),
        success=True,
    )


def _concat_content(raw_df: pd.DataFrame) -> str:
    """按 tushare 返回的行序 concat 所有 content, 章节之间用换行分隔."""
    if "content" not in raw_df.columns:
        return ""
    parts = [str(c) for c in raw_df["content"].tolist() if c]
    return "\n".join(parts)


def _persist(
    db_path: str,
    *,
    trade_date: str,
    raw_df: pd.DataFrame,
    mentions: list,
) -> None:
    """原子写 cctv_news_raw + cctv_news_mentions (先删当日旧记录再插)."""
    now = datetime.now().isoformat()
    # mypy 对 schema_sql 的 Protocol[_RecordSchema] 做类型窄化, 对 ClassVar-based
    # 的 dataclass record 类推断不够深 — 这里 4 处 type: ignore 是已知 Protocol 误报.
    raw_ddl = schema_sql.ddl(CctvNewsRaw)  # type: ignore[arg-type]
    mention_ddl = schema_sql.ddl(CctvNewsMention)  # type: ignore[arg-type]
    raw_upsert = schema_sql.upsert_sql(CctvNewsRaw)  # type: ignore[arg-type]
    mention_upsert = schema_sql.upsert_sql(CctvNewsMention)  # type: ignore[arg-type]

    with sqlite3.connect(db_path) as conn:
        # 建表 (幂等)
        conn.execute(raw_ddl)
        conn.execute(mention_ddl)
        # 迁移: v2 新增 weighted_score 列; 已有库 ALTER 加列, 新库 DDL 已含此列
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute("ALTER TABLE cctv_news_mentions ADD COLUMN weighted_score REAL")
        # 清当日旧记录 (允许重跑)
        conn.execute("DELETE FROM cctv_news_raw WHERE trade_date = ?", (trade_date,))
        conn.execute("DELETE FROM cctv_news_mentions WHERE trade_date = ?", (trade_date,))

        # 插原始
        for seq, row in enumerate(raw_df.itertuples(index=False), start=1):
            conn.execute(
                raw_upsert,
                {
                    "trade_date": trade_date,
                    "seq": seq,
                    "title": getattr(row, "title", None),
                    "content": getattr(row, "content", None),
                    "ingested_at": now,
                },
            )
        # 插 mentions
        for m in mentions:
            conn.execute(
                mention_upsert,
                {
                    "trade_date": trade_date,
                    "l1_code": m.l1_code,
                    "mentioned": 1,
                    "match_score": m.score,
                    "weighted_score": getattr(m, "weighted_score", None),
                    "matched_keywords": json.dumps(m.matched_keywords, ensure_ascii=False),
                    "news_source": "tushare_cctv",
                    "ingested_at": now,
                },
            )
        conn.commit()
