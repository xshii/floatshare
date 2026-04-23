"""新闻联播行业提及 (新闻联播文字稿经本地 NLP 匹配到 SW L1 行业).

两类表:
    CctvNewsRaw       — Tushare cctv_news 原始文本 (date, title, content)
                        保留原始数据便于 NLP 算法迭代时重新提取.
    CctvNewsMention   — NLP 匹配结果 (date, l1_code, mentioned, score, keywords)
                        feature 计算阶段只 join 这张, 原始表是冷备份.

时间语义 (与 features.py::_CctvNewsSource 对齐):
    T 日 19:30  联播播出结束
    T 日 20:15  tushare 服务端入库
    T 日 20:00+ ingest task 带 retry 拉取
    T 日 22:30  pipeline stage 4 训练/推理时 cutoff=20, shift_days=0, feats[T] = T 日 mention
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from floatshare.domain.records._meta import FieldSpec


@dataclass(frozen=True, slots=True)
class CctvNewsRaw:
    """新闻联播原始文字稿 (Tushare cctv_news 的冷备份)."""

    TABLE: ClassVar[str] = "cctv_news_raw"
    PK: ClassVar[tuple[str, ...]] = ("trade_date", "seq")
    FIELDS: ClassVar[dict[str, FieldSpec]] = {
        "title": FieldSpec("章节标题", primary=True),
        "content": FieldSpec("正文", primary=True),
    }

    trade_date: str  # YYYY-MM-DD
    seq: int  # 该日内章节顺序 (tushare 返回的行序)
    title: str | None = None
    content: str | None = None
    ingested_at: str | None = None  # ISO8601 时间戳


@dataclass(frozen=True, slots=True)
class CctvNewsMention:
    """NLP 匹配后的行业提及 flag — feature 直接 join 这张.

    每个 (trade_date, l1_code) pair 一行. 未被提及的行业不出现 (稀疏表).
    compute_features 里对 "不在表里" 视为 mentioned=0.
    """

    TABLE: ClassVar[str] = "cctv_news_mentions"
    PK: ClassVar[tuple[str, ...]] = ("trade_date", "l1_code")
    FIELDS: ClassVar[dict[str, FieldSpec]] = {
        "mentioned": FieldSpec("是否提及", primary=True),
        "match_score": FieldSpec("匹配分数 (raw, 0-1)"),
        "weighted_score": FieldSpec("TF-IDF 加权分数 (score × idf)"),
        "matched_keywords": FieldSpec("命中关键词 (JSON)"),
    }

    trade_date: str  # YYYY-MM-DD
    l1_code: str  # SW L1 '801770.SI' 等
    mentioned: int  # 0 / 1 (NLP 算法判定)
    match_score: float | None = None  # 0.0-1.0 原始 score = hits / n_kws
    weighted_score: float | None = None  # match_score × idf, baseline 缺时为 None
    matched_keywords: str | None = None  # JSON array e.g. '["5G", "通信"]'
    news_source: str | None = None  # 'tushare_cctv' / 'manual' / ...
    ingested_at: str | None = None
