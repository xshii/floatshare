"""新闻联播文字稿 → SW L1 行业提及 flag 的本地 NLP.

算法:
    1. 拿完整联播文字稿 (tushare cctv_news 的所有 content concat)
    2. (可选) jieba 精确分词, 若未装则退化为 substring 命中
    3. 对每个 L1 行业词典, 数命中词数
    4. hits >= match_min_count 且 raw_score >= threshold → mentioned=1
       raw_score = hits / len(keywords)   — 归一到 [0, 1]
       weighted_score = raw_score * idf   — 冷门行业被提及时加大权重 (TF-IDF 风格)

词典源: data/news/industry_keywords.json (30 行业 × 10-20 词, v2 删了"综合").
基线源: data/news/industry_baseline.json (90 天 rolling, 月更, 由 build_news_baseline.py 生成).
调用方: application/news_ingest.py 每日 T 日 20:00 ingest 时调用.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

INDUSTRY_KEYWORDS_PATH = Path("data/news/industry_keywords.json")
INDUSTRY_BASELINE_PATH = Path("data/news/industry_baseline.json")


@dataclass(frozen=True, slots=True)
class IndustryMention:
    """一个行业的匹配结果 (未命中不生成此对象)."""

    l1_code: str  # '801770.SI'
    l1_name: str  # '通信'
    score: float  # 0.0-1.0 (hits / len(keywords))
    matched_keywords: list[str]  # 命中的词 (debug / 存 DB)
    weighted_score: float | None = None  # score * idf; baseline 缺时为 None


@dataclass(frozen=True, slots=True)
class KeywordDict:
    """行业关键词词典 + 匹配阈值."""

    entries: dict[str, tuple[str, tuple[str, ...]]]  # l1_code → (name, keywords)
    match_min_count: int = 2
    match_score_threshold: float = 0.1


@lru_cache(maxsize=1)
def load_industry_keywords(path: Path | None = None) -> KeywordDict:
    """读 data/news/industry_keywords.json, 缓存结果."""
    p = path or INDUSTRY_KEYWORDS_PATH
    raw = json.loads(p.read_text(encoding="utf-8"))
    meta = raw.pop("_meta", {})
    entries: dict[str, tuple[str, tuple[str, ...]]] = {
        l1: (v["name"], tuple(v["keywords"])) for l1, v in raw.items()
    }
    return KeywordDict(
        entries=entries,
        match_min_count=int(meta.get("match_min_count", 2)),
        match_score_threshold=float(meta.get("match_score_threshold", 0.1)),
    )


@lru_cache(maxsize=1)
def load_industry_baseline(path: Path | None = None) -> dict[str, float]:
    """读 data/news/industry_baseline.json 返回 {l1_code: idf}.

    文件不存在 / 损坏 → 返回空 dict, 调用方降级 (weighted_score=None).
    由 scripts/build_news_baseline.py 生成, 推荐每月重算一次.
    """
    p = path or INDUSTRY_BASELINE_PATH
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    industries = raw.get("industries", {})
    return {l1: float(v["idf"]) for l1, v in industries.items() if "idf" in v}


def extract_industry_mentions(
    text: str,
    keywords: KeywordDict | None = None,
    baseline_idf: dict[str, float] | None = None,
) -> list[IndustryMention]:
    """输入新闻联播完整文字稿 → 输出被提及的行业列表.

    未装 jieba 退化为 substring match (对短语 / 英文缩写 如 '5G' 表现一致).
    装 jieba 后走精确分词 + 词袋匹配, 对歧义词 ('银行' vs '银河') 更准.

    baseline_idf 默认从 data/news/industry_baseline.json 自动读. 文件缺 →
    weighted_score = None; 文件在 → weighted_score = raw_score * idf.
    """
    kd = keywords or load_industry_keywords()
    if not text:
        return []
    idf_map = baseline_idf if baseline_idf is not None else load_industry_baseline()
    tokens = _tokenize(text)
    token_set = set(tokens)

    mentions: list[IndustryMention] = []
    for l1_code, (name, kws) in kd.entries.items():
        hits = [kw for kw in kws if _kw_hit(kw, token_set, text)]
        if len(hits) < kd.match_min_count:
            continue
        score = len(hits) / max(len(kws), 1)
        if score < kd.match_score_threshold:
            continue
        idf = idf_map.get(l1_code)
        weighted = float(score) * idf if idf is not None else None
        mentions.append(
            IndustryMention(
                l1_code=l1_code,
                l1_name=name,
                score=float(score),
                matched_keywords=hits,
                weighted_score=weighted,
            )
        )
    return mentions


def _tokenize(text: str) -> list[str]:
    """优先 jieba 精确模式; 未装 jieba 返回空 list (由 _kw_hit 走 substring fallback)."""
    try:
        import jieba
    except ImportError:
        return []
    return list(jieba.cut(text, cut_all=False))


def _kw_hit(kw: str, token_set: set[str], text: str) -> bool:
    """关键词命中判定:

    - 英文 / 含数字 (如 '5G', 'AI'): 只走 substring (jieba 对短英文不准)
    - 中文短语: 先试分词 token_set (精确), 否则 fallback substring

    substring fallback 对大部分中文短语 OK, 只在歧义词 (~ 2-3 个词典里) 可能误匹.
    """
    if _is_ascii_like(kw):
        return kw in text
    if token_set and kw in token_set:
        return True
    return kw in text  # substring fallback


def _is_ascii_like(s: str) -> bool:
    return all(ord(c) < 128 for c in s)
