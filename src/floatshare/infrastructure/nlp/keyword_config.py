"""行业关键词配置的 Python API + 质量 lint.

设计目的:
    1. 暴露 data/news/industry_keywords.json 作为**可程序化访问**的系统资源,
       而不是散落在调用方的硬编码.
    2. 提供 lint: 词数边界 + 跨行业冲突 + 命名规范, 防止词典维护退化.

API:
    get_keywords(l1_code)           → 该行业关键词 tuple
    get_industry_name(l1_code)      → 'xxx'
    list_industries()               → [(l1_code, name), ...]
    find_conflicts()                → {keyword: [l1_code, ...]} 同词跨行业
    lint_dictionary(min=, max=)     → list[LintIssue]

约定: 所有函数读 `cctv_local.load_industry_keywords()` (带 lru_cache), 热改 JSON
后调 `load_industry_keywords.cache_clear()` 生效.
"""

from __future__ import annotations

from dataclasses import dataclass

from floatshare.infrastructure.nlp.cctv_local import load_industry_keywords


def get_keywords(l1_code: str) -> tuple[str, ...]:
    """查某 L1 行业的关键词 tuple. 不存在 → 空 tuple."""
    kd = load_industry_keywords()
    entry = kd.entries.get(l1_code)
    return entry[1] if entry else ()


def get_industry_name(l1_code: str) -> str | None:
    kd = load_industry_keywords()
    entry = kd.entries.get(l1_code)
    return entry[0] if entry else None


def list_industries() -> list[tuple[str, str]]:
    """返回 (l1_code, name), 按 l1_code 排序."""
    kd = load_industry_keywords()
    return sorted((code, name) for code, (name, _) in kd.entries.items())


def find_conflicts() -> dict[str, list[str]]:
    """跨行业重复词 — `{关键词: [l1_code, ...]}` (只含出现 >= 2 行业的词).

    冲突不必然是错 (例: "银行" 在 801780 银行 + 801790 非银金融 都合理),
    但应人工审视; 过多冲突会降低 score 分辨率.
    """
    kd = load_industry_keywords()
    kw_to_industries: dict[str, list[str]] = {}
    for l1_code, (_, kws) in kd.entries.items():
        for kw in kws:
            kw_to_industries.setdefault(kw, []).append(l1_code)
    return {kw: ls for kw, ls in kw_to_industries.items() if len(ls) >= 2}


@dataclass(frozen=True, slots=True)
class KeywordLintIssue:
    """词典 lint 的一个问题条目."""

    l1_code: str  # '_CROSS' 表示跨行业冲突
    l1_name: str
    issue: str  # 'too_few' / 'too_many' / 'conflict'
    severity: str  # 'warn' / 'error'
    detail: str


def lint_dictionary(
    min_keywords: int = 5,
    max_keywords: int = 40,
    max_conflict_industries: int = 2,
) -> list[KeywordLintIssue]:
    """词典质量检查.

    Args:
        min_keywords: 每行业至少 N 个关键词 (防稀疏 → 匹配率过低)
        max_keywords: 每行业最多 N 个关键词 (防稀释 → score 被灌水)
        max_conflict_industries: 一个词最多出现在 N 个行业 (超过 warn)

    Returns list of issues (空 list = 全过).
    """
    kd = load_industry_keywords()
    issues: list[KeywordLintIssue] = []

    # 1. 词数边界
    for l1_code, (name, kws) in kd.entries.items():
        if len(kws) < min_keywords:
            issues.append(
                KeywordLintIssue(
                    l1_code=l1_code,
                    l1_name=name,
                    issue="too_few",
                    severity="error",
                    detail=f"仅 {len(kws)} 词 (min {min_keywords}), 该行业几乎永不 match",
                )
            )
        if len(kws) > max_keywords:
            issues.append(
                KeywordLintIssue(
                    l1_code=l1_code,
                    l1_name=name,
                    issue="too_many",
                    severity="warn",
                    detail=f"{len(kws)} 词 (max {max_keywords}), score 会被稀释",
                )
            )

    # 2. 跨行业冲突
    conflicts = find_conflicts()
    for kw, l1s in sorted(conflicts.items()):
        if len(l1s) > max_conflict_industries:
            names = [f"{lc}({kd.entries[lc][0]})" for lc in l1s]
            issues.append(
                KeywordLintIssue(
                    l1_code="_CROSS",
                    l1_name="_CROSS",
                    issue="conflict",
                    severity="warn",
                    detail=f"'{kw}' 出现在 {len(l1s)} 行业: {', '.join(names)}",
                )
            )

    # 3. 空/重复词 (每行业内部)
    for l1_code, (name, kws) in kd.entries.items():
        if len(set(kws)) < len(kws):
            dups = [k for k in kws if kws.count(k) > 1]
            issues.append(
                KeywordLintIssue(
                    l1_code=l1_code,
                    l1_name=name,
                    issue="duplicate",
                    severity="error",
                    detail=f"重复词: {sorted(set(dups))}",
                )
            )
        if any(not k or not k.strip() for k in kws):
            issues.append(
                KeywordLintIssue(
                    l1_code=l1_code,
                    l1_name=name,
                    issue="empty_keyword",
                    severity="error",
                    detail="词典含空字符串",
                )
            )

    return issues
