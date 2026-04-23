"""DB 完整性体检 — daily-sync 后置检测 + floatshare-healthcheck --source localdb 用.

提供 4 类检查, 都以 raise DbIntegrityError 触发 healthcheck FAIL:

    1. check_trade_date_duplicates   — (code, trade_date) 重复行 (sync 幂等 bug)
    2. check_cross_table_alignment   — raw_daily 有但 daily_basic/moneyflow 缺
    3. check_universe_calendar_gaps  — universe 股上市后的停牌缺日
    4. check_daily_row_count_stability — 每天股数 vs 相邻日跳变 (数据源异常)

重复或跨表缺失会让 pandas rolling 按行算实际跨更多日 → feature 语义漂移.

用法:
    raise_if_any_duplicates('data/floatshare.db')         # #1
    raise_if_cross_table_missing('data/floatshare.db')    # #2
    raise_if_universe_gaps(db_path, universe, lookback)   # #3 (universe 必传)
    raise_if_row_count_unstable('data/floatshare.db')     # #4

CLI: floatshare-healthcheck --source localdb  (包含以上所有 probes)
     scripts/check_trade_date_dups.py         (legacy, 单 probe wrapper)
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise

from floatshare.domain.enums import ExchangeSuffix

# 需要检测 (code, trade_date) 唯一性的表
# Hint: 跳过无 trade_date 列的表 (如 trade_calendar / industry)
_DAILY_TABLES: tuple[str, ...] = (
    "raw_daily",
    "daily_basic",
    "moneyflow",
    "adj_factor",
    "margin_detail",
    "chip_perf",
    "index_daily",
)

# 跨表对齐: raw_daily 作基准, 其它表必须有同样 (code, trade_date). adj_factor 除外
# (停牌日也有复权因子, 反向多出是正常)
_CROSS_TABLE_CHILDREN: tuple[str, ...] = ("daily_basic", "moneyflow")


@dataclass(frozen=True, slots=True)
class DuplicateRow:
    """单条重复记录."""

    table: str
    code: str
    trade_date: str
    count: int


@dataclass(frozen=True, slots=True)
class CrossTableMissing:
    """raw_daily 有但子表缺."""

    child_table: str
    missing_pairs: int  # (code, trade_date) 对数


@dataclass(frozen=True, slots=True)
class UniverseGap:
    """某股上市后的停牌缺日."""

    code: str
    actual_days: int
    expected_days: int
    first_day: str


@dataclass(frozen=True, slots=True)
class RowCountJump:
    """相邻两天 raw_daily 股数跳变超阈值."""

    trade_date: str
    prev_date: str
    n_today: int
    n_prev: int
    change_pct: float


def _summarize_duplicates(issues: list[DuplicateRow]) -> str:
    table_counts: dict[str, int] = {}
    for d in issues:
        table_counts[d.table] = table_counts.get(d.table, 0) + 1
    return f"发现 {len(issues)} 条重复行 — " + ", ".join(
        f"{t}:{n}" for t, n in sorted(table_counts.items())
    )


def _summarize_cross_missing(issues: list[CrossTableMissing]) -> str:
    return "跨表缺失 — " + ", ".join(f"{m.child_table}缺{m.missing_pairs:,}对" for m in issues)


def _summarize_universe_gaps(issues: list[UniverseGap]) -> str:
    worst = max(issues, key=lambda g: g.expected_days - g.actual_days)
    return (
        f"universe {len(issues)} 只股有缺日, 最差 {worst.code} "
        f"缺 {worst.expected_days - worst.actual_days} 天"
    )


def _summarize_row_jumps(issues: list[RowCountJump]) -> str:
    return f"{len(issues)} 天股数跳变超阈值, 可能 sync 异常"


# 分派表 — 加新 issue 类型只需 append 一行 (Cookbook Recipe 9.21 思想)
_SUMMARIZERS: dict[type, Callable[[list], str]] = {
    DuplicateRow: _summarize_duplicates,
    CrossTableMissing: _summarize_cross_missing,
    UniverseGap: _summarize_universe_gaps,
    RowCountJump: _summarize_row_jumps,
}


class DbIntegrityError(RuntimeError):
    """DB 完整性检查失败 — 通常是 sync 重复插入 / 跨表不同步 / 数据源异常."""

    def __init__(self, issues: object) -> None:
        self.issues = issues
        super().__init__(self._summarize(issues))

    @staticmethod
    def _summarize(issues: object) -> str:
        """根据首元素类型查分派表, 找不到 fallback str()."""
        if isinstance(issues, list) and issues:
            fn = _SUMMARIZERS.get(type(issues[0]))
            if fn is not None:
                return fn(issues)
        return str(issues)


def check_trade_date_duplicates(
    db_path: str,
    tables: Sequence[str] = _DAILY_TABLES,
    lookback_days: int = 30,
) -> list[DuplicateRow]:
    """扫描每表的 (code, trade_date) 重复行. 返回全量 duplicate 列表.

    Args:
        db_path: floatshare.db 路径
        tables: 要检测的表 (默认 7 张日频主表)
        lookback_days: 只检查最近 N 天 (默认 30). 日常 healthcheck 不该全表扫 —
            GROUP BY 16M 行 ×7 表要 ~25s. 设 0 = 全表扫 (历史 audit 用).

    Returns:
        list[DuplicateRow]. 空 list = 全干净.
    """
    duplicates: list[DuplicateRow] = []
    where_clause = (
        f"WHERE trade_date >= date('now', '-{int(lookback_days)} days')"
        if lookback_days > 0
        else ""
    )
    with sqlite3.connect(db_path) as conn:
        for table in tables:
            if not _table_exists(conn, table):
                continue
            # 同 (code, trade_date) 行数 > 1 即重复
            rows = conn.execute(f"""
                SELECT code, trade_date, COUNT(*) AS n
                FROM {table}
                {where_clause}
                GROUP BY code, trade_date
                HAVING n > 1
                ORDER BY n DESC, code, trade_date
                LIMIT 1000
            """).fetchall()
            duplicates.extend(
                DuplicateRow(table=table, code=r[0], trade_date=r[1], count=r[2]) for r in rows
            )
    return duplicates


def check_cross_table_alignment(
    db_path: str,
    children: Sequence[str] = _CROSS_TABLE_CHILDREN,
    base: str = "raw_daily",
    lookback_days: int = 30,
    exclude_suffix: Sequence[str] = (ExchangeSuffix.BJ,),
) -> list[CrossTableMissing]:
    """检查 base 有但 children 缺的 (code, trade_date) 对.

    默认 base=raw_daily, children=(daily_basic, moneyflow). adj_factor 跳过 (停牌日
    也有复权因子, 反向多出正常).

    Args:
        lookback_days: 只检查最近 N 天 (默认 30). 日常 healthcheck 不该全表扫 —
            full-scan 1600 万行 ×2 表要 ~2min, 生产不可接受. 设 0 = 全表扫 (历史
            audit 用).
        exclude_suffix: 跳过这些交易所后缀的 code (默认过滤 BJ —— tushare
            moneyflow 完全不覆盖北交所, 无须告警). 传 () 关闭过滤.
    """
    missing: list[CrossTableMissing] = []
    with sqlite3.connect(db_path) as conn:
        if not _table_exists(conn, base):
            return missing
        where_parts = ["c.code IS NULL"]
        if lookback_days > 0:
            where_parts.append(f"r.trade_date >= date('now', '-{int(lookback_days)} days')")
        where_parts.extend(f"r.code NOT LIKE '%{s}'" for s in exclude_suffix)
        where_clause = " AND ".join(where_parts)
        for child in children:
            if not _table_exists(conn, child):
                continue
            n = conn.execute(f"""
                SELECT COUNT(*) FROM {base} r
                LEFT JOIN {child} c ON c.code = r.code AND c.trade_date = r.trade_date
                WHERE {where_clause}
            """).fetchone()[0]
            if n > 0:
                missing.append(CrossTableMissing(child_table=child, missing_pairs=int(n)))
    return missing


def check_universe_calendar_gaps(
    db_path: str,
    universe: Sequence[str],
    lookback_days: int = 1825,
    base: str = "raw_daily",
    min_gap_days: int = 1,
) -> list[UniverseGap]:
    """universe 里每只股上市后的停牌缺日 (相对 base 表的交易日历).

    Args:
        lookback_days: 只看近 N 天 (默认 5 年 ≈ 1825 日历日)
        min_gap_days: 缺日少于此数不报 (避免小噪音)
    """
    if not universe:
        return []
    gaps: list[UniverseGap] = []
    placeholders = ",".join("?" * len(universe))
    with sqlite3.connect(db_path) as conn:
        if not _table_exists(conn, base):
            return gaps
        # 日历基准: base 表最近 lookback_days 里所有 distinct trade_date
        cal_rows = conn.execute(f"""
            SELECT DISTINCT trade_date FROM {base}
            WHERE trade_date >= date('now', '-{int(lookback_days)} days')
            ORDER BY trade_date
        """).fetchall()
        calendar = [r[0] for r in cal_rows]
        if not calendar:
            return gaps
        # 每股 actual_days + first_day
        stock_rows = conn.execute(
            f"""
            SELECT code, COUNT(DISTINCT trade_date), MIN(trade_date)
            FROM {base}
            WHERE code IN ({placeholders})
              AND trade_date >= date('now', '-{int(lookback_days)} days')
            GROUP BY code
            """,
            list(universe),
        ).fetchall()
    for code, actual, first_day in stock_rows:
        if first_day is None:
            continue
        expected = sum(1 for d in calendar if d >= first_day)
        if expected - actual >= min_gap_days:
            gaps.append(
                UniverseGap(
                    code=code,
                    actual_days=actual,
                    expected_days=expected,
                    first_day=first_day,
                )
            )
    return gaps


def check_daily_row_count_stability(
    db_path: str,
    table: str = "raw_daily",
    lookback_days: int = 30,
    max_change_pct: float = 0.05,
) -> list[RowCountJump]:
    """检测相邻交易日股数变化超 max_change_pct 的天 (可能是 sync 失败信号).

    正常情况下 A 股每天股数 ±5 只 (新上市/退市), 超 5% 变化常是数据源某天挂了.
    """
    jumps: list[RowCountJump] = []
    with sqlite3.connect(db_path) as conn:
        if not _table_exists(conn, table):
            return jumps
        rows = conn.execute(f"""
            SELECT trade_date, COUNT(*) AS n
            FROM {table}
            WHERE trade_date >= date('now', '-{int(lookback_days)} days')
            GROUP BY trade_date ORDER BY trade_date
        """).fetchall()
    for (prev_d, prev_n), (cur_d, cur_n) in pairwise(rows):
        if prev_n == 0:
            continue
        change_pct = (cur_n - prev_n) / prev_n
        if abs(change_pct) > max_change_pct:
            jumps.append(
                RowCountJump(
                    trade_date=cur_d,
                    prev_date=prev_d,
                    n_today=cur_n,
                    n_prev=prev_n,
                    change_pct=float(change_pct),
                )
            )
    return jumps


# --- 统一的 raise_if 入口 (消除重复) -----------------------------------------


def _raise_if_issues(issues: list) -> None:
    """任何 check_* 返回的 issues list, 非空就 raise DbIntegrityError."""
    if issues:
        raise DbIntegrityError(issues)


def raise_if_any_duplicates(
    db_path: str,
    tables: Sequence[str] = _DAILY_TABLES,
) -> None:
    _raise_if_issues(check_trade_date_duplicates(db_path, tables))


def raise_if_cross_table_missing(
    db_path: str,
    children: Sequence[str] = _CROSS_TABLE_CHILDREN,
) -> None:
    _raise_if_issues(check_cross_table_alignment(db_path, children))


def raise_if_universe_gaps(
    db_path: str,
    universe: Sequence[str],
    lookback_days: int = 1825,
) -> None:
    _raise_if_issues(check_universe_calendar_gaps(db_path, universe, lookback_days))


def raise_if_row_count_unstable(
    db_path: str,
    lookback_days: int = 30,
    max_change_pct: float = 0.05,
) -> None:
    _raise_if_issues(
        check_daily_row_count_stability(
            db_path,
            lookback_days=lookback_days,
            max_change_pct=max_change_pct,
        )
    )


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None
