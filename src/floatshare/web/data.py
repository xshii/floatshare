"""Dash 应用的数据访问层 — 从 DatabaseStorage + Treasury 读取展示所需数据。

单例风格：模块级 DB/book，多 callback 复用同一连接池。
"""

from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from floatshare.application.data_syncer import apply_adjustment
from floatshare.application.db_snapshot import (
    read_counts_snapshot,
    refresh_counts_snapshot,
)
from floatshare.application.treasury import Treasury
from floatshare.domain.enums import AdjustType
from floatshare.domain.records import (
    AdjFactor,
    CashAccount,
    RawDaily,
    StockLifecycle,
)
from floatshare.infrastructure.storage.database import DatabaseStorage

DB = DatabaseStorage()
BOOK = Treasury(DB)


# ============================================================================
# 账户
# ============================================================================


def list_accounts() -> pd.DataFrame:
    """所有活期账户列表。"""
    from sqlalchemy import text

    try:
        return pd.read_sql(
            text(
                f"SELECT account_id, name, created_at, memo "
                f"FROM {CashAccount.TABLE} ORDER BY created_at"
            ),
            DB.engine,
        )
    except Exception:
        return pd.DataFrame(columns=["account_id", "name", "created_at", "memo"])


def account_summary(account_id: str) -> dict:
    """账户现金 + 持仓市值 + 浮盈。"""
    cash = BOOK.balance(account_id)
    holdings = holdings_with_prices(account_id)
    if holdings.empty:
        market_value = 0.0
        cost = 0.0
    else:
        market_value = float(pd.Series(holdings["market_value"]).sum())
        cost = float(pd.Series(holdings["total_cost"]).sum())
    total_asset = cash + market_value
    invested = BOOK.total_invested(account_id)
    return {
        "cash": cash,
        "market_value": market_value,
        "total_asset": total_asset,
        "cost_basis": cost,
        "floating_pnl": market_value - cost,
        "invested": invested,
        "total_pnl": total_asset - invested if invested else 0.0,
    }


# ============================================================================
# 持仓 (批量单 SQL，避免 N+1)
# ============================================================================


def holdings_with_prices(account_id: str) -> pd.DataFrame:
    """定投持仓 + 最新收盘价 + 浮盈。"""
    portfolio = BOOK.portfolio(account_id)
    if portfolio.empty:
        return pd.DataFrame(
            columns=[
                "code",
                "shares",
                "avg_cost",
                "total_cost",
                "last_price",
                "market_value",
                "pnl",
                "pnl_pct",
            ]
        )
    prices = _latest_closes(portfolio["code"].tolist())
    portfolio["last_price"] = portfolio["code"].map(prices.get)
    portfolio["market_value"] = portfolio["shares"] * portfolio["last_price"].fillna(0)
    portfolio["pnl"] = portfolio["market_value"] - portfolio["total_cost"]
    portfolio["pnl_pct"] = portfolio["pnl"] / portfolio["total_cost"]
    return portfolio


def _latest_closes(codes: list[str]) -> dict[str, float]:
    """一次 SQL 拿一批 code 的最新收盘价。"""
    if not codes:
        return {}
    from sqlalchemy import text

    placeholders = ", ".join(f":c{i}" for i in range(len(codes)))
    params = {f"c{i}": c for i, c in enumerate(codes)}
    query = text(f"""
        SELECT code, close
        FROM {RawDaily.TABLE}
        WHERE (code, trade_date) IN (
            SELECT code, MAX(trade_date)
            FROM {RawDaily.TABLE}
            WHERE code IN ({placeholders})
            GROUP BY code
        )
    """)
    try:
        with DB.engine.connect() as conn:
            rows = conn.execute(query, params).all()
        return {str(r[0]): float(r[1]) for r in rows if r[1] is not None}
    except Exception:
        return {}


# ============================================================================
# K 线 + 股票候选
# ============================================================================


def load_klines(
    code: str,
    start: date | None = None,
    end: date | None = None,
    adj: AdjustType = AdjustType.NONE,
) -> pd.DataFrame:
    """从 raw_daily 加载 OHLCV (已按 trade_date 排序)。

    adj=NONE 直接读 raw_daily; QFQ/HFQ 额外拼 adj_factor 并调价。
    """
    df = DB.load(RawDaily.TABLE, code, start, end)
    if df.empty:
        return df
    if adj != AdjustType.NONE:
        factors = DB.load(
            AdjFactor.TABLE,
            code,
            start,
            end,
            columns="trade_date, adj_factor",
        )
        df = apply_adjustment(df, factors, adj)
    return pd.DataFrame(df[["trade_date", "open", "high", "low", "close", "volume"]])


def listed_codes() -> list[tuple[str, str]]:
    """返回 [(code, 'code 名称'), ...] — 只要在 raw_daily 里有数据的。"""
    from sqlalchemy import text

    query = text(f"""
        SELECT r.code, COALESCE(l.name, r.code) AS name
        FROM (SELECT DISTINCT code FROM {RawDaily.TABLE}) r
        LEFT JOIN {StockLifecycle.TABLE} l ON l.code = r.code
        ORDER BY r.code
    """)
    try:
        with DB.engine.connect() as conn:
            rows = conn.execute(query).all()
        return [(r[0], f"{r[0]} {r[1]}") for r in rows]
    except Exception:
        return []


# ============================================================================
# 运维
# ============================================================================


# 近 N 天每日 daily 表的 code 覆盖数 — sync 健康诊断用
# 每表期望 cover 的 code 数(全市场约 5500, 但部分表特性 cover 不全)
# severity 阈值: >=expected*0.9 ok / >=expected*0.5 warn / >0 bad / 0 empty
_DAILY_STATUS_TABLES: dict[str, int] = {
    "raw_daily": 5000,  # 全市场
    "adj_factor": 5000,  # 全市场
    "daily_basic": 5000,  # 全市场
    "moneyflow": 5000,  # 全市场
    "margin_detail": 1500,  # 仅融资融券标的
    "chip_perf": 3000,  # 筹码胜率, cover 不全
}

# daily_status 结果 60s 内存 cache, 防止重复点击重算
_DAILY_STATUS_TTL = 60.0
_daily_status_cache: tuple[float, int, list[DayStatus]] | None = None


@dataclass(frozen=True, slots=True)
class DayStatus:
    """单元格: (表, 日期, code 数)。Long-format, 让前端自己 pivot。

    severity 把"语义判定"留在 dataclass 里，UI 层只做颜色/类名映射。
    阈值按表自适应 —— margin_detail/chip_perf 本来就 cover 不到 5000。
    """

    table: str
    day: date
    code_count: int

    @property
    def severity(self) -> str:
        """ok | warn | bad | empty — 用于 UI 颜色映射。"""
        if self.code_count == 0:
            return "empty"
        expected = _DAILY_STATUS_TABLES.get(self.table, 5000)
        if self.code_count >= expected * 0.9:
            return "ok"
        if self.code_count >= expected * 0.5:
            return "warn"
        return "bad"


def _count_one_table(table: str, start_iso: str, end_iso: str) -> dict[date, int]:
    """单表 GROUP BY: 一条 SQL 拿 N 天每天的 distinct code 数。

    依赖 (trade_date, code) 覆盖索引 — 没建会回表全扫, 慢 50x。
    `scripts/add_status_indexes.py` 一次性建好。
    """
    from sqlalchemy import text

    query = text(f"""
        SELECT substr(trade_date, 1, 10) AS day, COUNT(DISTINCT code) AS n
        FROM {table}
        WHERE trade_date >= :start AND trade_date < :end_excl
        GROUP BY substr(trade_date, 1, 10)
    """)
    try:
        with DB.engine.connect() as conn:
            rows = conn.execute(
                query,
                {"start": start_iso, "end_excl": end_iso},
            ).all()
        return {date.fromisoformat(str(r[0])): int(r[1]) for r in rows}
    except Exception:
        return {}


def daily_status_cells(days: int = 7) -> list[DayStatus]:
    """近 N 天 × 6 个 daily 表的 code 覆盖数, long-format。

    优化: 1 表 1 GROUP BY (6 次代替 42 次) + 6 表 ThreadPool 并行 + 60s TTL cache。
    依赖 (trade_date, code) 覆盖索引 — 索引-only 扫描, 总耗时 <150ms。
    """
    global _daily_status_cache
    now = time.time()
    if (
        _daily_status_cache
        and _daily_status_cache[1] == days
        and now - _daily_status_cache[0] < _DAILY_STATUS_TTL
    ):
        return _daily_status_cache[2]

    today = date.today()
    days_list = [today - timedelta(days=i) for i in range(days - 1, -1, -1)]
    # 闭区间 [days_list[0], days_list[-1]] → 半开 [start, end_excl)
    start_iso = days_list[0].isoformat()
    end_excl_iso = (days_list[-1] + timedelta(days=1)).isoformat()

    tables = list(_DAILY_STATUS_TABLES)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tables)) as pool:
        results = list(
            pool.map(
                lambda t: (t, _count_one_table(t, start_iso, end_excl_iso)),
                tables,
            )
        )

    cells: list[DayStatus] = []
    for table, counts in results:
        cells.extend(DayStatus(table=table, day=d, code_count=counts.get(d, 0)) for d in days_list)

    _daily_status_cache = (now, days, cells)
    return cells


# 表行数 — 优先读 sync 写的 snapshot (instant), fallback 现算 (~25s, 阻塞)
# 60s 内存 TTL 防止 60s 慢 tick 反复读 JSON
_COUNTS_TTL = 60.0
_counts_cache: tuple[float, pd.DataFrame] | None = None


def table_counts() -> pd.DataFrame:
    """全表行数 — snapshot 优先，cache 后置防抖。"""
    global _counts_cache
    now = time.time()
    if _counts_cache and now - _counts_cache[0] < _COUNTS_TTL:
        return _counts_cache[1]
    rows = read_counts_snapshot()
    df = pd.DataFrame(rows) if rows else pd.DataFrame(refresh_counts_snapshot(DB))
    _counts_cache = (now, df)
    return df


# sync 进度直接走 SyncProgress dataclass —
#   from floatshare.application.sync_progress import SyncProgress
#   SyncProgress.read() / .percent / .eta_seconds / .is_running
