"""Universe 选股 — 沪深 300 / 中证 500 / top-N 流通市值 / 每行业 top-K.

输入: DataConfig (universe_mode + 参数) 或显式 db_path 等
输出: list[str] — 选定的 code 列表 (含 .SH/.SZ 后缀)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine

from floatshare.domain.enums import ExchangeSuffix

if TYPE_CHECKING:
    from floatshare.ml.config import DataConfig


_INDEX_CODE: dict[str, str] = {
    "hs300": "000300.SH",
    "zz500": "000905.SH",
}


def select_universe(cfg: DataConfig, as_of_date: str | None = None) -> list[str]:
    """根据 cfg.universe_mode 选股池。

    Args:
        cfg: DataConfig
        as_of_date: 选股池快照时间 (None=最新一期)。Backtest 应传训练集结束日,
                    避免 universe 含未来上市的股 → 前视偏差。

    Returns:
        排序后的 code 列表
    """
    engine = _make_engine(cfg.db_path)
    with engine.connect() as conn:
        if cfg.universe_mode in _INDEX_CODE:
            return _from_index_weight(conn, _INDEX_CODE[cfg.universe_mode], as_of_date)
        if cfg.universe_mode == "top_mv":
            return _top_mv(conn, cfg.top_mv_n, as_of_date)
        raise ValueError(f"unknown universe_mode: {cfg.universe_mode}")


def select_per_industry_top_k(
    db_path: str,
    as_of_date: str,
    top_k: int = 15,
    turnover_window: int = 20,
    vola_window: int = 60,
    circ_mv_min: float = 20_0000.0,  # circ_mv 单位 万元 = 20 亿元
    circ_mv_max: float = 500_0000.0,  # = 500 亿元
    w_turnover: float = 0.5,
    w_vola: float = 0.35,
    w_cmv: float = 0.15,
) -> list[str]:
    """每个 SW L1 行业内按换手/波动/流通盘合成打分取 top-K.

    硬过滤:
        - 主板 (code 前缀 60xx / 00xx)
        - 非 ST / *ST
        - 上市 >= 120 天
        - 流通市值 ∈ [circ_mv_min, circ_mv_max]
        - snapshot 当日 volume > 0

    打分 (行业内 z-score 合成):
        score = w_turnover · z(turnover_ma20)        — 换手率 (业界打板首选信号)
              + w_vola     · z(vola_60d)             — 60 天 ret_1d σ (涨停潜力)
              + w_cmv      · z(-log(circ_mv))        — 偏小盘 (负号, sweet spot 内)
    """
    engine = _make_engine(db_path)
    with engine.connect() as conn:
        snap = _get_snapshot_date(conn, as_of_date)
        if not snap:
            return []

    base = _query_base_candidates(engine, snap, circ_mv_min, circ_mv_max)
    if base.empty:
        return []
    turnover = _query_turnover_window(engine, snap, turnover_window)
    vola = _query_vola_window(engine, snap, vola_window)

    df = _merge_and_fill(base, turnover, vola)
    df = _compute_industry_scores(df, w_turnover, w_vola, w_cmv)
    return _topk_per_group(df, group_col="l1_code", top_k=top_k)


# --- helpers -----------------------------------------------------------------


def _make_engine(db_path: str) -> Engine:
    return create_engine(f"sqlite:///{db_path}", connect_args={"timeout": 30.0})


def _get_snapshot_date(conn: Connection, as_of_date: str) -> str | None:
    """取 daily_basic 中 <= as_of_date 的最新一期 trade_date."""
    row = conn.execute(
        text(
            "SELECT MAX(trade_date) FROM daily_basic WHERE trade_date <= :d",
        ),
        {"d": as_of_date},
    ).scalar()
    return row if row else None


def _query_base_candidates(
    engine: Engine,
    snap: str,
    cmv_min: float,
    cmv_max: float,
) -> pd.DataFrame:
    """硬过滤 — 主板 + 非 ST + 上市 120 天 + circ_mv 区间 + 成交."""
    sql = text("""
        SELECT db.code, db.circ_mv, ind.l1_code, sl.name as stock_name
        FROM daily_basic db
        JOIN industry ind       ON ind.code = db.code
        JOIN stock_lifecycle sl ON sl.code  = db.code
        WHERE db.trade_date = :d
          AND (db.code LIKE '60%' OR db.code LIKE '00%')
          AND sl.name NOT LIKE 'ST%'
          AND sl.name NOT LIKE '*ST%'
          AND sl.list_date IS NOT NULL
          AND date(sl.list_date) < date(substr(:d, 1, 10), '-120 days')
          AND db.circ_mv BETWEEN :cmv_min AND :cmv_max
          AND db.turnover_rate > 0
          AND ind.l1_code IS NOT NULL
    """)
    return pd.read_sql(
        sql,
        engine,
        params={"d": snap, "cmv_min": cmv_min, "cmv_max": cmv_max},
    )


def _query_turnover_window(engine: Engine, snap: str, window: int) -> pd.DataFrame:
    """窗口内每股 avg turnover_rate. 1.7 倍日历天覆盖交易日 + HAVING 保证样本数."""
    lookback_days = int(window * 1.7)
    min_samples = max(5, window // 4)
    sql = text(f"""
        SELECT code, AVG(turnover_rate) as turnover_ma
        FROM daily_basic
        WHERE trade_date <= :d
          AND trade_date >= date(substr(:d, 1, 10), '-{lookback_days} days')
          AND turnover_rate IS NOT NULL
        GROUP BY code
        HAVING COUNT(*) >= {min_samples}
    """)
    return pd.read_sql(sql, engine, params={"d": snap})


def _query_vola_window(engine: Engine, snap: str, window: int) -> pd.DataFrame:
    """窗口内每股 pct_change 标准差 (在 SQL 里直接算 E[x²]-E[x]²)."""
    lookback_days = int(window * 1.7)
    min_samples = max(10, window // 3)
    sql = text(f"""
        WITH r AS (
            SELECT code, trade_date, pct_change
            FROM raw_daily
            WHERE trade_date <= :d
              AND trade_date >= date(substr(:d, 1, 10), '-{lookback_days} days')
              AND pct_change IS NOT NULL
        )
        SELECT code,
               AVG(pct_change * pct_change) - AVG(pct_change) * AVG(pct_change) as vola_sq
        FROM r GROUP BY code HAVING COUNT(*) >= {min_samples}
    """)
    df = pd.read_sql(sql, engine, params={"d": snap})
    df["vola_60d"] = np.sqrt(df["vola_sq"].clip(lower=0))
    return df[["code", "vola_60d"]]


def _merge_and_fill(
    base: pd.DataFrame,
    turnover: pd.DataFrame,
    vola: pd.DataFrame,
) -> pd.DataFrame:
    """Merge 打分维度, 缺失填全局中位数 (保证所有行业都有候选)."""
    df = base.merge(turnover, on="code", how="left").merge(vola, on="code", how="left")
    df["turnover_ma"] = df["turnover_ma"].fillna(df["turnover_ma"].median())
    df["vola_60d"] = df["vola_60d"].fillna(df["vola_60d"].median())
    return df


def _compute_industry_scores(
    df: pd.DataFrame,
    w_turnover: float,
    w_vola: float,
    w_cmv: float,
) -> pd.DataFrame:
    """行业内 z-score 合成打分 (modify in-place)."""
    df["neg_log_cmv"] = -np.log(df["circ_mv"].clip(lower=1))
    df["z_turnover"] = df.groupby("l1_code")["turnover_ma"].transform(_zscore)
    df["z_vola"] = df.groupby("l1_code")["vola_60d"].transform(_zscore)
    df["z_cmv"] = df.groupby("l1_code")["neg_log_cmv"].transform(_zscore)
    df["score"] = w_turnover * df["z_turnover"] + w_vola * df["z_vola"] + w_cmv * df["z_cmv"]
    return df


def _zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std(ddof=0) + 1e-9)


def _topk_per_group(df: pd.DataFrame, group_col: str, top_k: int) -> list[str]:
    """按 group 取 score 降序前 top_k, 返回 sorted code 列表."""
    top = (
        df.sort_values([group_col, "score"], ascending=[True, False]).groupby(group_col).head(top_k)
    )
    return sorted(top["code"].tolist())


def _from_index_weight(
    conn: Connection,
    index_code: str,
    as_of_date: str | None,
) -> list[str]:
    """从 index_weight 表取某指数最新或指定日期的成分股."""
    if as_of_date is None:
        as_of_date = conn.execute(
            text(
                "SELECT MAX(trade_date) FROM index_weight WHERE index_code = :idx",
            ),
            {"idx": index_code},
        ).scalar()
    snap = conn.execute(
        text("""
        SELECT MAX(trade_date) FROM index_weight
        WHERE index_code = :idx AND trade_date <= :d
    """),
        {"idx": index_code, "d": as_of_date},
    ).scalar()
    if not snap:
        return []
    rows = conn.execute(
        text("""
        SELECT con_code FROM index_weight
        WHERE index_code = :idx AND trade_date = :d
        ORDER BY con_code
    """),
        {"idx": index_code, "d": snap},
    ).all()
    return [r[0] for r in rows]


def _top_mv(conn: Connection, n: int, as_of_date: str | None) -> list[str]:
    """≤ as_of_date 最近日按流通市值 top-N, 过滤 ST / 新股 / 停牌 / 北交所.

    业界经验过滤:
        - circ_mv (而非 total_mv) — 实际可交易池
        - turnover_rate > 0 (排除停盘)
        - 上市 ≥ 120 交易日 (新股 IPO 前期波动大, 不够 seq_len 历史)
        - ST / *ST (业绩差 + 流动性差, 分布不同)
        - 北交所 (*.BJ) — tushare moneyflow 不覆盖 (7/39 特征全 NaN),
          涨跌幅规则 ±30% 与 A 股 ±10% 不同, 抓涨停策略语义不适用
    """
    params: dict[str, Any] = {"d": as_of_date or "9999-12-31"}
    snap = conn.execute(
        text("""
        SELECT MAX(trade_date) FROM daily_basic WHERE trade_date <= :d
    """),
        params,
    ).scalar()
    if not snap:
        return []
    rows = conn.execute(
        text(f"""
        SELECT db.code FROM daily_basic db
        JOIN stock_lifecycle sl ON sl.code = db.code
        WHERE db.trade_date = :d
          AND db.circ_mv IS NOT NULL
          AND db.turnover_rate > 0
          AND db.code NOT LIKE '%{ExchangeSuffix.BJ}'
          AND sl.name NOT LIKE 'ST%'
          AND sl.name NOT LIKE '*ST%'
          AND sl.list_date IS NOT NULL
          AND date(sl.list_date) < date(substr(:d, 1, 10), '-120 days')
        ORDER BY db.circ_mv DESC LIMIT :n
    """),
        {"d": snap, "n": n},
    ).all()
    return sorted(r[0] for r in rows)
