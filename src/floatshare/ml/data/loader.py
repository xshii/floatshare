"""SQLite → 多表 join → DataFrame，喂给 features.py 算指标。

raw_daily / daily_basic / moneyflow / industry / index_daily 五张表的高效 join,
按 (code, trade_date) 索引、按 trade_date 排序。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from sqlalchemy import create_engine, text

if TYPE_CHECKING:
    from collections.abc import Sequence


def _pad_day_start(d: str) -> str:
    """'YYYY-MM-DD' → 'YYYY-MM-DDT00:00:00'. 已含 T 的透传."""
    return d if "T" in d else f"{d}T00:00:00"


def _pad_day_end(d: str) -> str:
    """'YYYY-MM-DD' → 'YYYY-MM-DDT23:59:59'. 已含 T 的透传."""
    return d if "T" in d else f"{d}T23:59:59"


def load_panel(
    db_path: str,
    codes: Sequence[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """读 OHLCV + 估值 + 资金流 — long format (code, trade_date, ...26 列).

    返回 DataFrame 列:
        code, trade_date
        open, high, low, close, volume, amount                   # raw (不复权)
        open_qfq, high_qfq, low_qfq, close_qfq                   # 前复权 (技术指标专用)
        pe_ttm, pb, turnover_rate, total_mv, circ_mv              # daily_basic
        net_mf_amount, buy_{lg/elg/sm/md}_amount,
        sell_{sm/md/lg/elg}_amount                                # moneyflow

    前复权公式: p_qfq[t] = p[t] × adj_factor[t] / latest_adj_factor[code]
        确保 p_qfq 最后一天 == p 最后一天, 除权日无跳空. 与 tushare stk_factor.xxx_qfq
        一致, 技术指标 (RSI/MACD/KDJ/ATR) 基于 qfq 跟业界对齐.

    缺失值不填 — features.py 决定怎么处理.
    """
    if not codes:
        return pd.DataFrame()
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"timeout": 30.0})
    placeholders = ",".join(f":c{i}" for i in range(len(codes)))
    params: dict[str, object] = {f"c{i}": c for i, c in enumerate(codes)}
    # DB trade_date 存 'YYYY-MM-DDT00:00:00' (ISO with time). 纯日期串 end='2026-04-21'
    # 字典序 < 'YYYY-MM-DDT00:00:00', 导致 `<= end` 排除当日整天 — 补时间分量防漏
    params.update({"start": _pad_day_start(start), "end": _pad_day_end(end)})

    # 三表 LEFT JOIN: raw_daily 主, daily_basic + moneyflow 辅. adj_factor 另取.
    sql = text(f"""
        SELECT
            r.code, r.trade_date,
            r.open, r.high, r.low, r.close, r.volume, r.amount,
            a.adj_factor,
            b.pe_ttm, b.pb, b.turnover_rate, b.total_mv, b.circ_mv,
            m.net_mf_amount,
            m.buy_lg_amount, m.buy_elg_amount,
            m.buy_sm_amount, m.buy_md_amount,
            m.sell_sm_amount, m.sell_md_amount,
            m.sell_lg_amount, m.sell_elg_amount
        FROM raw_daily r
        LEFT JOIN daily_basic b
               ON b.code = r.code AND b.trade_date = r.trade_date
        LEFT JOIN moneyflow m
               ON m.code = r.code AND m.trade_date = r.trade_date
        LEFT JOIN adj_factor a
               ON a.code = r.code AND a.trade_date = r.trade_date
        WHERE r.code IN ({placeholders})
          AND r.trade_date >= :start
          AND r.trade_date <= :end
        ORDER BY r.code, r.trade_date
    """)
    df = pd.read_sql(sql, engine, params=params)
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="ISO8601")
    # 前复权: 每只股按 latest adj_factor 归一化 (最后一天价 == raw 价)
    return _attach_qfq_prices(df)


def _attach_qfq_prices(df: pd.DataFrame) -> pd.DataFrame:
    """对每只股算 close_qfq/open_qfq/high_qfq/low_qfq.

    p_qfq[t] = p[t] * adj_factor[t] / latest_adj_factor
    latest = df 里该股的 adj_factor 最后一行 (如果 adj_factor NaN, qfq = raw).
    """
    import numpy as np

    if "adj_factor" not in df.columns:
        # fallback: 没 adj_factor 就复制 raw
        for col in ("open", "high", "low", "close"):
            df[f"{col}_qfq"] = df[col]
        return df
    # 每只股的 latest adj_factor (组内最后一行的 adj_factor)
    latest = df.groupby("code")["adj_factor"].transform("last")
    ratio = df["adj_factor"] / latest
    # NaN adj_factor (无复权记录) → qfq = raw
    ratio = ratio.fillna(1.0)
    for col in ("open", "high", "low", "close"):
        df[f"{col}_qfq"] = (df[col] * ratio).astype(np.float64)
    return df.drop(columns=["adj_factor"])


def load_industry_map(db_path: str, level: int = 1) -> pd.DataFrame:
    """code → SW 行业 (L1/L2/L3) 一对一映射。

    level=1 → l1_code, l1_name
    level=2 → l2_code, l2_name
    level=3 → l3_code, l3_name
    """
    if level not in (1, 2, 3):
        raise ValueError(f"level must be 1/2/3, got {level}")
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"timeout": 30.0})
    cols = f"code, l{level}_code AS ind_code, l{level}_name AS ind_name"
    return pd.read_sql(text(f"SELECT {cols} FROM industry"), engine)


def load_market_returns(
    db_path: str,
    start: str,
    end: str,
    market_code: str = "000300.SH",
) -> pd.Series:
    """加载市场基准的日 log return — 默认沪深 300, 用于 reward benchmark。

    返回: Series, index=trade_date, value=log_return (length 同 cube.dates - 1)
    """
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"timeout": 30.0})
    sql = text("""
        SELECT trade_date, close FROM index_daily
        WHERE code = :code AND trade_date >= :start AND trade_date <= :end
        ORDER BY trade_date
    """)
    df = pd.read_sql(
        sql,
        engine,
        params={"code": market_code, "start": _pad_day_start(start), "end": _pad_day_end(end)},
    )
    if df.empty:
        return pd.Series(dtype="float32")
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="ISO8601")
    df = df.set_index("trade_date")
    import numpy as np

    return np.log(df["close"] / df["close"].shift(1)).dropna().astype("float32")


def load_index_panel(
    db_path: str,
    index_codes: Sequence[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """读指数日线 — 用于 reward benchmark (SW 行业指数 + 市场指数)。

    返回: code, trade_date, close, pct_change
    """
    if not index_codes:
        return pd.DataFrame()
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"timeout": 30.0})
    placeholders = ",".join(f":c{i}" for i in range(len(index_codes)))
    params: dict[str, object] = {f"c{i}": c for i, c in enumerate(index_codes)}
    params.update({"start": _pad_day_start(start), "end": _pad_day_end(end)})
    sql = text(f"""
        SELECT code, trade_date, close, pct_change
        FROM index_daily
        WHERE code IN ({placeholders})
          AND trade_date >= :start
          AND trade_date <= :end
        ORDER BY code, trade_date
    """)
    df = pd.read_sql(sql, engine, params=params)
    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="ISO8601")
    return df
