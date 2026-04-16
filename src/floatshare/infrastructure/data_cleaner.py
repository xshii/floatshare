"""数据清洗 — 模块级函数，无副作用。"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from floatshare.domain.enums import AdjustType

_PRICE_COLS = ("open", "high", "low", "close")


def clean_daily(df: pd.DataFrame) -> pd.DataFrame:
    """清洗日线数据：去重、补缺、去极值、按时间排序。"""
    if df.empty:
        return df

    df = df.copy()
    df = df.drop_duplicates(subset=["code", "trade_date"], keep="last")
    df = df.sort_values("trade_date")
    df = fill_missing_prices(df)
    df = handle_outliers(df)
    return df.reset_index(drop=True)


def fill_missing_prices(df: pd.DataFrame) -> pd.DataFrame:
    """前值填充价格列，量额缺失填 0。"""
    for col in _PRICE_COLS:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0)
    if "amount" in df.columns:
        df["amount"] = df["amount"].fillna(0)
    return df


def handle_outliers(df: pd.DataFrame, std_threshold: float = 5.0) -> pd.DataFrame:
    """对收益率超过 N 倍标准差的样本前向填充。"""
    for col in _PRICE_COLS:
        if col not in df.columns:
            continue
        series = cast(pd.Series, df[col])
        returns = series.pct_change()
        std_raw = returns.std()
        std_val = float(std_raw) if not isinstance(std_raw, pd.Series) else float("nan")
        if not std_val or std_val <= 0:
            continue
        mean_raw = returns.mean()
        mean_val = float(mean_raw) if not isinstance(mean_raw, pd.Series) else 0.0
        outliers = np.abs(returns - mean_val) > std_threshold * std_val
        if bool(outliers.any()):
            df.loc[outliers, col] = np.nan
            df[col] = df[col].ffill()
    return df


def adjust_price(
    df: pd.DataFrame,
    adj_factor: pd.Series,
    method: AdjustType = AdjustType.QFQ,
) -> pd.DataFrame:
    """按复权因子重新计算价格列。"""
    if method == AdjustType.NONE:
        return df

    df = df.copy()
    if method == AdjustType.QFQ:
        factor = adj_factor / adj_factor.iloc[-1]
    else:  # HFQ
        factor = adj_factor / adj_factor.iloc[0]

    for col in _PRICE_COLS:
        if col in df.columns:
            df[col] = df[col] * factor
    return df


def resample(
    df: pd.DataFrame,
    freq: str = "W",
    agg_dict: dict[str, str] | None = None,
) -> pd.DataFrame:
    """日线 → 周/月线重采样。"""
    if agg_dict is None:
        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "amount": "sum",
        }
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index("trade_date")
    valid = {k: v for k, v in agg_dict.items() if k in df.columns}
    return df.resample(freq).agg(valid).dropna(how="all").reset_index()
