"""OHLCV DataFrame schema — 数据源适配后的统一形状约定。

每个数据源（Tushare/AKShare/EastMoney）的 daily 类方法返回前应该调用
`normalize_ohlcv()` 把 DataFrame 转成此 schema:
- 必需列存在且类型正确 (缺失 → ValueError)
- 可选列缺失时补 pd.NA
- 列顺序统一为 REQUIRED + OPTIONAL + 其它额外列(按字母序)

这样 DataLoader 链式降级（Tushare → AKShare → EastMoney）时，
无论命中哪个源，下游拿到的 DataFrame 形状一致，避免 KeyError 风险。
"""

from __future__ import annotations

import pandas as pd

# 必需列 — 任何 OHLCV 数据源都必须返回
OHLCV_REQUIRED: tuple[str, ...] = (
    "code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

# 可选列 — 不同数据源覆盖不同
OHLCV_OPTIONAL: tuple[str, ...] = (
    "amount",  # 成交额
    "pre_close",  # 前收盘
    "pct_change",  # 涨跌幅 (%)
    "turnover",  # 换手率
    "adj_factor",  # 复权因子（merge 后的 DataFrame）
)


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """把 DataFrame 调整到统一 OHLCV schema (空 df 透传)。

    Raises:
        ValueError: 必需列缺失时
    """
    if df.empty:
        return df

    missing = [c for c in OHLCV_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV DataFrame 缺少必需列: {missing}")

    df = df.copy()
    for col in OHLCV_OPTIONAL:
        if col not in df.columns:
            df[col] = pd.NA

    # 标准列在前，额外列按字母序在后（保留所有原始信息）
    standard = list(OHLCV_REQUIRED) + list(OHLCV_OPTIONAL)
    extras = sorted(c for c in df.columns if c not in standard)
    return df[standard + extras]


def validate_required(
    df: pd.DataFrame, required: tuple[str, ...], label: str = "df"
) -> pd.DataFrame:
    """通用必需列校验 (空 df 透传)。

    Args:
        df: 待校验 DataFrame
        required: 必需列元组
        label: 错误信息中显示的名字

    Raises:
        ValueError: 必需列缺失时
    """
    if df.empty:
        return df
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} 缺少必需列: {missing}")
    return df
