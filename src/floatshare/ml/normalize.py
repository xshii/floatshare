"""Cross-sectional 标准化 — 每天对每个特征做截面 z-score。

策略:
    1. 每个 trade_date 内, 对每个特征算 cross-sectional mean / std
    2. z = (x - mean) / std
    3. clip 到 [-clip_thresh, clip_thresh] 防异常值
    4. NaN → 0 (z-score 后 0 是中位数, 视作"中性")

为什么不做 time-series z-score:
    - 市场风格漂移大 (低利率 vs 高利率年代)
    - cross-sectional 自带"相对市场"语义, agent 学的是"哪只股比同类强"
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd


def cross_sectional_zscore(
    feats: pd.DataFrame,
    clip_thresh: float = 5.0,
) -> pd.DataFrame:
    """长 format DataFrame (index 含 trade_date 列), 按 trade_date 截面标准化。

    Args:
        feats: index 必须含 trade_date 列 (作为索引或普通列均可)
        clip_thresh: |z| 截尾阈值

    Returns:
        同 shape, 每个 trade_date 内所有 code 截面标准化后的值
    """
    if feats.empty:
        return feats
    df = feats.copy()
    if "trade_date" not in df.columns and df.index.name == "trade_date":
        df = df.reset_index()
    elif "trade_date" not in df.columns:
        # 假设 index 是 (trade_date,) 或 (code, trade_date)
        df = df.reset_index()

    feature_cols = [
        c for c in df.columns if c not in ("code", "trade_date") and df[c].dtype != object
    ]

    # 按 trade_date 分组做 z-score (vectorized)
    g = df.groupby("trade_date")[feature_cols]
    mean = g.transform("mean")
    std = g.transform("std").replace(0, np.nan)
    z = (df[feature_cols] - mean) / std
    z = cast(pd.DataFrame, z.clip(lower=-clip_thresh, upper=clip_thresh)).fillna(0.0)

    out = cast(pd.DataFrame, df[[c for c in ("code", "trade_date") if c in df.columns]].copy())
    out[feature_cols] = z
    return out
