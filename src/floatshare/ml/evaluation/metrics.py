"""纯 numpy 指标 — AUC / Sharpe / Drawdown / Turnover / Top-K precision.

所有函数无状态、无副作用, 可直接单测.
"""

from __future__ import annotations

import numpy as np


def rank_based_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U statistic), 比 sklearn 快 + 无 sklearn 依赖."""
    if len(y) < 2 or y.sum() in (0, len(y)):
        return 0.5
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    sum_pos_rank = float(ranks[y == 1].sum())
    return float((sum_pos_rank - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def top_k_precision(
    probs: np.ndarray,
    labels: np.ndarray,
    top_k: int,
) -> float:
    """每天 top-K (按 prob) 实际命中率, 跨天求均值.

    Args:
        probs: (n_days, n_tokens) — P(hit)
        labels: (n_days, n_tokens) — 1/0/-1 (-1 = invalid)
        top_k: 每天取 top K 个 prob 最高的

    Returns:
        total_hits / total_picks (聚合, 不是 per-day 均值).
    """
    pk_hits, pk_total = 0, 0
    for d in range(probs.shape[0]):
        valid_d = labels[d] >= 0
        if valid_d.sum() < top_k:
            continue
        p_d = probs[d][valid_d]
        y_d = labels[d][valid_d]
        top_idx = np.argpartition(p_d, -top_k)[-top_k:]
        pk_hits += int(y_d[top_idx].sum())
        pk_total += top_k
    return pk_hits / pk_total if pk_total else 0.0


def compute_sharpe(rewards: np.ndarray, K: int = 5) -> float:
    """K-day step 序列的年化 Sharpe (假设 252 交易日 / K 步 per year)."""
    if len(rewards) < 2:
        return 0.0
    return float(rewards.mean() / (rewards.std() + 1e-8) * np.sqrt(252 / K))


def compute_max_drawdown(rewards: np.ndarray) -> float:
    """累计收益的最大回撤 (负值, 0 表示无回撤)."""
    if len(rewards) < 2:
        return 0.0
    cum = np.cumsum(rewards)
    peak = np.maximum.accumulate(cum)
    return float((cum - peak).min())


def compute_turnover_avg(weights_history: list[np.ndarray]) -> float:
    """相邻步权重 L1 差均值 (换手率)."""
    if len(weights_history) < 2:
        return 0.0
    ws = np.stack(weights_history)
    return float(np.abs(np.diff(ws, axis=0)).sum(axis=1).mean())
