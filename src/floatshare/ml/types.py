"""ml/ 共享数据类型 — 替代散落的字符串 dict key。

EnvState              — env._state() 输出, 喂 ActorCritic.forward
IndustryActionOut     — Phase 1 actor 输出
HierarchicalActionOut — Phase 2 actor 输出
ActionOut             — 上面二者的 Union (PPO/rollout dispatch 用)
PPOMetrics            — PPO update 一轮的训练指标
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from torch import Tensor


@dataclass(frozen=True, slots=True)
class EnvState:
    """env._state() 输出 (单步 state, batch 维由调用方加)。"""

    x: np.ndarray  # (N, T, F) float32 features
    token_types: np.ndarray  # (N,) int64, 0=industry / 1=stock
    industry_ids: np.ndarray  # (N,) int64, 0..n_industries-1
    mask: np.ndarray  # (N,) bool, True=可交易 (非停盘)


@dataclass(frozen=True, slots=True)
class IndustryActionOut:
    """Phase 1: 仅行业权重."""

    weights: Tensor  # (B, N) softmax 概率, 行权重和=1
    logits: Tensor  # (B, N) 原始 logit, masked = -inf
    value: Tensor  # (B,) state value V(s)


@dataclass(frozen=True, slots=True)
class HierarchicalActionOut:
    """Phase 2: 顶层 industry × 底层 stock 复合."""

    industry_weights: Tensor  # (B, n_industries) 顶层 softmax
    stock_weights: Tensor  # (B, n_stocks) 最终 per-stock 权重 (顶 × 底, sum=1)
    ind_logits: Tensor  # (B, n_industries) 行业 logit
    stock_logits: Tensor  # (B, n_stocks) 个股 logit
    value: Tensor  # (B,)


@dataclass(frozen=True, slots=True)
class PopActionOut:
    """Phase 3 抓涨停 (1 天 5% 超短线).

    输出语义 (方案 C — 单 head 4 类 timing+sizing):
        action_probs[0] = P(不买)
        action_probs[1] = P(买 top1, 100%)
        action_probs[2] = P(买 top1+top2, 60/40)
        action_probs[3] = P(买 top1+top2+top3, 50/33/17)

    选股:
        stock_probs = softmax over N 股 → 排序选 top-N (N 由 action 决定)

    监督 (Stage A):
        p_hit = P(D+1→D+2 开盘涨 ≥5%) per 股, BCE 训练
    """

    action_logits: Tensor  # (B, 4) 原始 logits — PPO log_prob 用
    action_probs: Tensor  # (B, 4) softmax — 推理用
    stock_logits: Tensor  # (B, N) 个股排序 logit
    stock_probs: Tensor  # (B, N) softmax — 排序 / top-K
    p_hit: Tensor  # (B, N) sigmoid → P(hit) (BCE 监督)
    value: Tensor  # (B,)


# 类型别名 — 用于 forward / rollout / ppo 的返回签名
ActionOut = IndustryActionOut | HierarchicalActionOut | PopActionOut


@dataclass(frozen=True, slots=True)
class PPOMetrics:
    """PPO update 一次完整 update_epochs 跑完的平均指标."""

    policy_loss: float
    value_loss: float
    entropy: float
    kl: float
