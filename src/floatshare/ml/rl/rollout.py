"""Rollout — 在 MarketEnv 跑 N 步, 收集 (state, action, log_prob, value, reward)。

action 用 Dirichlet 分布采样:
    α = softplus(logits) + 1e-3, masked → α=1e-6
    weights ~ Dirichlet(α)
    log_prob = Dirichlet(α).log_prob(weights)

Phase 1 (IndustryHead) 单层 Dirichlet。Phase 2 (HierarchicalHead) 拆两个
Dirichlet (industry + 每 industry 内 stocks), log_prob 相加 (条件概率独立)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, assert_never

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Dirichlet
from torch.nn import functional as F

from floatshare.ml.types import (
    ActionOut,
    EnvState,
    HierarchicalActionOut,
    IndustryActionOut,
    PopActionOut,
)

if TYPE_CHECKING:
    from floatshare.ml.config import ModelConfig
    from floatshare.ml.model.agent import ActorCritic
    from floatshare.ml.rl.env import MarketEnv


@dataclass
class RolloutBatch:
    """一次 rollout 收集的 (T_steps,) 时序数据 → torch tensors。"""

    states: list[EnvState]  # 长 T_steps
    actions: Tensor  # (T_steps, N) — 采样到的 weights
    log_probs: Tensor  # (T_steps,) — Dirichlet log_prob
    values: Tensor  # (T_steps,)
    rewards: Tensor  # (T_steps,)
    dones: Tensor  # (T_steps,) bool


def state_to_tensors(
    state: EnvState,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """EnvState → batch=1 的 4 个 tensor (model.forward 用)."""
    x = torch.from_numpy(state.x[None]).to(device)
    tt = torch.from_numpy(state.token_types[None]).to(device)
    ind = torch.from_numpy(state.industry_ids[None]).to(device)
    mask = torch.from_numpy(state.mask[None]).to(device)
    return x, tt, ind, mask


def collect_rollout(
    env: MarketEnv,
    model: ActorCritic,
    n_steps: int,
    device: torch.device,
    model_cfg: ModelConfig,
) -> RolloutBatch:
    """跑 n_steps 步, 收集数据。done 时自动 reset。"""
    states: list[EnvState] = []
    actions, log_probs, values, rewards, dones = [], [], [], [], []

    state = env.reset()
    for _ in range(n_steps):
        x, tt, ind, mask = state_to_tensors(state, device)
        with torch.no_grad():
            out: ActionOut = model(x, tt, ind, mask)

        weights, log_prob = _sample_action(
            out,
            mask,
            n_industries=model_cfg.n_industries,
        )
        weights_np = weights[0].cpu().numpy().astype(np.float32)

        reward, next_state, done = env.step(weights_np)

        states.append(state)
        actions.append(weights[0].cpu())
        log_probs.append(log_prob[0].cpu())
        values.append(out.value[0].cpu())
        rewards.append(reward)
        dones.append(done)

        if done:
            state = env.reset()
        else:
            assert next_state is not None, "env 未 done 却返回 None state"
            state = next_state

    return RolloutBatch(
        states=states,
        actions=torch.stack(actions),
        log_probs=torch.stack(log_probs),
        values=torch.stack(values),
        rewards=torch.tensor(rewards, dtype=torch.float32),
        dones=torch.tensor(dones, dtype=torch.bool),
    )


def _sample_action(
    out: ActionOut,
    mask: Tensor,
    n_industries: int,
) -> tuple[Tensor, Tensor]:
    """从模型输出采样 action + 算 log_prob。

    Returns:
        weights: (B, N_total) on simplex, masked 位置 ≈ 0
        log_prob: (B,)
    """
    if isinstance(out, IndustryActionOut):
        return _sample_dirichlet(out.logits, mask)
    if isinstance(out, PopActionOut):
        # Phase 3: 所有 token 是股票, 从 stock_logits 整片采 Dirichlet
        return _sample_dirichlet(out.stock_logits, mask)
    if isinstance(out, HierarchicalActionOut):
        # 顶层 industry + 底层 stock 各自 Dirichlet, log_prob 相加 (条件独立)
        ind_w, ind_lp = _sample_dirichlet(
            out.ind_logits,
            mask[:, :n_industries],
        )
        stock_w, stock_lp = _sample_dirichlet(
            out.stock_logits,
            mask[:, n_industries:],
        )
        return torch.cat([ind_w, stock_w], dim=-1), ind_lp + stock_lp
    assert_never(out)  # 穷尽 ActionOut 的 3 个子类, 不可达


# 兼容旧 dict 入口 (PPO update 需要从更新后的 logits 重新构 dist)
# 仅 Phase 1 需要; Phase 2 需另外 stock 部分


def select_action_logits(out: ActionOut) -> Tensor:
    """提取用于 PPO log_prob 重算的 logits — Phase 1 用 logits, Phase 2 用 ind_logits。

    (Phase 2 目前只用 industry 部分做 PPO update, stock 部分 head 训练通过链式梯度。)
    """
    if isinstance(out, IndustryActionOut):
        return out.logits
    if isinstance(out, HierarchicalActionOut):
        return out.ind_logits
    if isinstance(out, PopActionOut):
        return out.stock_logits  # Phase 3: 股票权重 logits
    assert_never(out)


# Dirichlet alpha clamp 上限 5.0 防止策略坍缩 (alpha 大 → 分布尖 → 探索不足);
# 下限 0.1 保留可学习信号 (alpha < 1 时分布趋近角点, 适合稀疏组合).
_ALPHA_MIN, _ALPHA_MAX = 0.1, 5.0


def _build_alpha(logits: Tensor, mask: Tensor) -> Tensor:
    """logits → Dirichlet 浓度参数 α (clamp 在 [0.1, 5.0])."""
    safe_logits = torch.where(torch.isinf(logits), torch.zeros_like(logits), logits)
    alpha = F.softplus(safe_logits).clamp(min=_ALPHA_MIN, max=_ALPHA_MAX)
    return torch.where(mask, alpha, torch.full_like(alpha, 1e-6))


def _sample_dirichlet(logits: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """logits → Dirichlet(α) → sample weights + log_prob."""
    dist = Dirichlet(_build_alpha(logits, mask))
    weights = dist.sample()
    return weights, dist.log_prob(weights)


def make_dist_from_logits(logits: Tensor, mask: Tensor) -> Dirichlet:
    """从 logits 构 Dirichlet (PPO update 时复用)。"""
    return Dirichlet(_build_alpha(logits, mask))
