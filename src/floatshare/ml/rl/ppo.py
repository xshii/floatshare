"""PPO update — clipped surrogate + GAE + value loss clip + entropy bonus.

参考: Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
     Engstrom et al. "Implementation Matters in Deep RL" (2020)

业界标准做法:
    ✓ GAE 全 tensor 化 (无 .item() GPU sync)
    ✓ ratio = (new_lp - old_lp).clamp(-20, 20).exp()  数值稳定
    ✓ value loss clip (PPO2 trick): max(L_v, L_v_clipped)
    ✓ minibatch states 一次性 stack 成 tensor, mini-batch 间复用
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from floatshare.ml.rl.rollout import make_dist_from_logits, select_action_logits
from floatshare.ml.types import PPOMetrics

if TYPE_CHECKING:
    from floatshare.ml.config import ModelConfig, PPOConfig
    from floatshare.ml.model.agent import ActorCritic
    from floatshare.ml.rl.rollout import RolloutBatch


@dataclass(frozen=True, slots=True)
class _PreparedBatch:
    """PPO update 一次循环要用的所有 device tensor."""

    state_x: Tensor
    state_tt: Tensor
    state_ind: Tensor
    state_mask: Tensor
    actions: Tensor
    old_log_probs: Tensor
    old_values: Tensor
    returns: Tensor
    adv_norm: Tensor


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    gamma: float,
    lam: float,
    last_value: float = 0.0,
) -> Tensor:
    """Generalized Advantage Estimation (Schulman et al. 2016).

    全 tensor in-place 倒序累加, 不 .item() — 比 250 次 GPU sync 快 ~50ms.
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    nonterminal = (~dones).float()

    next_v = torch.tensor(last_value, dtype=rewards.dtype, device=rewards.device)
    next_adv = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * nonterminal[t] * next_v - values[t]
        advantages[t] = delta + gamma * lam * nonterminal[t] * next_adv
        next_v = values[t]
        next_adv = advantages[t]
    return advantages


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    cfg: PPOConfig,
    model_cfg: ModelConfig,
    device: torch.device,
) -> PPOMetrics:
    """跑一次 PPO update (update_epochs × minibatch SGD over 同一 rollout)."""
    prepared = _prepare_batch(batch, cfg, device)
    n = prepared.returns.shape[0]
    pols, vals, ents, kls = [], [], [], []

    for _ in range(cfg.update_epochs):
        idx = torch.randperm(n, device=device)
        for start in range(0, n, cfg.minibatch_size):
            mb = idx[start : start + cfg.minibatch_size]
            losses = _compute_minibatch_loss(
                model,
                prepared,
                mb,
                cfg,
                model_cfg,
            )
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            pols.append(losses["policy"].item())
            vals.append(losses["value"].item())
            ents.append(losses["entropy"].item())
            kls.append(losses["kl"].item())

    return PPOMetrics(
        policy_loss=float(np.mean(pols)),
        value_loss=float(np.mean(vals)),
        entropy=float(np.mean(ents)),
        kl=float(np.mean(kls)),
    )


# --- helpers -----------------------------------------------------------------


def _prepare_batch(
    batch: RolloutBatch,
    cfg: PPOConfig,
    device: torch.device,
) -> _PreparedBatch:
    """全 rollout 数据迁到 device + GAE + advantage 标准化 + stack states."""
    rewards = batch.rewards.to(device)
    values = batch.values.to(device)
    dones = batch.dones.to(device)
    actions = batch.actions.to(device)
    old_log_probs = batch.log_probs.to(device)

    advantages = compute_gae(
        rewards,
        values,
        dones,
        cfg.gamma,
        cfg.gae_lambda,
        last_value=values[-1].item(),
    )
    returns = advantages + values
    adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    states = batch.states
    state_x = torch.from_numpy(np.stack([s.x for s in states])).to(device)
    state_tt = torch.from_numpy(np.stack([s.token_types for s in states])).to(device)
    state_ind = torch.from_numpy(np.stack([s.industry_ids for s in states])).to(device)
    state_mask = torch.from_numpy(np.stack([s.mask for s in states])).to(device)

    return _PreparedBatch(
        state_x=state_x,
        state_tt=state_tt,
        state_ind=state_ind,
        state_mask=state_mask,
        actions=actions,
        old_log_probs=old_log_probs,
        old_values=values.detach(),
        returns=returns,
        adv_norm=adv_norm,
    )


def _compute_minibatch_loss(
    model: ActorCritic,
    p: _PreparedBatch,
    mb: Tensor,
    cfg: PPOConfig,
    model_cfg: ModelConfig,
) -> dict[str, Tensor]:
    """一个 mini-batch 的 policy + value + entropy loss. 不 step optimizer."""
    out = model(p.state_x[mb], p.state_tt[mb], p.state_ind[mb], p.state_mask[mb])

    # policy loss
    logits = select_action_logits(out)
    mask_for_dist = (
        p.state_mask[mb] if model_cfg.phase == 1 else p.state_mask[mb][:, : model_cfg.n_industries]
    )
    dist = make_dist_from_logits(logits, mask_for_dist)
    actions_for_lp = (
        p.actions[mb] if model_cfg.phase == 1 else p.actions[mb][:, : model_cfg.n_industries]
    )
    new_log_prob = dist.log_prob(actions_for_lp)
    entropy = dist.entropy().mean()

    ratio = (new_log_prob - p.old_log_probs[mb]).clamp(-20.0, 20.0).exp()
    surr1 = ratio * p.adv_norm[mb]
    surr2 = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * p.adv_norm[mb]
    policy_loss = -torch.min(surr1, surr2).mean()

    # value loss (PPO2 clip)
    v_pred = out.value
    v_clipped = p.old_values[mb] + (v_pred - p.old_values[mb]).clamp(
        -cfg.clip_ratio,
        cfg.clip_ratio,
    )
    vl_unclipped = (v_pred - p.returns[mb]).pow(2)
    vl_clipped = (v_clipped - p.returns[mb]).pow(2)
    value_loss = torch.max(vl_unclipped, vl_clipped).mean()

    entropy_loss = -cfg.entropy_bonus * entropy
    total = policy_loss + cfg.value_coef * value_loss + entropy_loss

    with torch.no_grad():
        kl = (p.old_log_probs[mb] - new_log_prob).mean()

    return {
        "total": total,
        "policy": policy_loss,
        "value": value_loss,
        "entropy": entropy,
        "kl": kl,
    }
