"""GRPO (Group Relative Policy Optimization) — Shao et al. DeepSeek-Math 2024.

核心 vs PPO:
    - 扔 critic (无 value loss, 无 GAE)
    - 每个 state 采 G 个 action, 组内 (reward - mean) / std 当 advantage
    - 天然适合金融 RL: V(s) 难估 / 一日 G 个 portfolio 比较正好归一化市场层共振

实现约定:
    - group_advantage 按 "组" 维度归一化 (G 轴), 不跨时间
    - KL penalty 对 reference policy (可用 initial ckpt 或 EMA), 防漂
    - 无 GAE → 每个 reward 直接就是当步 advantage 原料
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from floatshare.ml.rl.rollout import make_dist_from_logits, select_action_logits
from floatshare.ml.types import PPOMetrics

if TYPE_CHECKING:
    from floatshare.ml.config import GRPOConfig, ModelConfig
    from floatshare.ml.model.agent import ActorCritic
    from floatshare.ml.rl.rollout import RolloutBatch


def group_advantage(rewards: Tensor, group_size: int, eps: float = 1e-6) -> Tensor:
    """组内归一化 advantage.

    Input shape: (N, ...) 其中 N 能被 group_size 整除.
        典型: (T * G,) — T 个 state, 每个 state 采 G 个 action 展平.
    Output: 同 shape, 每组内减均值除标准差.

    公式 (Shao et al. 2024):
        A_i = (R_i - mean_group(R)) / (std_group(R) + eps)

    这是 GRPO 唯一与 PPO 不同的数值步骤 — 其它 (clip, KL) 都兼容.
    """
    n = rewards.shape[0]
    if n % group_size != 0:
        raise ValueError(f"rewards 长度 {n} 不能被 group_size {group_size} 整除")
    groups = rewards.view(-1, group_size, *rewards.shape[1:])  # (N/G, G, ...)
    mean = groups.mean(dim=1, keepdim=True)
    # correction=0 (biased std) — group_size=1 时返回 0 而非 NaN, 配合 eps 防 /0
    std = groups.std(dim=1, keepdim=True, correction=0)
    adv = (groups - mean) / (std + eps)
    return adv.view(n, *rewards.shape[1:])


def grpo_update(
    model: ActorCritic,
    ref_model: ActorCritic | None,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    cfg: GRPOConfig,
    model_cfg: ModelConfig,
    device: torch.device,
) -> PPOMetrics:
    """GRPO policy update — clip + KL penalty, 无 value loss.

    batch.rewards 形状 (T*G,), 其中 T = rollout_days, G = cfg.group_size.
    states 列表按 [s_0_g0, s_0_g1, ..., s_0_gG-1, s_1_g0, ...] 展平 (T*G 项).

    ref_model 可为 None — KL term 用 zero 占位 (首次训练无参考).
    """
    prepared = _prepare_batch(batch, cfg, device)
    n = prepared.advantages.shape[0]
    pols, kls, ents = [], [], []

    for _ in range(cfg.update_epochs):
        idx = torch.randperm(n, device=device)
        mb_size = max(1, n // 4)  # 4 mini-batch 经验默认
        for start in range(0, n, mb_size):
            mb = idx[start : start + mb_size]
            losses = _compute_minibatch_loss(
                model,
                ref_model,
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
            kls.append(losses["kl"].item())
            ents.append(losses["entropy"].item())

    return PPOMetrics(
        policy_loss=float(np.mean(pols)),
        value_loss=0.0,  # GRPO 无 value loss, 保持 dataclass 兼容
        entropy=float(np.mean(ents)),
        kl=float(np.mean(kls)),
    )


# --- helpers -----------------------------------------------------------------


class _PreparedBatch:
    __slots__ = (
        "actions",
        "advantages",
        "old_log_probs",
        "state_ind",
        "state_mask",
        "state_tt",
        "state_x",
    )

    def __init__(
        self,
        state_x: Tensor,
        state_tt: Tensor,
        state_ind: Tensor,
        state_mask: Tensor,
        actions: Tensor,
        old_log_probs: Tensor,
        advantages: Tensor,
    ) -> None:
        self.state_x = state_x
        self.state_tt = state_tt
        self.state_ind = state_ind
        self.state_mask = state_mask
        self.actions = actions
        self.old_log_probs = old_log_probs
        self.advantages = advantages


def _prepare_batch(batch: RolloutBatch, cfg: GRPOConfig, device: torch.device) -> _PreparedBatch:
    """rewards → 组内归一化 advantage; stack states 到 device."""
    rewards = batch.rewards.to(device)
    actions = batch.actions.to(device)
    old_log_probs = batch.log_probs.to(device)

    advantages = group_advantage(rewards, cfg.group_size, cfg.adv_eps)

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
        advantages=advantages,
    )


def _compute_minibatch_loss(
    model: ActorCritic,
    ref_model: ActorCritic | None,
    p: _PreparedBatch,
    mb: Tensor,
    cfg: GRPOConfig,
    model_cfg: ModelConfig,
) -> dict[str, Tensor]:
    """GRPO loss = clipped PG + β · KL(policy || ref) - entropy_bonus · H."""
    out = model(p.state_x[mb], p.state_tt[mb], p.state_ind[mb], p.state_mask[mb])
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

    # Clipped surrogate (同 PPO)
    ratio = (new_log_prob - p.old_log_probs[mb]).clamp(-20.0, 20.0).exp()
    surr1 = ratio * p.advantages[mb]
    surr2 = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * p.advantages[mb]
    policy_loss = -torch.min(surr1, surr2).mean()

    # KL to reference (GRPO 特有, 无 ref_model 时为 0)
    if ref_model is not None:
        with torch.no_grad():
            ref_out = ref_model(p.state_x[mb], p.state_tt[mb], p.state_ind[mb], p.state_mask[mb])
            ref_logits = select_action_logits(ref_out)
            ref_dist = make_dist_from_logits(ref_logits, mask_for_dist)
            ref_log_prob = ref_dist.log_prob(actions_for_lp)
        kl = (new_log_prob - ref_log_prob).mean()
    else:
        kl = torch.zeros((), device=p.advantages.device)

    entropy_loss = -cfg.entropy_bonus * entropy
    total = policy_loss + cfg.kl_coef * kl + entropy_loss

    return {
        "total": total,
        "policy": policy_loss,
        "kl": kl,
        "entropy": entropy,
    }
