"""端到端 GRPO smoke test (合成数据) — 验证 group rollout + grpo_update 全链路不崩.

与 test_ppo_smoke.py 结构对齐, 5 iter 合成数据训练.
断言:
    - group rollout 输出 shape = T*G
    - batch.states 按 [s_0*G, s_1*G, ...] 展平 (同 state 重复 G 次)
    - grpo_update 不崩 + KL 在控
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from floatshare.ml.config import GRPOConfig, ModelConfig
from floatshare.ml.data.dataset import MarketCube, TokenMeta
from floatshare.ml.model.agent import ActorCritic
from floatshare.ml.rl.env import MarketEnv
from floatshare.ml.rl.grpo import grpo_update
from floatshare.ml.training.grpo import _collect_group_rollout

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _cpu() -> torch.device:
    return torch.device("cpu")


def _build_env_and_model(
    n_days: int = 100, n_ind: int = 5, seq_len: int = 10, F: int = 22
) -> tuple[MarketEnv, ActorCritic, ModelConfig, GRPOConfig]:
    rng = np.random.default_rng(0)
    log_ret = rng.normal(0.001, 0.02, (n_days, n_ind)).astype(np.float32)
    prices = (10 * np.exp(np.cumsum(log_ret, axis=0))).astype(np.float32)
    features = rng.normal(0, 1, (n_days, n_ind, F)).astype(np.float32)
    traded = np.ones((n_days, n_ind), dtype=bool)
    tokens = [TokenMeta(f"I{i}", 0, i) for i in range(n_ind)]
    cube = MarketCube(
        dates=np.arange(n_days),
        tokens=tokens,
        features=features,
        prices=prices,
        traded=traded,
    )
    mcfg = ModelConfig(
        phase=1,
        n_industries=n_ind,
        seq_len=seq_len,
        n_features=F,
        embed_dim=16,
        n_heads=2,
        n_layers_temporal=1,
        n_layers_cross=1,
    )
    grpo_cfg = GRPOConfig(reward_horizon=5, group_size=4, rollout_days=10, update_epochs=1)
    model = ActorCritic(mcfg).to(_cpu())
    env = MarketEnv(cube, grpo_cfg, mcfg)  # type: ignore[arg-type]
    return env, model, mcfg, grpo_cfg


def test_group_rollout_shape_is_T_times_G() -> None:
    env, model, mcfg, cfg = _build_env_and_model()
    n_steps = 8
    batch = _collect_group_rollout(
        env, model, n_steps=n_steps, group_size=cfg.group_size, device=_cpu(), model_cfg=mcfg
    )
    expected = n_steps * cfg.group_size
    assert batch.rewards.shape[0] == expected
    assert batch.actions.shape[0] == expected
    assert batch.log_probs.shape[0] == expected
    assert len(batch.states) == expected


def test_group_rollout_same_state_within_group() -> None:
    """每 G 条的 state 应该是同一个 (peek 不 mutate env)."""
    env, model, mcfg, cfg = _build_env_and_model()
    batch = _collect_group_rollout(
        env, model, n_steps=5, group_size=cfg.group_size, device=_cpu(), model_cfg=mcfg
    )
    G = cfg.group_size
    # 每组 G 条 state 应指向同一对象 (_collect_group_rollout 中 append 了 G 次同一 state)
    for step in range(5):
        group_slice = batch.states[step * G : (step + 1) * G]
        for s in group_slice[1:]:
            assert s is group_slice[0], f"step {step}: 组内 state 应相同"


def test_group_rollout_rewards_vary_within_group() -> None:
    """同 state 采 G 个不同 action → reward 应不完全相同 (Dirichlet stochastic)."""
    env, model, mcfg, cfg = _build_env_and_model()
    batch = _collect_group_rollout(
        env, model, n_steps=3, group_size=cfg.group_size, device=_cpu(), model_cfg=mcfg
    )
    G = cfg.group_size
    for step in range(3):
        group_rewards = batch.rewards[step * G : (step + 1) * G]
        # 极小概率所有 reward 完全一致, 保留 0 容差只要有 std > 0 即可
        assert group_rewards.std().item() > 0, f"step {step}: G 个 reward 全一致, sample 没起作用"


@pytest.mark.slow
def test_grpo_end_to_end_smoke() -> None:
    """5 iter group rollout + grpo_update, 不崩 + KL 有限."""
    env, model, mcfg, cfg = _build_env_and_model()
    device = _cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    last_kl = 0.0
    for _ in range(5):
        batch = _collect_group_rollout(
            env, model, n_steps=10, group_size=cfg.group_size, device=device, model_cfg=mcfg
        )
        m = grpo_update(model, None, optimizer, batch, cfg, mcfg, device)
        assert abs(m.kl) < 10.0  # ref_model=None → kl=0
        assert -10 < m.policy_loss < 10
        last_kl = m.kl
    assert abs(last_kl) < 1.0
