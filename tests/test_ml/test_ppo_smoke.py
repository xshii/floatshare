"""端到端 PPO smoke test (合成数据) — 验证 rollout + ppo_update 全链路不崩。

用 5 个行业 200 天合成 random walk 数据训 5 iter,
断言:
    - 模型 forward / rollout / ppo_update 不抛异常
    - reward 数值合理 (有 std, 不是常数)
    - PPO KL 不爆炸 (< 1.0 表示 ratio 没失控)

不验证收敛 (合成数据无 alpha, 没法收敛); 收敛验证留给真实数据训练。
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from floatshare.ml.config import ModelConfig, PPOConfig
from floatshare.ml.data.dataset import MarketCube, TokenMeta
from floatshare.ml.model.agent import ActorCritic
from floatshare.ml.rl.env import MarketEnv
from floatshare.ml.rl.ppo import ppo_update
from floatshare.ml.rl.rollout import collect_rollout

# Dirichlet 采样在 MPS 上不支持, fallback CPU
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _select_device() -> torch.device:
    """Smoke test 强制 CPU — MPS Dirichlet fallback 在 pytest 进程里不稳."""
    return torch.device("cpu")


@pytest.mark.slow
def test_ppo_end_to_end_smoke() -> None:
    """跑 5 iter rollout + update, 断言数值合理 + 不崩。"""
    n_days, n_ind, F = 200, 5, 22
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
        seq_len=20,
        n_features=F,
        embed_dim=16,
        n_heads=2,
        n_layers_temporal=1,
        n_layers_cross=1,
    )
    ppo_cfg = PPOConfig(reward_horizon=5, rollout_days=20, update_epochs=2, minibatch_size=4)

    device = _select_device()
    model = ActorCritic(mcfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    env = MarketEnv(cube, ppo_cfg, mcfg)

    assert 1_000 < model.n_params() < 100_000

    last_kl = 0.0
    for _ in range(5):
        batch = collect_rollout(env, model, n_steps=20, device=device, model_cfg=mcfg)
        assert batch.rewards.shape[0] == 20
        assert batch.rewards.std().item() > 1e-6, "reward 完全无变化, env 有问题"

        metrics = ppo_update(model, optimizer, batch, ppo_cfg, mcfg, device)
        assert abs(metrics.kl) < 1.0, f"KL 爆炸: {metrics.kl}"
        assert -10 < metrics.policy_loss < 10
        last_kl = metrics.kl

    # 5 iter 后 KL 应该稳定 (PPO clip 在工作)
    assert abs(last_kl) < 0.5
