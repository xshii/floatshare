"""GRPO 单测 — 重点验证 group_advantage 数值 + GRPOConfig 默认值."""

from __future__ import annotations

import pytest
import torch

from floatshare.ml.config import GRPOConfig
from floatshare.ml.rl.grpo import group_advantage


class TestGroupAdvantage:
    def test_shape_preserved_1d(self):
        r = torch.arange(16, dtype=torch.float32)
        adv = group_advantage(r, group_size=4)
        assert adv.shape == r.shape

    def test_each_group_zero_mean(self):
        """每组减均值后, 组内均值应 ≈ 0."""
        r = torch.tensor([1.0, 2, 3, 4, 10, 20, 30, 40, -1, -2, -3, -4])
        adv = group_advantage(r, group_size=4)
        groups = adv.view(-1, 4)
        for g in groups:
            assert abs(g.mean().item()) < 1e-5, f"group {g} mean should be 0"

    def test_each_group_unit_std(self):
        """除 biased std 后每组 biased std = 1 (除数就是 biased std 自己)."""
        r = torch.tensor([1.0, 2, 3, 4, 10, 20, 30, 40])
        adv = group_advantage(r, group_size=4, eps=0.0)
        groups = adv.view(-1, 4)
        for g in groups:
            # correction=0 (biased) 归一化后, biased std 应 ≈ 1
            assert abs(g.std(correction=0).item() - 1.0) < 1e-4

    def test_constant_group_returns_zero(self):
        """组内所有值相等 → advantage = 0 (eps 保护分母)."""
        r = torch.tensor([5.0, 5, 5, 5, 7.0, 7, 7, 7])
        adv = group_advantage(r, group_size=4, eps=1e-6)
        # 所有值应极小 (数值等同 0)
        assert adv.abs().max().item() < 1e-3

    def test_cross_group_independence(self):
        """A 组极大值不影响 B 组 advantage (组间完全独立)."""
        r = torch.tensor([1000.0, 2000, 3000, 4000, 1.0, 2, 3, 4])
        adv = group_advantage(r, group_size=4)
        # A/B 两组各自中心化后, advantage 数值应接近 (同样的 rank pattern)
        adv_a = adv[:4]
        adv_b = adv[4:]
        # 两组应有相同 sign + 相似量级 (因为 rank 一致)
        assert torch.allclose(adv_a, adv_b, atol=0.1)

    def test_raises_on_indivisible(self):
        r = torch.arange(10, dtype=torch.float32)
        with pytest.raises(ValueError, match="不能被 group_size"):
            group_advantage(r, group_size=3)

    def test_group_size_1_is_identity_up_to_centering(self):
        """G=1 时每组只有 1 个, mean=value 自身 → advantage 始终 0."""
        r = torch.tensor([1.0, 2, 3, 4])
        adv = group_advantage(r, group_size=1, eps=1e-6)
        assert adv.abs().max().item() < 1e-3


class TestGRPOConfig:
    def test_defaults_sensible(self):
        cfg = GRPOConfig()
        assert cfg.group_size == 8
        assert 0 < cfg.clip_ratio < 1
        assert cfg.kl_coef > 0, "KL penalty 必须 > 0, 否则防漂失效"
        assert cfg.update_epochs >= 1
        assert cfg.rollout_days > 0

    def test_frozen_dataclass(self):
        """GRPOConfig 应 frozen, 防意外改参数."""
        cfg = GRPOConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.group_size = 16  # type: ignore[misc]

    def test_reward_horizon_matches_ppo(self):
        """与 MarketEnv 接口兼容 — 必须有 reward_horizon / turnover_penalty 字段."""
        cfg = GRPOConfig()
        assert hasattr(cfg, "reward_horizon")
        assert hasattr(cfg, "turnover_penalty")
