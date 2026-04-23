"""Dual-axis encoder 单测 — 形状不变量 + mask 生效 + 两种模式都能过 ActorCritic."""

from __future__ import annotations

import torch

from floatshare.ml.config import ModelConfig
from floatshare.ml.model.agent import ActorCritic
from floatshare.ml.model.encoder import DualAxisBlock, DualAxisEncoder


def test_dual_axis_block_shape() -> None:
    cfg = ModelConfig(embed_dim=16, n_heads=2, ff_dim=64, seq_len=10, n_industries=4)
    blk = DualAxisBlock(cfg)
    B, N, T, D = 2, 5, 10, 16
    h = torch.randn(B, N, T, D)
    mask = torch.ones(B, N, dtype=torch.bool)
    out = blk(h, mask)
    assert out.shape == (B, N, T, D)


def test_dual_axis_block_mask_excludes_stocks() -> None:
    """mask=False 的 stock 对其他 stock 的 attention 权重为 0 (通过看对 mask 敏感)."""
    cfg = ModelConfig(embed_dim=16, n_heads=2, ff_dim=64, seq_len=10, n_industries=4)
    blk = DualAxisBlock(cfg)
    B, N, T, D = 1, 5, 10, 16
    torch.manual_seed(0)
    h = torch.randn(B, N, T, D)
    mask_full = torch.ones(B, N, dtype=torch.bool)
    mask_partial = mask_full.clone()
    mask_partial[0, 2] = False  # 屏蔽 stock 2

    out1 = blk(h, mask_full)
    out2 = blk(h, mask_partial)
    # 其他 stock (0, 1, 3, 4) 的 output 应该被 mask 影响 → 与 full-mask 不同
    assert not torch.allclose(out1[0, 0], out2[0, 0])


def test_dual_axis_encoder_shape_and_emb() -> None:
    cfg = ModelConfig(
        embed_dim=16,
        n_heads=2,
        ff_dim=64,
        seq_len=10,
        n_industries=4,
        n_dual_layers=2,
        encoder_mode="dual_axis",
    )
    enc = DualAxisEncoder(cfg)
    B, N, T, F = 2, 5, 10, cfg.n_features
    out = enc(
        torch.randn(B, N, T, F),
        mask=torch.ones(B, N, dtype=torch.bool),
        industry_ids=torch.randint(0, cfg.n_industries, (B, N)),
    )
    assert out.shape == (B, N, cfg.embed_dim)  # last-step pool


def test_actor_critic_dual_axis_mode() -> None:
    cfg = ModelConfig(
        phase=3,
        encoder_mode="dual_axis",
        embed_dim=32,
        n_heads=4,
        n_dual_layers=2,
        seq_len=30,
        n_industries=4,
    )
    model = ActorCritic(cfg)
    assert model.dual_enc is not None
    assert model.temporal_enc is None
    B, N = 2, 8
    out = model(
        x=torch.randn(B, N, cfg.seq_len, cfg.n_features),
        token_types=torch.ones(B, N, dtype=torch.long),
        industry_ids=torch.randint(0, cfg.n_industries, (B, N)),
        mask=torch.ones(B, N, dtype=torch.bool),
    )
    assert out.action_probs.shape == (B, 4)
    assert out.stock_probs.shape == (B, N)
    assert out.p_hit.shape == (B, N)


def test_actor_critic_single_axis_mode_still_works() -> None:
    """老的 single_axis 模式未破坏."""
    cfg = ModelConfig(
        phase=3,
        encoder_mode="single_axis",
        embed_dim=16,
        n_heads=2,
        n_layers_temporal=1,
        n_layers_cross=1,
        seq_len=20,
        n_industries=4,
    )
    model = ActorCritic(cfg)
    assert model.temporal_enc is not None
    assert model.dual_enc is None
    B, N = 2, 6
    out = model(
        x=torch.randn(B, N, cfg.seq_len, cfg.n_features),
        token_types=torch.ones(B, N, dtype=torch.long),
        industry_ids=torch.randint(0, cfg.n_industries, (B, N)),
        mask=torch.ones(B, N, dtype=torch.bool),
    )
    assert out.stock_probs.shape == (B, N)


def test_param_count_dual_vs_single() -> None:
    """dual_axis 参数量应该跟 single_axis 同数量级 (不爆炸)."""
    cfg_s = ModelConfig(encoder_mode="single_axis")
    cfg_d = ModelConfig(encoder_mode="dual_axis")
    m_s = ActorCritic(cfg_s)
    m_d = ActorCritic(cfg_d)
    # 允许差 3×, 但不该爆炸
    assert m_d.n_params() < m_s.n_params() * 3
    assert m_d.n_params() > m_s.n_params() * 0.3
