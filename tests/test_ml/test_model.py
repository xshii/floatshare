"""模型层单测 — RMSNorm + Attention + Encoder + Heads + ActorCritic。

主要验证:
    1. 形状不变量 (B, N, T, F → 各 head 输出)
    2. mask 生效 (停盘 → weight = 0)
    3. softmax 性质 (sum = 1, 非负)
    4. ckpt save/load 一致性 (含 Phase 1 → Phase 2 backbone 迁移)
"""

from __future__ import annotations

import pytest
import torch

from floatshare.ml.config import ModelConfig
from floatshare.ml.model.agent import ActorCritic
from floatshare.ml.model.heads import HierarchicalHead, IndustryHead
from floatshare.ml.model.norm import RMSNorm

# === RMSNorm ===


def test_rmsnorm_shape_and_magnitude() -> None:
    """RMS 归一化后, 沿最后维 RMS ≈ scale (默认 scale=1)."""
    norm = RMSNorm(dim=8)
    x = torch.randn(4, 16, 8) * 5
    y = norm(x)
    assert y.shape == x.shape
    rms = y.pow(2).mean(dim=-1).sqrt()
    # 误差容忍 5% (eps + 学习参数初值)
    assert (rms - 1.0).abs().max() < 0.05


# === Heads ===


def test_industry_head_softmax_sums_to_one() -> None:
    cfg = ModelConfig(embed_dim=16, n_industries=31, phase=1)
    head = IndustryHead(cfg)
    h = torch.randn(2, 31, 16)
    mask = torch.ones(2, 31, dtype=torch.bool)
    out = head(h, mask=mask)
    w = out.weights
    assert w.shape == (2, 31)
    torch.testing.assert_close(w.sum(dim=-1), torch.ones(2), atol=1e-5, rtol=1e-5)
    assert (w >= 0).all()


def test_industry_head_mask_zeros_out() -> None:
    """mask=False 的位置 weight 必须 = 0."""
    cfg = ModelConfig(embed_dim=16, n_industries=31, phase=1)
    head = IndustryHead(cfg)
    h = torch.randn(1, 31, 16)
    mask = torch.ones(1, 31, dtype=torch.bool)
    mask[0, 5] = False
    mask[0, 10] = False
    out = head(h, mask=mask)
    assert out.weights[0, 5].item() == 0.0
    assert out.weights[0, 10].item() == 0.0
    torch.testing.assert_close(out.weights[0].sum(), torch.tensor(1.0), atol=1e-5, rtol=1e-5)


def test_hierarchical_head_industry_x_stock() -> None:
    """Phase 2: stock 权重 = 顶层 industry × 底层 stock-in-industry."""
    cfg = ModelConfig(embed_dim=16, n_industries=4, phase=2)  # 缩小测
    head = HierarchicalHead(cfg)
    n_stocks = 8
    n_total = cfg.n_industries + n_stocks
    h = torch.randn(1, n_total, 16)
    token_types = (
        torch.cat([torch.zeros(cfg.n_industries), torch.ones(n_stocks)]).long().unsqueeze(0)
    )
    # stock_industry_ids: 0,0,1,1,2,2,3,3 (each industry has 2 stocks)
    stock_ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    industry_ids = torch.cat([torch.arange(cfg.n_industries), stock_ids]).unsqueeze(0)
    mask = torch.ones(1, n_total, dtype=torch.bool)

    out = head(h, token_types=token_types, industry_ids=industry_ids, mask=mask)
    w_ind = out.industry_weights
    w_stock = out.stock_weights

    torch.testing.assert_close(w_ind.sum(dim=-1), torch.ones(1), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(w_stock.sum(dim=-1), torch.ones(1), atol=1e-5, rtol=1e-5)
    assert (w_stock >= 0).all()


def test_hierarchical_head_mask_excludes_suspended() -> None:
    """停盘 stock 的最终权重 = 0."""
    cfg = ModelConfig(embed_dim=16, n_industries=4, phase=2)
    head = HierarchicalHead(cfg)
    n_stocks = 4
    n_total = cfg.n_industries + n_stocks
    h = torch.randn(1, n_total, 16)
    token_types = (
        torch.cat([torch.zeros(cfg.n_industries), torch.ones(n_stocks)]).long().unsqueeze(0)
    )
    industry_ids = torch.cat(
        [torch.arange(cfg.n_industries), torch.tensor([0, 0, 1, 2])]
    ).unsqueeze(0)
    mask = torch.ones(1, n_total, dtype=torch.bool)
    mask[0, cfg.n_industries + 0] = False  # 停盘第 1 只 stock (industry 0)

    out = head(h, token_types=token_types, industry_ids=industry_ids, mask=mask)
    assert out.stock_weights[0, 0].item() == 0.0  # 该 stock weight = 0
    torch.testing.assert_close(out.stock_weights.sum(dim=-1), torch.ones(1), atol=1e-5, rtol=1e-5)


# === ActorCritic 端到端 ===


@pytest.fixture
def sample_inputs() -> dict[str, torch.Tensor]:
    """合成 1 batch (B=2, N=31, T=seq_len, F=n_features)."""
    cfg = ModelConfig()
    B, N, T, F = 2, 31, cfg.seq_len, cfg.n_features
    return {
        "x": torch.randn(B, N, T, F),
        "token_types": torch.zeros(B, N, dtype=torch.long),  # all industry
        "industry_ids": torch.arange(N).unsqueeze(0).expand(B, N).long(),
        "mask": torch.ones(B, N, dtype=torch.bool),
    }


def test_actor_critic_phase1_forward(sample_inputs) -> None:
    cfg = ModelConfig(phase=1, embed_dim=32, n_heads=4, n_layers_temporal=2, n_layers_cross=1)
    model = ActorCritic(cfg)
    out = model(**sample_inputs)
    assert out.weights.shape == (2, 31)
    assert out.value.shape == (2,)
    torch.testing.assert_close(out.weights.sum(dim=-1), torch.ones(2), atol=1e-5, rtol=1e-5)


def test_actor_critic_phase2_forward() -> None:
    cfg = ModelConfig(
        phase=2, embed_dim=32, n_heads=4, n_layers_temporal=2, n_layers_cross=1, n_industries=31
    )
    model = ActorCritic(cfg)
    B, N_stocks = 2, 10
    T, F = cfg.seq_len, cfg.n_features
    N = cfg.n_industries + N_stocks
    inputs = {
        "x": torch.randn(B, N, T, F),
        "token_types": torch.cat(
            [
                torch.zeros(cfg.n_industries),
                torch.ones(N_stocks),
            ]
        )
        .long()
        .unsqueeze(0)
        .expand(B, N)
        .contiguous(),
        "industry_ids": torch.cat(
            [
                torch.arange(cfg.n_industries),
                torch.randint(0, cfg.n_industries, (N_stocks,)),
            ]
        )
        .unsqueeze(0)
        .expand(B, N)
        .contiguous(),
        "mask": torch.ones(B, N, dtype=torch.bool),
    }
    out = model(**inputs)
    assert out.industry_weights.shape == (B, cfg.n_industries)
    assert out.stock_weights.shape == (B, N_stocks)
    assert out.value.shape == (B,)
    torch.testing.assert_close(
        out.industry_weights.sum(dim=-1), torch.ones(B), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(out.stock_weights.sum(dim=-1), torch.ones(B), atol=1e-5, rtol=1e-5)


def test_actor_critic_backbone_transfer(tmp_path) -> None:
    """Phase 1 ckpt → Phase 2 backbone 加载 (single_axis 模式)."""
    cfg1 = ModelConfig(
        phase=1,
        encoder_mode="single_axis",
        embed_dim=32,
        n_heads=4,
        n_layers_temporal=2,
        n_layers_cross=1,
    )
    model1 = ActorCritic(cfg1)
    ckpt = tmp_path / "p1.pt"
    model1.save(ckpt)

    cfg2 = ModelConfig(
        phase=2,
        encoder_mode="single_axis",
        embed_dim=32,
        n_heads=4,
        n_layers_temporal=2,
        n_layers_cross=1,
        pretrained_backbone=str(ckpt),
    )
    model2 = ActorCritic(cfg2)

    # backbone 权重应一致
    for k in model1.temporal_enc.state_dict():
        torch.testing.assert_close(
            model1.temporal_enc.state_dict()[k],
            model2.temporal_enc.state_dict()[k],
        )
    for k in model1.cross_enc.state_dict():
        torch.testing.assert_close(
            model1.cross_enc.state_dict()[k],
            model2.cross_enc.state_dict()[k],
        )


def test_param_count_reasonable() -> None:
    """默认 cfg 下参数量在合理范围 (M4 训得动)."""
    cfg = ModelConfig()  # 默认: embed=64, heads=4, layers 3+2
    model = ActorCritic(cfg)
    n = model.n_params()
    # ~100K-1M 参数, M4 训练快
    assert 50_000 < n < 5_000_000, f"参数量 {n} 不在预期范围"
