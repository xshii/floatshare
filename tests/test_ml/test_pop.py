"""Phase 2 抓涨停: labels + FlatStockHead + ActorCritic phase=3 端到端形状测试."""

from __future__ import annotations

import numpy as np
import torch

from floatshare.ml.config import ModelConfig
from floatshare.ml.labels import HitLabelConfig, label_stats, make_hit_labels
from floatshare.ml.model.agent import ActorCritic
from floatshare.ml.model.heads import FlatStockHead
from floatshare.ml.types import PopActionOut

# === Labels ===


def test_make_hit_labels_basic_hit() -> None:
    """构造 D+1 open=10 → D+2 open=10.5 = +5% 命中 (新默认 1d 5%)."""
    n_days, n_tokens = 10, 2
    opens = np.full((n_days, n_tokens), 10.0, dtype=np.float64)
    highs = np.full((n_days, n_tokens), 100.0, dtype=np.float64)
    # 在 t=2 设 D+2 (即 idx 4) open = 10.5 → 5% 涨幅
    opens[4, 0] = 10.5
    labels = make_hit_labels(opens, highs)  # 用默认 cfg (sell_offset=2, threshold=0.05)
    assert labels[2, 0] == 1
    assert labels[2, 1] == 0


def test_make_hit_labels_yi_zi_excluded() -> None:
    """一字板 (open == high) 应被标 -1 (不可买)."""
    opens = np.array([[10, 11, 11, 12, 11, 12]] * 1, dtype=np.float64).T
    opens = np.tile(opens, (1, 2))  # 2 stocks
    highs = opens.copy()  # open == high, 一字板
    labels = make_hit_labels(opens, highs, HitLabelConfig(exclude_yi_zi=True))
    # 全部 -1
    assert (labels == -1).all()


def test_make_hit_labels_nan_marked_invalid() -> None:
    n_days, n_tokens = 10, 1
    opens = np.full((n_days, n_tokens), 10.0)
    opens[3, 0] = np.nan  # 停盘 D+1
    highs = np.full_like(opens, 100.0)
    labels = make_hit_labels(opens, highs)
    # t=2 的 buy at idx 3 = NaN → label=-1
    assert labels[2, 0] == -1


def test_label_stats_summary() -> None:
    labels = np.array([[1, 0, -1], [1, 1, 0]], dtype=np.int8)
    s = label_stats(labels)
    assert s["total"] == 6
    assert s["valid"] == 5
    assert s["hit"] == 3
    assert abs(s["hit_rate"] - 0.6) < 1e-9


# === FlatStockHead ===


def test_flat_stock_head_action_and_stock_softmax() -> None:
    cfg = ModelConfig(embed_dim=16, n_industries=4, phase=3)
    head = FlatStockHead(cfg)
    h = torch.randn(2, 5, 16)
    mask = torch.ones(2, 5, dtype=torch.bool)
    out = head(h, mask=mask)
    assert out.action_probs.shape == (2, 4)
    assert out.stock_probs.shape == (2, 5)
    assert out.p_hit.shape == (2, 5)
    torch.testing.assert_close(out.action_probs.sum(dim=-1), torch.ones(2), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out.stock_probs.sum(dim=-1), torch.ones(2), atol=1e-5, rtol=1e-5)
    assert (out.p_hit >= 0).all()
    assert (out.p_hit <= 1).all()


def test_flat_stock_head_mask_zeros_stock_and_phit() -> None:
    cfg = ModelConfig(embed_dim=16, n_industries=4, phase=3)
    head = FlatStockHead(cfg)
    h = torch.randn(1, 5, 16)
    mask = torch.tensor([[True, True, False, True, False]])
    out = head(h, mask=mask)
    assert out.stock_probs[0, 2].item() == 0.0
    assert out.stock_probs[0, 4].item() == 0.0
    assert out.p_hit[0, 2].item() < 1e-6
    assert out.p_hit[0, 4].item() < 1e-6


# === ActorCritic phase=3 ===


def test_actor_critic_phase3_returns_popactionout() -> None:
    cfg = ModelConfig(phase=3, embed_dim=32, n_heads=4, n_layers_temporal=2, n_layers_cross=1)
    model = ActorCritic(cfg)
    B, N, T, F = 2, 10, cfg.seq_len, cfg.n_features
    out = model(
        x=torch.randn(B, N, T, F),
        token_types=torch.ones(B, N, dtype=torch.long),
        industry_ids=torch.zeros(B, N, dtype=torch.long),
        mask=torch.ones(B, N, dtype=torch.bool),
    )
    assert isinstance(out, PopActionOut)
    assert out.action_probs.shape == (B, 4)
    assert out.stock_probs.shape == (B, N)
    assert out.p_hit.shape == (B, N)
    assert out.value.shape == (B,)
    torch.testing.assert_close(out.action_probs.sum(dim=-1), torch.ones(B), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out.stock_probs.sum(dim=-1), torch.ones(B), atol=1e-5, rtol=1e-5)
