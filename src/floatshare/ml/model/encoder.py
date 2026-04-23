"""Encoder — 时序 (per-token) + 跨 token 两层。

TemporalEncoder    : (B, T, F) → (B, D)  对单 token 做时序编码
CrossTokenEncoder  : (B, N, D) → (B, N, D)  跨 token attention (含 type/industry emb)

两个 encoder 都用 RMSNorm pre-norm Transformer block。配置全来自 ModelConfig。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from floatshare.ml.model.attention import MultiHeadAttention, TransformerBlock
from floatshare.ml.model.norm import RMSNorm

if TYPE_CHECKING:
    from floatshare.ml.config import ModelConfig


class TemporalEncoder(nn.Module):
    """单 token 时序编码 — (B, T, F) 输入, (B, D) latent 输出 (mean pool)。"""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.feat_proj = nn.Linear(cfg.n_features, cfg.embed_dim)
        # 学习式位置编码 (短序列 T=60, 学比 sinusoidal 简单)
        self.pos_emb = nn.Parameter(torch.randn(cfg.seq_len, cfg.embed_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=cfg.embed_dim,
                    n_heads=cfg.n_heads,
                    ff_dim=cfg.ff_dim,
                    dropout=cfg.dropout,
                    attn_dropout=cfg.attn_dropout,
                    head_dim=cfg.head_dim,
                    eps=cfg.rms_eps,
                )
                for _ in range(cfg.n_layers_temporal)
            ]
        )
        self.final_norm = RMSNorm(cfg.embed_dim, cfg.rms_eps)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, F) → (B, D).

        Pool 用 last token (因果时序 — 最后一步包含所有历史信息), 业界标准
        (PatchTST / Informer / GPT 都这样). 比 mean pool 保留更多近期信息。
        """
        h = self.feat_proj(x) + self.pos_emb[: x.size(1)]
        for blk in self.blocks:
            h = blk(h, mask=None)
        h = self.final_norm(h)
        return h[:, -1, :]  # last-step pool (causal)


class CrossTokenEncoder(nn.Module):
    """跨 token attention — N tokens 互相 attend。

    输入会加 token_type embedding (industry vs stock) 和 industry_id
    embedding (parent / self), 让 encoder 区分语义。
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.token_type_emb = nn.Embedding(2, cfg.embed_dim)
        self.industry_emb = nn.Embedding(cfg.n_industries, cfg.embed_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=cfg.embed_dim,
                    n_heads=cfg.n_heads,
                    ff_dim=cfg.ff_dim,
                    dropout=cfg.dropout,
                    attn_dropout=cfg.attn_dropout,
                    head_dim=cfg.head_dim,
                    eps=cfg.rms_eps,
                )
                for _ in range(cfg.n_layers_cross)
            ]
        )
        self.final_norm = RMSNorm(cfg.embed_dim, cfg.rms_eps)

    def forward(
        self,
        x: Tensor,
        token_types: Tensor,
        industry_ids: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """x: (B, N, D), token_types: (B, N), industry_ids: (B, N), mask: (B, N).

        Returns: (B, N, D)
        """
        h = x + self.token_type_emb(token_types) + self.industry_emb(industry_ids)
        for blk in self.blocks:
            h = blk(h, mask=mask)
        return self.final_norm(h)


# ============================================================================
# Dual-Axis Encoder (iTransformer / Crossformer 风格)
# ============================================================================


class DualAxisBlock(nn.Module):
    """Dual-axis block: TimeAttn (per-stock) + StockAttn (sparse-T) + FFN.

    输入/输出 (B, N, T, D). 全程保留时序+截面信息.

    Stock-attn 在 T 维稀疏 (cfg.stock_attn_every / last_dense) — 缓解 O(T·N²) 瓶颈,
    实测能省 3-5× 时间, 性能影响小 (最近时间步全做, 远端稀疏).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm_t = RMSNorm(cfg.embed_dim, cfg.rms_eps)
        self.time_attn = MultiHeadAttention(
            cfg.embed_dim,
            cfg.n_heads,
            cfg.head_dim,
            cfg.attn_dropout,
        )
        self.norm_s = RMSNorm(cfg.embed_dim, cfg.rms_eps)
        self.stock_attn = MultiHeadAttention(
            cfg.embed_dim,
            cfg.n_heads,
            cfg.head_dim,
            cfg.attn_dropout,
        )
        self.norm_ff = RMSNorm(cfg.embed_dim, cfg.rms_eps)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.ff_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ff_dim, cfg.embed_dim),
            nn.Dropout(cfg.dropout),
        )
        self.stock_attn_every = cfg.stock_attn_every
        self.stock_attn_last_dense = cfg.stock_attn_last_dense

    def _active_t(self, t: int, device: torch.device) -> Tensor:
        """返回需要做 stock-attn 的时间步索引."""
        last_dense_start = max(0, t - self.stock_attn_last_dense)
        sparse = list(range(0, last_dense_start, max(1, self.stock_attn_every)))
        dense = list(range(last_dense_start, t))
        return torch.tensor(sorted(set(sparse + dense)), dtype=torch.long, device=device)

    def forward(self, h: Tensor, stock_mask: Tensor) -> Tensor:
        b, n, t, d = h.shape
        # 1. Time-attention per stock
        h_t = h.reshape(b * n, t, d)
        h = h + self.time_attn(self.norm_t(h_t)).reshape(b, n, t, d)

        # 2. Sparse stock-attention (only on active_t)
        active = self._active_t(t, h.device)
        tp = active.shape[0]
        # 提取活跃时间步切片: (B, N, T', D)
        h_subset = h.index_select(dim=2, index=active)
        # 排成 (B*T', N, D) 跑跨 stock attention
        h_s = h_subset.permute(0, 2, 1, 3).reshape(b * tp, n, d)
        m_bt = stock_mask.unsqueeze(1).expand(b, tp, n).reshape(b * tp, n)
        h_s = self.stock_attn(self.norm_s(h_s), mask=m_bt)
        # 用零张量做 scatter, 保留梯度
        h_s_full = torch.zeros_like(h)
        h_s_scattered = h_s.reshape(b, tp, n, d).permute(0, 2, 1, 3)  # (B,N,T',D)
        h_s_full.index_copy_(2, active, h_s_scattered)
        h = h + h_s_full

        # 3. FFN
        return h + self.ffn(self.norm_ff(h))


class DualAxisEncoder(nn.Module):
    """双轴 Transformer encoder (iTransformer / Crossformer 风格).

    全程保留 (B, N, T, D), 每层交替做 time-attn 和 stock-attn。
    最后 take last timestep (因果) → (B, N, D) 喂 head。
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.feat_proj = nn.Linear(cfg.n_features, cfg.embed_dim)
        self.time_pos_emb = nn.Parameter(
            torch.randn(cfg.seq_len, cfg.embed_dim) * 0.02,
        )
        self.industry_emb = nn.Embedding(cfg.n_industries, cfg.embed_dim)
        self.blocks = nn.ModuleList([DualAxisBlock(cfg) for _ in range(cfg.n_dual_layers)])
        self.final_norm = RMSNorm(cfg.embed_dim, cfg.rms_eps)

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        industry_ids: Tensor,
    ) -> Tensor:
        """x: (B, N, T, F), mask: (B, N), industry_ids: (B, N).

        Returns: (B, N, D) — last timestep 的 latent, 喂 head.
        """
        # Feature proj + position + industry
        h = self.feat_proj(x)  # (B, N, T, D)
        h = h + self.time_pos_emb[: x.size(2)]  # broadcast T
        h = h + self.industry_emb(industry_ids).unsqueeze(2)  # broadcast T

        for blk in self.blocks:
            h = blk(h, mask)  # (B, N, T, D)
        h = self.final_norm(h)
        return h[:, :, -1, :]  # last-step pool
