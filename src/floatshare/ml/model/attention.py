"""可配置 Multi-Head Attention + Transformer Block (RMSNorm pre-norm)。

Pre-norm (LLaMA 风格) 比 post-norm 训练稳定, 不需要 warmup。
mask: (B, N) bool, True = 保留, False = 屏蔽 (设为 -inf logit)。

Attention 计算走 `F.scaled_dot_product_attention` — MPS 上有 fused Metal kernel,
避免物化 (B, H, N, N) 中间 scores 矩阵 (对大 N 如 stock-attn N=457 极重要).
"""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F

from floatshare.ml.model.norm import RMSNorm


class MultiHeadAttention(nn.Module):
    """标准 MHA, 支持 key padding mask. 用 SDPA fused kernel.

    Args:
        embed_dim: token 隐藏维度 D
        n_heads: 头数, head_dim 默认 = D // n_heads
        head_dim: 显式指定每头维度 (覆盖 D // n_heads)
        dropout: attention 权重 dropout
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        head_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = head_dim or (embed_dim // n_heads)
        self.inner_dim = self.head_dim * n_heads
        self.dropout_p = dropout

        self.q_proj = nn.Linear(embed_dim, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.inner_dim, bias=False)
        self.out_proj = nn.Linear(self.inner_dim, embed_dim, bias=False)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """x: (B, N, D), mask: (B, N) bool (True=keep). 返回 (B, N, D).

        SDPA 的 bool attn_mask 语义: True=可见, False=屏蔽 — 与我们的 mask 一致,
        直接广播 (B, 1, 1, N) 即可, SDPA 内部 fused scaling + softmax + dropout.
        """
        b, n, _ = x.shape
        q = self.q_proj(x).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        # (B, H, N, head_dim)

        attn_mask = mask[:, None, None, :] if mask is not None else None  # (B,1,1,N) broadcast
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )  # (B, H, N, head_dim)
        out = out.transpose(1, 2).contiguous().view(b, n, self.inner_dim)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: x → x + Attn(RMSNorm(x)) → x + FFN(RMSNorm(x))。"""

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        head_dim: int | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(embed_dim, eps)
        self.attn = MultiHeadAttention(embed_dim, n_heads, head_dim, attn_dropout)
        self.norm2 = RMSNorm(embed_dim, eps)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        h = x + self.attn(self.norm1(x), mask=mask)
        return h + self.ff(self.norm2(h))
