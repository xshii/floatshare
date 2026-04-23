"""RMSNorm — LayerNorm 简化版, 仅 RMS 归一化, 无 mean 中心化, 无 bias。

参数比 LN 少一半 (没 bias), 速度快 ~10%, 质量基本一致 (Llama/Qwen 都用)。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * scale。

    注意: 沿最后一个维度归一化 (typical for transformer hidden states)。
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.scale
