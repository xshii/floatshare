"""Cube-based batched 推理 — for Phase 3 监督评估 (AUC / P@K).

不走 MarketEnv.step. 直接拉 cube.features 的 (T-window) 切片批量推理, 收 p_hit.
用于 train_pop.py 每 N epoch 算 val 指标, 以及离线跑 AUC.

features 以 torch.Tensor (GPU-resident) 传入 — caller 负责一次性 `.to(device)`,
避免推理时 per-batch host→device 复制.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from floatshare.ml.evaluation.metrics import rank_based_auc, top_k_precision
from floatshare.ml.types import PopActionOut

if TYPE_CHECKING:
    from floatshare.ml.model.agent import ActorCritic


def classifier_metrics(
    model: ActorCritic,
    features: torch.Tensor,  # (n_days, n_tokens, F) device-resident
    labels: np.ndarray,  # (n_days, n_tokens) int8 (1/0/-1) on CPU (sklearn AUC 用)
    seq_len: int,
    device: torch.device,
    top_k: int = 10,
    batch_days: int = 32,
) -> dict[str, float]:
    """全 cube 批量推理 + 算 AUC / P@K / 阳性率 (监督 phase=3 标准评估)."""
    all_p, all_y = _batched_p_hit_inference(
        model,
        features,
        labels,
        seq_len,
        device,
        batch_days,
    )
    flat_valid = all_y >= 0
    p_flat = all_p[flat_valid]
    y_flat = all_y[flat_valid]
    return {
        "n_valid": int(flat_valid.sum()),
        "base_rate": float(y_flat.mean()) if len(y_flat) else 0.0,
        "auc": rank_based_auc(y_flat, p_flat),
        f"p@{top_k}": top_k_precision(all_p, all_y, top_k),
    }


def _batched_p_hit_inference(
    model: ActorCritic,
    features: torch.Tensor,
    labels: np.ndarray,
    seq_len: int,
    device: torch.device,
    batch_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """批量跑 forward, 返回 (all_p, all_y) — (n_valid_days, n_tokens) 两个数组."""
    model.eval()
    n_days = features.shape[0]
    valid_starts = list(range(seq_len - 1, n_days))
    if not valid_starts:
        # n_days < seq_len — 没法构造单个评估窗口 (通常是 val 太短); 返回空, caller 处理
        return np.array([], dtype=np.float32), np.array([], dtype=np.int8)
    p_chunks, y_chunks = [], []
    try:
        # bf16 autocast — eval 精度足够 (ranking 指标对 1e-3 误差不敏感), 速度 1.5-2×
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            for i in range(0, len(valid_starts), batch_days):
                batch_idx = valid_starts[i : i + batch_days]
                p, y = _infer_one_batch(model, features, labels, batch_idx, seq_len, device)
                p_chunks.append(p)
                y_chunks.append(y)
    finally:
        model.train()
    return np.concatenate(p_chunks), np.concatenate(y_chunks)


def _infer_one_batch(
    model: ActorCritic,
    features: torch.Tensor,
    labels: np.ndarray,
    batch_idx: list[int],
    seq_len: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """从 cube 切 B 天 window, forward → p_hit numpy. features 已在 device 上."""
    # (B, T, N, F) → (B, N, T, F), slice 都在 device 上做
    windows = torch.stack([features[t - seq_len + 1 : t + 1] for t in batch_idx])
    x_t = windows.permute(0, 2, 1, 3).contiguous()
    mask_t = torch.ones(x_t.shape[:2], dtype=torch.bool, device=device)
    tt = torch.ones(x_t.shape[:2], dtype=torch.long, device=device)
    ind = torch.zeros(x_t.shape[:2], dtype=torch.long, device=device)
    out = cast(PopActionOut, model(x_t, tt, ind, mask_t))
    # bf16 下 .numpy() 不支持, 显式 .float() 回 fp32
    return out.p_hit.float().cpu().numpy(), labels[np.array(batch_idx)]
