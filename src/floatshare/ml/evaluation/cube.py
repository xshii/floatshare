"""Cube-based batched 推理 — for Phase 3 监督评估 (AUC / P@K).

不走 MarketEnv.step. 直接拉 cube.features 的 (T-window) 切片批量推理, 收 p_hit.
用于 train_pop.py 每 N epoch 算 val 指标, 以及离线跑 AUC.
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
    features: np.ndarray,  # (n_days, n_tokens, F)
    labels: np.ndarray,  # (n_days, n_tokens) int8 (1/0/-1)
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
    features: np.ndarray,
    labels: np.ndarray,
    seq_len: int,
    device: torch.device,
    batch_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """批量跑 forward, 返回 (all_p, all_y) — (n_valid_days, n_tokens) 两个数组."""
    model.eval()
    n_days = features.shape[0]
    valid_starts = list(range(seq_len - 1, n_days))
    p_chunks, y_chunks = [], []
    try:
        with torch.no_grad():
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
    features: np.ndarray,
    labels: np.ndarray,
    batch_idx: list[int],
    seq_len: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """从 cube 切 B 天 window, forward → p_hit numpy."""
    x = np.stack([features[t - seq_len + 1 : t + 1] for t in batch_idx])
    x = np.transpose(x, (0, 2, 1, 3))  # (B, N, T, F)
    x_t = torch.from_numpy(x).to(device)
    mask_t = torch.ones(x_t.shape[:2], dtype=torch.bool, device=device)
    tt = torch.ones(x_t.shape[:2], dtype=torch.long, device=device)
    ind = torch.zeros(x_t.shape[:2], dtype=torch.long, device=device)
    out = cast(PopActionOut, model(x_t, tt, ind, mask_t))
    return out.p_hit.cpu().numpy(), labels[np.array(batch_idx)]
