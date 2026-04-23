"""Env-based step 推理 — for Phase 1/2 PPO 评估 + ckpt 生成 signal.

提供 run_deterministic_rollout callback 式 API, 每步暴露 (t, state, weights, reward).
调用方自己决定消费 weights 还是 reward.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch

from floatshare.ml.rl.rollout import state_to_tensors
from floatshare.ml.types import HierarchicalActionOut, IndustryActionOut

if TYPE_CHECKING:
    from floatshare.ml.config import ModelConfig
    from floatshare.ml.model.agent import ActorCritic
    from floatshare.ml.rl.env import MarketEnv
    from floatshare.ml.types import ActionOut, EnvState


StepCallback = Callable[[int, "EnvState", np.ndarray, float], None]


def decode_eval_weights(
    out: ActionOut,
    model_cfg: ModelConfig,
    n_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    """从 actor 输出取 deterministic weights.

    Returns:
        weights_full: (n_tokens,) — phase 1 行业权重; phase 2 [industry | stock] 完整
        weights_for_step: 送给 env.step 的 weights (phase 2 里 industry 位置置 0,
                          因为 env 里 selection alpha 只用 stock 部分, industry timing
                          部分由 env 内部用 industry_ids 聚合 stock_w 得到).
    """
    if isinstance(out, IndustryActionOut):
        w = out.weights[0].cpu().numpy().astype(np.float32)
        return w, w
    if not isinstance(out, HierarchicalActionOut):
        raise TypeError(f"env-based 评估不支持 {type(out).__name__} (Pop 走监督路径)")
    n_ind = model_cfg.n_industries
    w = np.zeros(n_tokens, dtype=np.float32)
    w[:n_ind] = out.industry_weights[0].cpu().numpy()
    w[n_ind:] = out.stock_weights[0].cpu().numpy()
    w_for_step = w.copy()
    if model_cfg.phase == 2:
        w_for_step[:n_ind] = 0
    return w, w_for_step


def run_deterministic_rollout(
    env: MarketEnv,
    model: ActorCritic,
    device: torch.device,
    model_cfg: ModelConfig,
    on_step: StepCallback,
    start_idx: int | None = None,
) -> None:
    """从 env.reset() 跑到 done, 每步调 on_step(t, state, weights_full, reward).

    on_step 在 env.step 之后调用, 所以:
        t = env.t AFTER step (下一步起点)
        weights_full = 该步送给 decode_eval_weights 的完整权重
        reward = env.step 返回的 reward

    注意: 如果 on_step 需要 "step 之前的 t", 可以在 callback 外部维护.
    """
    model.eval()
    try:
        state = env.reset(start_idx=start_idx if start_idx is not None else model_cfg.seq_len)
        while True:
            x, tt, ind, mask = state_to_tensors(state, device)
            with torch.no_grad():
                out = model(x, tt, ind, mask)
            weights_full, weights_for_step = decode_eval_weights(
                out,
                model_cfg,
                env.n_tokens,
            )
            t_before = env.t
            reward, next_state, done = env.step(weights_for_step)
            on_step(t_before, state, weights_full, reward)
            if done:
                break
            assert next_state is not None, "env 未 done 却返回 None state"
            state = next_state
    finally:
        model.train()
