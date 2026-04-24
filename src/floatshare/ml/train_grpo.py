"""GRPO 训练 CLI — Phase 1/2/3, critic-free + group-relative baseline.

实际训练逻辑在 floatshare.ml.training.grpo.GRPOTrainer.

Usage:
    floatshare-train-grpo --phase 1 --epochs 100
    floatshare-train-grpo --phase 2 --pretrained data/ml/ckpts/p1_grpo_best.pt
    floatshare-train-grpo --phase 3 --pretrained data/ml/ckpts/pop_anchor.pt \\
        --reward-horizon 1  # 抓涨停 1 天持仓
"""

from __future__ import annotations

import argparse
import os

from floatshare.ml.config import DataConfig, GRPOConfig, ModelConfig, TrainConfig
from floatshare.ml.training import GRPOTrainer

_DEFAULT_CFG = DataConfig()


def main() -> None:
    p = argparse.ArgumentParser(description="FloatShare GRPO 训练 (critic-free)")
    p.add_argument("--phase", type=int, choices=(1, 2, 3), default=1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--start", default=_DEFAULT_CFG.train_start)
    p.add_argument("--end", default=_DEFAULT_CFG.train_end)
    p.add_argument("--val-start", default=_DEFAULT_CFG.val_start)
    p.add_argument("--val-end", default=_DEFAULT_CFG.val_end)
    p.add_argument("--device", default="mps", choices=("mps", "cuda", "cpu"))
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--group-size", type=int, default=8, help="G: 每 state 采样 action 数")
    p.add_argument("--kl-coef", type=float, default=0.04, help="KL(policy||ref) 系数, 防漂")
    p.add_argument("--rollout-days", type=int, default=250)
    p.add_argument(
        "--reward-horizon",
        type=int,
        default=None,
        help="持仓 K 天 (默认: phase=3 用 1, phase 1/2 用 GRPOConfig 默认 5)",
    )
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers-temporal", type=int, default=3)
    p.add_argument("--n-layers-cross", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--pretrained", default=None, help="加载 backbone ckpt 路径 (Pop anchor / 前一个 GRPO)"
    )
    p.add_argument("--universe-mode", default="top_mv", choices=("top_mv", "hs300", "zz500"))
    p.add_argument("--top-mv-n", type=int, default=300)
    p.add_argument("--note", default=None, help="run 备注 (metrics.db)")
    args = p.parse_args()

    # Phase 3 默认 reward_horizon=1 (对齐 HitLabelConfig sell_offset=2 的 1 天持仓)
    if args.reward_horizon is None:
        args.reward_horizon = 1 if args.phase == 3 else 5

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    model_cfg = ModelConfig(
        phase=args.phase,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers_temporal=args.n_layers_temporal,
        n_layers_cross=args.n_layers_cross,
        pretrained_backbone=args.pretrained,
    )
    data_cfg = DataConfig(
        train_start=args.start,
        train_end=args.end,
        val_start=args.val_start,
        val_end=args.val_end,
        universe_mode=args.universe_mode,
        top_mv_n=args.top_mv_n,
    )
    grpo_cfg = GRPOConfig(
        group_size=args.group_size,
        kl_coef=args.kl_coef,
        rollout_days=args.rollout_days,
        reward_horizon=args.reward_horizon,
    )
    train_cfg = TrainConfig(device=args.device, epochs=args.epochs, lr=args.lr, seed=args.seed)
    trainer = GRPOTrainer(
        model_cfg, data_cfg, grpo_cfg, train_cfg, epochs=args.epochs, note=args.note
    )
    trainer.fit()


if __name__ == "__main__":
    main()
