"""PPO 训练入口 CLI — 薄壳.

实际训练逻辑在 floatshare.ml.training.ppo.PPOTrainer.

Usage:
    floatshare-train-ppo --phase 1 --epochs 100
    floatshare-train-ppo --phase 1 --epochs 100 --start 2018-01-01 --end 2024-12-31
    floatshare-train-ppo --phase 2 --pretrained data/ml/ckpts/p1_best.pt
"""

from __future__ import annotations

import argparse
import os

from floatshare.ml.config import DataConfig, ModelConfig, PPOConfig, TrainConfig
from floatshare.ml.training import PPOTrainer

_DEFAULT_CFG = DataConfig()


def main() -> None:
    p = argparse.ArgumentParser(description="FloatShare PPO 训练")
    p.add_argument("--phase", type=int, choices=(1, 2), default=1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--start", default=_DEFAULT_CFG.train_start)
    p.add_argument("--end", default=_DEFAULT_CFG.train_end)
    p.add_argument("--val-start", default=_DEFAULT_CFG.val_start)
    p.add_argument("--val-end", default=_DEFAULT_CFG.val_end)
    p.add_argument("--device", default="mps", choices=("mps", "cuda", "cpu"))
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--rollout-days", type=int, default=250)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers-temporal", type=int, default=3)
    p.add_argument("--n-layers-cross", type=int, default=2)
    p.add_argument("--pretrained", default=None, help="Phase 2 用: 加载 Phase 1 backbone ckpt 路径")
    p.add_argument("--universe-mode", default="top_mv", choices=("top_mv", "hs300", "zz500"))
    p.add_argument("--top-mv-n", type=int, default=300)
    p.add_argument(
        "--market-baseline",
        default="equal_sw",
        choices=("equal_sw", "hs300"),
        help="reward benchmark: equal_sw (31 行业等权) | hs300",
    )
    args = p.parse_args()

    # MPS Dirichlet 采样不支持 → 自动 fallback CPU
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
    ppo_cfg = PPOConfig(rollout_days=args.rollout_days)
    train_cfg = TrainConfig(
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        market_baseline=args.market_baseline,
    )
    trainer = PPOTrainer(
        model_cfg,
        data_cfg,
        ppo_cfg,
        train_cfg,
        epochs=args.epochs,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
