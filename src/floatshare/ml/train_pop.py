"""Phase 3 抓涨停 Stage A 监督预训 CLI — 薄壳.

实际训练逻辑在 floatshare.ml.training.pop.PopTrainer.
"""

from __future__ import annotations

import argparse
import os

from floatshare.ml.config import DataConfig, ModelConfig, TrainConfig
from floatshare.ml.training import PopTrainer

_DEFAULT_CFG = DataConfig()


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 3 抓涨停 Stage A 监督预训")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-days", type=int, default=16)
    p.add_argument("--start", default=_DEFAULT_CFG.train_start)
    p.add_argument("--end", default=_DEFAULT_CFG.train_end)
    p.add_argument("--val-start", default=_DEFAULT_CFG.val_start)
    p.add_argument("--val-end", default=_DEFAULT_CFG.val_end)
    p.add_argument("--device", default="mps", choices=("mps", "cuda", "cpu"))
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--top-mv-n",
        type=int,
        default=300,
        help="universe_mode=top_mv 时 top N (per-industry 模式会覆盖)",
    )
    p.add_argument("--seq-len", type=int, default=60)
    p.add_argument("--encoder-mode", default="dual_axis", choices=("single_axis", "dual_axis"))
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-dual-layers", type=int, default=3, help="dual_axis block 层数")
    p.add_argument("--ff-dim", type=int, default=None, help="FFN 中间维度, None → 4×embed_dim")
    p.add_argument(
        "--resume-from",
        default=None,
        help="warm-start: 从该 ckpt 路径加载 backbone (shape 不匹配的层自动跳过)",
    )
    args = p.parse_args()

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    model_cfg = ModelConfig(
        phase=3,
        seq_len=args.seq_len,
        encoder_mode=args.encoder_mode,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim if args.ff_dim else args.embed_dim * 4,
        n_dual_layers=args.n_dual_layers,
    )
    data_cfg = DataConfig(
        train_start=args.start,
        train_end=args.end,
        val_start=args.val_start,
        val_end=args.val_end,
        universe_mode="top_mv",
        top_mv_n=args.top_mv_n,
    )
    train_cfg = TrainConfig(device=args.device, epochs=args.epochs, lr=args.lr)
    trainer = PopTrainer(
        model_cfg,
        data_cfg,
        train_cfg,
        epochs=args.epochs,
        batch_days=args.batch_days,
        resume_from_ckpt=args.resume_from,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
