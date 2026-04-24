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
    p.add_argument(
        "--batch-days",
        type=int,
        default=48,
        help="per-step 天数. M4 统一内存支持 48+, 摊销 MPS kernel overhead",
    )
    p.add_argument("--start", default=_DEFAULT_CFG.train_start)
    p.add_argument("--end", default=_DEFAULT_CFG.train_end)
    p.add_argument("--val-start", default=_DEFAULT_CFG.val_start)
    p.add_argument("--val-end", default=_DEFAULT_CFG.val_end)
    p.add_argument("--device", default="mps", choices=("mps", "cuda", "cpu"))
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="种子 (model init + np shuffle + MPS dropout 全部 seed)",
    )
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
    p.add_argument(
        "--ckpt-out",
        default=None,
        help=(
            "best ckpt 存储文件名 (只文件名, 不含目录). 默认 'phase3_pretrain_best.pt'. "
            "典型用法: 冷启 --ckpt-out pop_anchor.pt; 每日 warm --ckpt-out pop_warm_latest.pt"
        ),
    )
    p.add_argument(
        "--note",
        default=None,
        help="本次 run 的备注 (存进 metrics.db, 后续 floatshare-ml-runs list 看)",
    )
    p.add_argument(
        "--profile",
        type=int,
        default=0,
        metavar="N",
        help="profile 前 N 个 batch 后退出 (分段打点, 不 save ckpt, 不进 metrics.db). 0=禁用",
    )
    p.add_argument(
        "--profile-no-modules",
        action="store_true",
        help="profile 模式下不装模块级 forward hook — hook 的 sync 会扰乱 MPS 调度, "
        "去掉可得 forward/backward 的真实总耗时 (失去细粒度 breakdown)",
    )
    p.add_argument(
        "--eval-every",
        type=int,
        default=2,
        help="每 N 个 epoch 跑 val eval + 存 ckpt (1 = 每 epoch)",
    )
    p.add_argument(
        "--push-top-k",
        type=int,
        default=10,
        help="训完推 Bark 的今日选股数量 (默认 10)",
    )
    p.add_argument(
        "--no-push-picks",
        action="store_true",
        help="关闭训完后的 Bark 今日选股推送 (默认开启)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "smoke mode: train 9 月 (2024-01~09) + 10 epoch + eval_every=1, 约 30min/round, "
            "快速超参迭代. 会 override --start/--epochs/--eval-every. Note 加 [smoke] 前缀 "
            "(注意: batch 数少, 看相对指标对比即可, 不追求收敛)"
        ),
    )
    args = p.parse_args()

    if args.smoke:
        args.start = "2024-01-01"
        args.epochs = 10
        args.eval_every = 1
        args.note = f"[smoke] {args.note}" if args.note else "[smoke]"

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
    train_cfg = TrainConfig(device=args.device, epochs=args.epochs, lr=args.lr, seed=args.seed)
    trainer = PopTrainer(
        model_cfg,
        data_cfg,
        train_cfg,
        epochs=args.epochs,
        batch_days=args.batch_days,
        resume_from_ckpt=args.resume_from,
        note=args.note,
        eval_every=args.eval_every,
        ckpt_name_override=args.ckpt_out,
    )

    if args.profile > 0:
        from floatshare.ml.profiling import run_profile

        run_profile(
            trainer,
            n_batches=args.profile,
            module_level=not args.profile_no_modules,
        )
        return

    trainer.fit()

    if not args.no_push_picks:
        from pathlib import Path

        from floatshare.ml.recommend import push_today_picks

        ckpt_path = Path(trainer.ckpt_dir) / (args.ckpt_out or trainer._ckpt_name)
        try:
            push_today_picks(ckpt_path, trainer.device, top_k=args.push_top_k)
        except Exception as e:  # 推送失败不影响训练结果
            from floatshare.observability import logger

            logger.warning(f"[push_picks] 失败: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
