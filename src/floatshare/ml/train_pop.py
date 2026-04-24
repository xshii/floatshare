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
        try:
            _push_today_picks(trainer, top_k=args.push_top_k)
        except Exception as e:  # 推送失败不影响训练结果
            from floatshare.observability import logger

            logger.warning(f"[push_picks] 失败: {type(e).__name__}: {e}")


def _push_today_picks(trainer, top_k: int = 10) -> None:
    """训完后用 best ckpt 跑今日 top-K 选股, Bark 推送.

    从 `data/ml/ckpts/{_ckpt_name}` 重载 (训练期间 best 已存入), 不直接用
    `trainer.model` 因为后者是 raw (非 EMA), 且可能已经过了 best epoch.
    """
    from datetime import date
    from pathlib import Path
    from typing import cast

    import numpy as np
    import pandas as pd
    import torch

    from floatshare.ml.config import DataConfig
    from floatshare.ml.data.dataset import build_cube
    from floatshare.ml.data.universe import select_per_industry_top_k
    from floatshare.ml.model.agent import load_ckpt
    from floatshare.ml.types import PopActionOut
    from floatshare.observability import logger, notify

    today = date.today().strftime("%Y-%m-%d")
    ckpt_path = Path(trainer.ckpt_dir) / trainer._ckpt_name
    if not ckpt_path.exists():
        logger.warning(f"[push_picks] ckpt 不存在: {ckpt_path}, 跳过")
        return

    device = trainer.device
    model = load_ckpt(str(ckpt_path)).to(device)
    model.eval()
    seq_len = model.cfg.seq_len

    universe = select_per_industry_top_k("data/floatshare.db", today)
    if not universe:
        logger.warning("[push_picks] universe 空 (可能 today snapshot 未就绪), 跳过")
        return

    # Lookback: seq_len 是**交易日**, 转 calendar 要 × 1.5 (含周末) + buffer
    lookback_cal = int(seq_len * 1.5) + 60
    cube_start = (pd.Timestamp(today) - pd.Timedelta(days=lookback_cal)).strftime("%Y-%m-%d")
    cube = build_cube(DataConfig(), cube_start, today, phase=3, universe=universe)
    if cube.n_days < seq_len:
        logger.warning(f"[push_picks] cube 不够长 ({cube.n_days} < {seq_len}), 跳过")
        return

    feats = torch.from_numpy(cube.features).to(device)
    t_last = cube.n_days - 1
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        window = (
            feats[t_last - seq_len + 1 : t_last + 1].unsqueeze(0).permute(0, 2, 1, 3).contiguous()
        )
        n = window.shape[1]
        out = cast(
            PopActionOut,
            model(
                window,
                torch.ones(1, n, dtype=torch.long, device=device),
                torch.zeros(1, n, dtype=torch.long, device=device),
                torch.ones(1, n, dtype=torch.bool, device=device),
            ),
        )
        p_hit = out.p_hit.float().cpu().numpy()[0]

    top_idx = np.argsort(-p_hit)[:top_k]
    token_ids = [tk.token_id for tk in cube.tokens]
    picks = [(token_ids[i], float(p_hit[i])) for i in top_idx]
    trade_date = pd.Timestamp(cube.dates[t_last]).date()

    # 锁屏通知预览只会展头部 ~3-4 行 (~180 字), 前面塞一大段"人设化"闲聊 + 颜文字
    # 掩饰; picks 拼在后面, 点开 Bark app 展开才能看全量.
    # 6 位 code 去掉交易所后缀, 进一步降低一眼认出是股票的概率.
    import hashlib

    # md5 纯装饰用 (不做安全用途), usedforsecurity=False 显式告知 linter
    tag = hashlib.md5(str(trade_date).encode(), usedforsecurity=False).hexdigest()[:6]
    camo = (
        f"〜(꒪꒳꒪)〜 今日小日记 ~ 猫猫今天吃了三条鱼 /ᐠ｡ꞈ｡ᐟ\\ "
        f"窗外花开了呢 ✿ 温度 21°C 心情不错 (๑•̀ㅂ•́)و✧ "
        f"session_id={tag} 加油鸭~ 🦆 · "
    )
    picks_str = " ".join(f"{c[:6]}({p:.2f})" for c, p in picks)
    body = camo + picks_str
    notify(title="日常小记 (*´꒳`*)", body=body)
    logger.info(f"[push_picks] {trade_date} pushed (camo): {picks_str}")


if __name__ == "__main__":
    main()
