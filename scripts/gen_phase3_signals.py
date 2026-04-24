"""用 phase3 ckpt 推 p_hit 生成 signal CSV — **rolling universe + batch inference**.

关键设计:
    - **Rolling universe**: 每月 1 号重选 top-15 per industry 的 universe, 避免
      "用回测末尾数据选股"带来的 look-ahead + survivorship bias
    - **Batch inference**: 一次喂 B 天 window 给模型, 摊销 MPS kernel overhead
      (N=461 stocks 的 forward, B=1 太亏, 典型 B=32)
    - **Per-month cube**: 每月 universe 不同, 各建一份 cube (cube_builder 有缓存)

用法:
    python scripts/gen_phase3_signals.py \\
        --ckpt data/ml/ckpts/phase3_pretrain_best.pt \\
        --start 2025-04-01 --end 2026-03-31 \\
        --top-k 10 \\
        --out data/ml/signals/phase3_daily.csv

CSV 格式 (strategies/ppo_signal.py 消费): trade_date, token_id, weight
"""

from __future__ import annotations

import argparse
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

_INFER_BATCH_DAYS = 32  # 一次 forward 多少天 — MPS 吃 batch, 32 + seq=60 + N=461 约 2GB


def _month_starts(start: str, end: str) -> list[pd.Timestamp]:
    """生成 [start, end] 区间内每月 1 号的 Timestamp 列表."""
    idx = pd.date_range(start=start, end=end, freq="MS")  # Month Start
    s_ts = pd.Timestamp(start)
    # pd.date_range freq='MS' 从 start 之后第一个月 1 号开始, 需手动补 start 月
    if len(idx) == 0 or idx[0] > s_ts:
        first = pd.Timestamp(f"{s_ts.year}-{s_ts.month:02d}-01")
        idx = pd.DatetimeIndex([first, *idx])
    return list(idx)


def _run_batch_inference(
    model,
    features: torch.Tensor,
    batch_t: list[int],
    seq_len: int,
    n_tokens: int,
    device: torch.device,
) -> np.ndarray:
    """forward B 天 window, 返回 (B, N) p_hit numpy."""
    windows = torch.stack([features[t - seq_len + 1 : t + 1] for t in batch_t])  # (B, T, N, F)
    x_t = windows.permute(0, 2, 1, 3).contiguous()  # (B, N, T, F)
    mask = torch.ones(len(batch_t), n_tokens, dtype=torch.bool, device=device)
    tt = torch.ones(len(batch_t), n_tokens, dtype=torch.long, device=device)
    ind = torch.zeros(len(batch_t), n_tokens, dtype=torch.long, device=device)
    out = cast(PopActionOut, model(x_t, tt, ind, mask))
    return out.p_hit.float().cpu().numpy()  # (B, N)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ckpt", required=True, help="phase3 ckpt 路径 (.pt, 同名 .json 必须存在)")
    p.add_argument("--start", required=True, help="YYYY-MM-DD (signal 起点)")
    p.add_argument("--end", required=True, help="YYYY-MM-DD (signal 终点)")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--out", default="data/ml/signals/phase3_daily.csv")
    p.add_argument("--device", default="mps", choices=("mps", "cuda", "cpu"))
    args = p.parse_args()

    device = torch.device(args.device)
    model = load_ckpt(args.ckpt).to(device)
    model.eval()
    seq_len = model.cfg.seq_len
    print(f"[gen] ckpt loaded, seq_len={seq_len}, n_features={model.cfg.n_features}")

    month_starts = _month_starts(args.start, args.end)
    end_ts = pd.Timestamp(args.end)
    start_ts = pd.Timestamp(args.start)
    month_bounds = [*month_starts, end_ts + pd.Timedelta(days=1)]
    print(f"[gen] rolling universe: {len(month_starts)} 个月, 每月头 1 号重选")

    # 预算每月 universe (一次性批量算 — 每次 ~100ms)
    db_path = DataConfig().db_path
    monthly_universes: list[tuple[pd.Timestamp, list[str]]] = []
    for m_start in month_starts:
        # universe 用 m_start 当天的 snapshot — 严格只用历史信息
        universe = select_per_industry_top_k(
            db_path=db_path,
            as_of_date=m_start.strftime("%Y-%m-%d"),
        )
        monthly_universes.append((m_start, universe))
        print(f"  {m_start.date()} universe: {len(universe)} 股")

    rows: list[dict] = []
    total_days = 0
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for i, (m_start, universe) in enumerate(monthly_universes):
            m_end = month_bounds[i + 1]  # 下个月头 (exclusive)
            # 本月实际要推的区间: [max(m_start, start_ts), min(m_end, end_ts+1))
            sig_start = max(m_start, start_ts)
            sig_end = min(m_end, end_ts + pd.Timedelta(days=1))
            if sig_start >= sig_end:
                continue

            # Cube 范围: 向前多留 seq_len + 30 天做 window 历史
            cube_start = (sig_start - pd.Timedelta(days=seq_len + 30)).strftime("%Y-%m-%d")
            cube_end = (sig_end - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            cube = build_cube(DataConfig(), cube_start, cube_end, phase=3, universe=universe)

            features_gpu = torch.from_numpy(cube.features).to(device)
            token_ids = [t.token_id for t in cube.tokens]
            n_tokens = cube.n_tokens

            # 过滤到本月要出 signal 的交易日
            valid = [
                t
                for t in range(seq_len - 1, cube.n_days)
                if sig_start <= pd.Timestamp(cube.dates[t]) < sig_end
            ]
            if not valid:
                continue

            # Batch forward
            for bi in range(0, len(valid), _INFER_BATCH_DAYS):
                batch_t = valid[bi : bi + _INFER_BATCH_DAYS]
                p_hit_batch = _run_batch_inference(
                    model, features_gpu, batch_t, seq_len, n_tokens, device
                )
                for k, t in enumerate(batch_t):
                    p_hit = p_hit_batch[k]
                    top_idx = np.argsort(-p_hit)[: args.top_k]
                    trade_date = pd.Timestamp(cube.dates[t]).strftime("%Y-%m-%d")
                    rows.extend(
                        {
                            "trade_date": trade_date,
                            "token_id": token_ids[idx_],
                            "weight": float(p_hit[idx_]),
                        }
                        for idx_ in top_idx
                    )
            total_days += len(valid)
            print(f"  {m_start.date()} 完成: {len(valid)} 天推理")

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[gen] signals: {len(df)} rows ({total_days} days × {args.top_k}) → {out_path}")
    print(f"[gen] 独立 codes: {df['token_id'].nunique()} (--codes 用)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
