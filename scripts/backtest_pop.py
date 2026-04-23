#!/usr/bin/env python
"""回测 Phase 3 抓涨停 — best ckpt 在指定时段模拟打板。

规则:
    D 日收盘后, model 算 P(hit)
    → 排除停盘/一字板后取 top-K
    → D+1 open 买, D+2 open 卖
    → top-K 按 (3:2:1) 分仓 (或可配)
    → 扣成本 (默认单边手续费 0.2% = 印花税 0.1% + 佣金 0.03% + 滑点 0.05%)

输出: 累计收益 / Sharpe / 最大回撤 / 胜率 / 日均命中数
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from floatshare.ml.config import DataConfig
from floatshare.ml.data.dataset import build_cube
from floatshare.ml.data.universe import select_universe
from floatshare.ml.model.agent import load_ckpt


def backtest(
    ckpt_path: str,
    start: str,
    end: str,
    top_k: int = 3,
    sub_weights: tuple[float, ...] = (3, 2, 1),
    cost: float = 0.002,
    device: str = "mps",
    universe_as_of: str = "2023-12-31",
) -> dict:
    model = load_ckpt(ckpt_path).to(device)
    model.eval()
    mcfg = model.cfg

    dcfg = DataConfig(
        train_start=start,
        train_end=end,
        val_start=start,
        val_end=end,
        test_start=start,
        test_end=end,
        universe_mode="top_mv",
        top_mv_n=300,
    )
    # 用 train_end 时点的 universe (跟训练时一致, 无后验)
    universe = select_universe(dcfg, as_of_date=universe_as_of)
    print(f"universe @ {universe_as_of}: {len(universe)} 股")
    print(f"加载 cube {start} ~ {end} …")
    cube = build_cube(dcfg, start, end, phase=3, universe=universe)
    print(f"  {cube.n_days} days × {cube.n_tokens} stocks")

    sub_w = np.asarray(sub_weights, dtype=np.float64)
    sub_w = sub_w / sub_w.sum()
    if len(sub_w) != top_k:
        raise ValueError(f"sub_weights 长度 {len(sub_w)} != top_k {top_k}")

    seq_len = mcfg.seq_len
    n_days, n_tok = cube.n_days, cube.n_tokens

    # === 全时段批量推理 P(hit) ===
    print("推理 P(hit) 全时段 …")
    p_hits = np.zeros((n_days, n_tok), dtype=np.float32)
    valid_ts = list(range(seq_len - 1, n_days))
    batch = 32
    with torch.no_grad():
        for i in range(0, len(valid_ts), batch):
            idx = valid_ts[i : i + batch]
            x = np.stack([cube.features[t - seq_len + 1 : t + 1] for t in idx])
            x = np.transpose(x, (0, 2, 1, 3))
            xt = torch.from_numpy(x).to(device)
            mt = torch.ones(xt.shape[:2], dtype=torch.bool, device=device)
            tt = torch.ones(xt.shape[:2], dtype=torch.long, device=device)
            ind = torch.zeros(xt.shape[:2], dtype=torch.long, device=device)
            out = model(xt, tt, ind, mt)
            p = out.p_hit.cpu().numpy()
            for bi, t in enumerate(idx):
                p_hits[t] = p[bi]

    # === 回测循环 ===
    print(f"模拟交易 top-{top_k} 持 1 天 (3:2:1 分仓, cost={cost}) …")
    daily_rets: list[float] = []
    daily_hit5: list[int] = []
    all_rets: list[float] = []  # 单股 return 分布

    for t in range(seq_len - 1, n_days - 2):
        # 可交易: D+1 / D+2 open 都有, D+1 非一字板
        has_open = ~np.isnan(cube.opens[t + 1]) & ~np.isnan(cube.opens[t + 2])
        not_yi_zi = cube.opens[t + 1] != cube.highs[t + 1]
        tradable = has_open & not_yi_zi
        if tradable.sum() < top_k:
            continue

        p_adj = np.where(tradable, p_hits[t], -np.inf)
        top_idx = np.argpartition(-p_adj, top_k - 1)[:top_k]
        # 按 p_hit 降序排 — 决定分仓顺序
        top_idx = np.array(sorted(top_idx, key=lambda i: -p_adj[i]))

        buy_p = cube.opens[t + 1, top_idx]
        sell_p = cube.opens[t + 2, top_idx]
        rets = sell_p / buy_p - 1
        all_rets.extend(rets.tolist())

        port_ret = float((sub_w * rets).sum()) - cost
        daily_rets.append(port_ret)
        daily_hit5.append(int((rets >= 0.05).sum()))

    rets_arr = np.array(daily_rets)
    cum = np.cumprod(1 + rets_arr) - 1
    peak = np.maximum.accumulate(cum)
    max_dd = float((cum - peak).min())

    return {
        "n_days": len(rets_arr),
        "cum_return": float(cum[-1]) if len(cum) else 0.0,
        "annualized_return": float((1 + rets_arr.mean()) ** 252 - 1) if len(rets_arr) else 0.0,
        "sharpe": (
            float(rets_arr.mean() / rets_arr.std() * np.sqrt(252)) if rets_arr.std() > 0 else 0.0
        ),
        "max_drawdown": max_dd,
        "win_rate": float((rets_arr > 0).mean()) if len(rets_arr) else 0.0,
        "daily_ret_mean": float(rets_arr.mean()) if len(rets_arr) else 0.0,
        "daily_ret_std": float(rets_arr.std()) if len(rets_arr) else 0.0,
        "avg_hit5_per_day": float(np.mean(daily_hit5)) if daily_hit5 else 0.0,
        "stock_ret_mean": float(np.mean(all_rets)) if all_rets else 0.0,
        "stock_ret_hit5_rate": float(np.mean(np.array(all_rets) >= 0.05)) if all_rets else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="data/ml/ckpts/phase3_pretrain_best.pt")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--cost", type=float, default=0.002, help="单次买卖往返成本")
    p.add_argument("--device", default="mps")
    args = p.parse_args()

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    sub_w = {1: (1,), 2: (3, 2), 3: (3, 2, 1)}.get(args.top_k, tuple([1] * args.top_k))  # K>3 等权
    r = backtest(
        args.ckpt,
        args.start,
        args.end,
        top_k=args.top_k,
        sub_weights=sub_w,
        cost=args.cost,
        device=args.device,
    )

    print("\n" + "=" * 60)
    print(f"回测 {args.start} ~ {args.end}  top-{args.top_k}, cost={args.cost}")
    print("=" * 60)
    print(f"  交易日数:          {r['n_days']}")
    print(f"  累计收益:          {r['cum_return'] * 100:+.2f}%")
    print(f"  年化收益 (复利):   {r['annualized_return'] * 100:+.2f}%")
    print(f"  年化 Sharpe:       {r['sharpe']:.2f}")
    print(f"  最大回撤:          {r['max_drawdown'] * 100:+.2f}%")
    print(f"  日胜率:            {r['win_rate'] * 100:.1f}%")
    print(f"  日均收益:          {r['daily_ret_mean'] * 100:+.3f}%")
    print(f"  日波动率:          {r['daily_ret_std'] * 100:.3f}%")
    print(f"  日均涨≥5% 数 (top-{args.top_k}): {r['avg_hit5_per_day']:.2f} / {args.top_k}")
    print(f"  单股命中率 (≥5%):  {r['stock_ret_hit5_rate'] * 100:.1f}%")
    print(f"  单股平均 ret:      {r['stock_ret_mean'] * 100:+.3f}%")


if __name__ == "__main__":
    main()
