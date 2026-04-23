"""通用策略调参 — 任何 registered strategy 都可用。

用法:
    python notebooks/tune_strategy.py --strategy sector_relay
    python notebooks/tune_strategy.py --strategy dual_thrust --code 600519.SH
    python notebooks/tune_strategy.py --strategy ma_cross --code 000300.SH \\
        --train-years 3 --test-years 1 --n-trials 30

策略必须实现 `search_space(trial) -> dict[str, Any]` classmethod (在 strategies/* 里)。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

# 让 strategies/ 可 import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sqlalchemy import text

from floatshare.application.optimization import (
    make_walk_forward_splits,
    walk_forward_optimize,
)
from floatshare.infrastructure.storage.database import DatabaseStorage
from floatshare.observability import logger
from floatshare.registry import discover, get

OUT = Path("notebooks/output")
OUT.mkdir(parents=True, exist_ok=True)


def load_data(table: str, codes: list[str], start: date, end: date) -> pd.DataFrame:
    """从 DB 拼 OHLCV panel — 任何 daily 类表 + code 列表都行。"""
    db = DatabaseStorage()
    code_list = "', '".join(codes)
    df = pd.read_sql(
        text(f"""
        SELECT code, trade_date, open, high, low, close, volume,
               COALESCE(amount, 0) AS amount
        FROM {table}
        WHERE code IN ('{code_list}')
          AND date(trade_date) >= :start
          AND date(trade_date) <= :end
        ORDER BY code, trade_date
    """),
        db.engine,
        params={"start": start.isoformat(), "end": end.isoformat()},
    )
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="WFO 通用调参")
    p.add_argument("--strategy", required=True, help="策略 registry name")
    p.add_argument("--table", default="raw_daily", help="数据表 (raw_daily / index_daily)")
    p.add_argument("--code", nargs="+", help="code 列表 (留空 = sector_relay 默认 31 SW)")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--train-years", type=int, default=3)
    p.add_argument("--test-years", type=int, default=1)
    p.add_argument("--n-trials", type=int, default=30, help="每 split 的 optuna 试验次数")
    p.add_argument("--cap", type=float, default=100_000, help="初始资金")
    args = p.parse_args()

    discover("strategies")
    cls = get(args.strategy)
    if cls is None:
        raise SystemExit(f"✗ 找不到策略 {args.strategy}")
    if not hasattr(cls, "search_space"):
        raise SystemExit(
            f"✗ {cls.__name__} 缺 @classmethod search_space (无法调参)。"
            f"参考 strategies/sector_relay.py"
        )

    # sector_relay 默认拉 SW L1 31 行业 (跟 strategy 内部 SW_L1 dict 对齐)
    if not args.code:
        if args.strategy == "sector_relay":
            from strategies.sector_relay import SW_L1

            args.code = list(SW_L1.keys())
            args.table = "index_daily"
        else:
            raise SystemExit("✗ 请用 --code 指定标的")

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    print(
        f"=== 加载 {args.table} {args.code[:3]}{'...' if len(args.code) > 3 else ''} "
        f"{start} ~ {end} ==="
    )
    data = load_data(args.table, args.code, start, end)
    if data.empty:
        raise SystemExit("✗ 数据为空")

    # sector_relay 走 ETF 价格归一化
    if args.strategy == "sector_relay" and args.table == "index_daily":
        for col in ("open", "high", "low", "close"):
            data[col] = data[col] / 1000.0
        print("  (sector_relay: 价格归一化 / 1000 → ETF 量级)")

    splits = make_walk_forward_splits(
        start,
        end,
        train_years=args.train_years,
        test_years=args.test_years,
    )
    print(f"\n=== {len(splits)} 个 walk-forward 切分 ===")
    for sp in splits:
        print(
            f"  train [{sp.train_start} ~ {sp.train_end}] → test [{sp.test_start} ~ {sp.test_end}]"
        )

    print(
        f"\n=== 调参中 ({args.n_trials} trials × {len(splits)} splits, "
        f"约 {args.n_trials * len(splits) * 5 // 60} 分钟) ==="
    )
    results = walk_forward_optimize(
        cls,
        data,
        splits,
        n_trials=args.n_trials,
        initial_capital=args.cap,
    )

    print("\n=== OOS 汇总 ===")
    print(f"{'Test 期':<24} {'Sharpe':>8} {'Return':>10} {'MaxDD':>10} {'Trades':>8}")
    for r in results:
        print(
            f"{str(r.split.test_start) + '~' + str(r.split.test_end):<24} "
            f"{r.oos_sharpe:>8.2f} "
            f"{r.oos_total_return:>+9.2%} "
            f"{r.oos_max_drawdown:>+9.2%} "
            f"{r.oos_n_trades:>8d}"
        )

    if results:
        avg_sharpe = sum(r.oos_sharpe for r in results) / len(results)
        avg_ret = sum(r.oos_total_return for r in results) / len(results)
        std_sharpe = (sum((r.oos_sharpe - avg_sharpe) ** 2 for r in results) / len(results)) ** 0.5
        print(f"\n  mean OOS Sharpe = {avg_sharpe:.2f}  (std {std_sharpe:.2f})")
        print(f"  mean OOS return = {avg_ret:+.2%}")
        print(f"  → 稳定性: {'稳定' if std_sharpe < 0.5 else 'noisy, params 不稳'}")

    # 落盘
    out_path = OUT / f"wfo_{args.strategy}.json"
    out = {
        "strategy": args.strategy,
        "data_table": args.table,
        "n_codes": len(args.code),
        "window": f"{start} ~ {end}",
        "train_years": args.train_years,
        "test_years": args.test_years,
        "n_trials_per_split": args.n_trials,
        "results": [
            {
                "train": f"{r.split.train_start} ~ {r.split.train_end}",
                "test": f"{r.split.test_start} ~ {r.split.test_end}",
                "best_params": r.best_params,
                "oos_sharpe": r.oos_sharpe,
                "oos_total_return": r.oos_total_return,
                "oos_max_drawdown": r.oos_max_drawdown,
                "oos_n_trades": r.oos_n_trades,
            }
            for r in results
        ],
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    logger.info(f"✓ 写入 {out_path}")


if __name__ == "__main__":
    main()
