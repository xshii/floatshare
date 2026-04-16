#!/usr/bin/env python
"""命令行回测工具 — 基于 backtrader。"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest import run_backtest  # noqa: E402
from src.data.loader import DataLoader  # noqa: E402
from src.monitor import logger  # noqa: E402
from src.strategy.registry import StrategyRegistry  # noqa: E402

import strategies  # noqa: E402,F401  保证策略被注册


def _parse_params(items: list[str] | None) -> dict:
    out: dict = {}
    for kv in items or []:
        k, v = kv.split("=", 1)
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="FloatShare 回测工具")
    parser.add_argument("--strategy", required=True, help="策略名称")
    parser.add_argument("--codes", nargs="+", required=True, help="股票代码列表")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-12-01")
    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--source", default="akshare")
    parser.add_argument("--params", nargs="*", help="key=value 形式的策略参数")
    args = parser.parse_args()

    strategy_cls = StrategyRegistry.get(args.strategy)
    if strategy_cls is None:
        logger.error(f"策略未注册: {args.strategy}; 可用: {StrategyRegistry.list_strategies()}")
        return

    loader = DataLoader(source=args.source)
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    frames = []
    for code in args.codes:
        logger.info(f"加载 {code}")
        df = loader.get_daily(code, start, end)
        if not df.empty:
            frames.append(df)

    if not frames:
        logger.error("没有获取到数据")
        return

    data = pd.concat(frames, ignore_index=True)
    logger.info(f"共 {len(data)} 条数据")

    result = run_backtest(
        strategy_cls=strategy_cls,
        data=data,
        initial_capital=args.capital,
        strategy_params=_parse_params(args.params),
        start_date=start,
        end_date=end,
    )
    result.print_summary()


if __name__ == "__main__":
    main()
