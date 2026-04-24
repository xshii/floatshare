"""命令行入口 — `floatshare-backtest` entry point。"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import date

import pandas as pd

from floatshare.application import create_default_loader, run_backtest
from floatshare.ml.backtest_tracking import BacktestTracker
from floatshare.observability import logger
from floatshare.registry import discover, get, list_strategies


def _parse_params(items: list[str] | None) -> dict[str, object]:
    out: dict[str, object] = {}
    for kv in items or []:
        k, v = kv.split("=", 1)
        try:
            out[k] = int(v)
            continue
        except ValueError:
            pass
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="FloatShare 回测工具")
    parser.add_argument("--strategy", required=True, help="策略名称")
    parser.add_argument("--codes", nargs="+", required=True, help="股票代码列表")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-12-01")
    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--params", nargs="*", help="key=value 形式的策略参数")
    parser.add_argument(
        "--note",
        default=None,
        help="本次 backtest 的备注 (存进 metrics.db, floatshare-bt-runs list 可看)",
    )
    parser.add_argument(
        "--linked-run-id",
        default=None,
        help="关联的 training_run run_id (建立 ckpt → backtest 溯源)",
    )
    args = parser.parse_args()

    discover()  # 触发 strategies/ 下所有策略的 @register
    strategy_cls = get(args.strategy)
    if strategy_cls is None:
        logger.error(f"策略未注册: {args.strategy}; 可用: {list_strategies()}")
        return

    loader = create_default_loader()
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

    strategy_params = _parse_params(args.params)
    tracker = BacktestTracker()
    with tracker.run(
        strategy=args.strategy,
        codes_count=len(args.codes),
        window_start=args.start,
        window_end=args.end,
        capital=args.capital,
        strategy_params=strategy_params,
        note=args.note,
        linked_run_id=args.linked_run_id,
    ) as handle:
        result = run_backtest(
            strategy_cls=strategy_cls,
            data=data,
            initial_capital=args.capital,
            strategy_params=strategy_params,
            start_date=start,
            end_date=end,
        )
        result.print_summary()
        # Promote 6 个核心指标到列 + 全量 JSON, floatshare-bt-runs 直接筛
        m = dataclasses.asdict(result.metrics)
        handle.log_result(
            total_return=float(m.get("total_return") or 0.0),
            cagr=float(m.get("cagr") or 0.0),
            sharpe=float(m.get("sharpe") or 0.0),
            max_drawdown=float(m.get("max_drawdown") or 0.0),
            volatility=float(m.get("volatility") or 0.0),
            win_rate=float(m.get("win_rate") or 0.0),
            all_metrics=m,
        )
        logger.info(f"[bt] recorded run_id: {handle.run_id}")


if __name__ == "__main__":
    main()
