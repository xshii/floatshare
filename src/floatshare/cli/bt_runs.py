"""`floatshare-bt-runs` — 列 / 查看 / 删 回测记录.

用法:
    floatshare-bt-runs list                    # 最近 30 个 backtest
    floatshare-bt-runs list --strategy ppo_signal
    floatshare-bt-runs show <run_id>           # 看单个 backtest 全指标 + 策略参数
    floatshare-bt-runs delete <run_id>

数据来源: data/ml/metrics.db 的 backtest_runs 表 (跟 training_runs 共库).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from floatshare.ml.backtest_tracking import (
    delete_backtest,
    get_backtest,
    list_backtests,
)


def _fmt(v: Any, ndigits: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:+.{ndigits}f}"
    return str(v)


def _cmd_list(args: argparse.Namespace) -> int:
    runs = list_backtests(strategy=args.strategy, limit=args.limit)
    if not runs:
        print("(no backtests)")
        return 0
    print(
        f"{'run_id':50} {'strategy':12} {'started':19} "
        f"{'return':>8} {'sharpe':>7} {'maxDD':>8}  note"
    )
    print("-" * 130)
    for r in runs:
        note = (r.get("note") or "").replace("\n", " ")[:40]
        print(
            f"{r['run_id']:50} "
            f"{r['strategy']:12} "
            f"{r['started_at']:19} "
            f"{_fmt(r['total_return']):>8} "
            f"{_fmt(r['sharpe']):>7} "
            f"{_fmt(r['max_drawdown']):>8}  "
            f"{note}"
        )
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    r = get_backtest(args.run_id)
    if r is None:
        print(f"不存在: {args.run_id}", file=sys.stderr)
        return 1
    print(f"run_id       : {r['run_id']}")
    print(f"strategy     : {r['strategy']}")
    print(f"started_at   : {r['started_at']}")
    print(f"finished_at  : {r['finished_at']}")
    print(f"status       : {r['status']}")
    print(f"window       : {r['window_start']} .. {r['window_end']}")
    print(f"codes_count  : {r['codes_count']}")
    print(f"capital      : {r['capital']:,.0f}" if r["capital"] else "capital      : -")
    print(f"linked_run   : {r['linked_run_id'] or '-'}")
    print(f"git_sha      : {r['git_sha'] or '-'}")
    print(f"note         : {r['note'] or '-'}")
    print()
    print("=== 核心指标 ===")
    for k in ("total_return", "cagr", "sharpe", "max_drawdown", "volatility", "win_rate"):
        print(f"  {k:<16} {_fmt(r[k])}")
    if r.get("metrics_json"):
        extra = json.loads(r["metrics_json"])
        others = {
            k: v
            for k, v in extra.items()
            if k not in ("total_return", "cagr", "sharpe", "max_drawdown", "volatility", "win_rate")
        }
        if others:
            print()
            print("=== 其它指标 ===")
            for k, v in others.items():
                print(f"  {k:<16} {_fmt(v)}")
    if r.get("strategy_params_json"):
        print()
        print("=== 策略参数 ===")
        for k, v in json.loads(r["strategy_params_json"]).items():
            print(f"  {k:<16} {v}")
    return 0


def _cmd_delete(args: argparse.Namespace) -> int:
    ok = delete_backtest(args.run_id)
    if not ok:
        print(f"不存在或删除失败: {args.run_id}", file=sys.stderr)
        return 1
    print(f"已删: {args.run_id}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="floatshare-bt-runs — 回测记录 CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="列出 backtest")
    p_list.add_argument("--strategy", default=None, help="按策略名过滤")
    p_list.add_argument("--limit", type=int, default=30)
    p_list.set_defaults(fn=_cmd_list)

    p_show = sub.add_parser("show", help="查单个 backtest 全部信息")
    p_show.add_argument("run_id")
    p_show.set_defaults(fn=_cmd_show)

    p_del = sub.add_parser("delete", help="删 backtest")
    p_del.add_argument("run_id")
    p_del.set_defaults(fn=_cmd_delete)

    args = p.parse_args()
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
