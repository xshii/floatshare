"""`floatshare-ml-runs` — 列 / 查看 / 加备注 / 删 ML 训练记录.

用法:
    floatshare-ml-runs list                    # 最近 30 个 run
    floatshare-ml-runs list --trainer PopTrainer
    floatshare-ml-runs show <run_id>           # 看单个 run 的 epoch 指标曲线
    floatshare-ml-runs note <run_id> "备注内容"
    floatshare-ml-runs delete <run_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from floatshare.ml.tracking import (
    delete_run,
    get_metrics,
    get_run,
    list_runs,
    set_note,
)


def _fmt_metric(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _cmd_list(args: argparse.Namespace) -> int:
    runs = list_runs(trainer=args.trainer, limit=args.limit)
    if not runs:
        print("(no runs)")
        return 0
    print(f"{'run_id':40} {'trainer':14} {'started':19} {'status':8} {'best':7} {'@ep':4}  note")
    print("-" * 120)
    for r in runs:
        note = (r.get("note") or "").replace("\n", " ")[:40]
        print(
            f"{r['run_id']:40} "
            f"{r['trainer']:14} "
            f"{r['started_at']:19} "
            f"{r['status']:8} "
            f"{_fmt_metric(r['best_metric']):7} "
            f"{_fmt_metric(r['best_epoch']):>4}  "
            f"{note}"
        )
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    r = get_run(args.run_id)
    if r is None:
        print(f"run {args.run_id} not found", file=sys.stderr)
        return 1
    print(f"run_id      : {r['run_id']}")
    print(f"trainer     : {r['trainer']}")
    print(f"started_at  : {r['started_at']}")
    print(f"finished_at : {r['finished_at']}")
    print(f"status      : {r['status']}")
    print(f"n_params    : {r['n_params']}")
    print(
        f"metric      : {r['metric_key']} best={_fmt_metric(r['best_metric'])} @ E{r['best_epoch']}"
    )
    print(f"git_sha     : {r['git_sha']}")
    print(f"note        : {r['note']}")
    print()

    metrics = get_metrics(args.run_id)
    if not metrics:
        print("(no per-epoch metrics)")
        return 0
    print(
        f"{'epoch':5} {'train_loss':11} {'val_auc':8} {'val_p@10':9} "
        f"{'lr':10} {'train_s':8} {'eval_s':7}"
    )
    print("-" * 72)
    for m in metrics:
        tr = json.loads(m["train_metrics_json"]) if m["train_metrics_json"] else {}
        va = json.loads(m["val_metrics_json"]) if m["val_metrics_json"] else {}
        # val_auc / auc 二选一但不能用 `or` — auc=0.0 会被 falsy 误跳过
        auc = va.get("val_auc", va.get("auc"))
        train_s = m["train_time_s"] if m["train_time_s"] is not None else 0.0
        eval_s = m["eval_time_s"] if m["eval_time_s"] is not None else 0.0
        print(
            f"{m['epoch']:5} "
            f"{_fmt_metric(tr.get('train_loss')):11} "
            f"{_fmt_metric(auc):8} "
            f"{_fmt_metric(va.get('val_p@10')):9} "
            f"{m['lr']:10.2e} "
            f"{train_s:8.1f} "
            f"{eval_s:7.1f}"
        )
    return 0


def _cmd_note(args: argparse.Namespace) -> int:
    ok = set_note(args.run_id, args.text)
    if ok:
        print(f"✓ note set on {args.run_id}")
        return 0
    print(f"run {args.run_id} not found", file=sys.stderr)
    return 1


def _cmd_delete(args: argparse.Namespace) -> int:
    ok = delete_run(args.run_id)
    if ok:
        print(f"✓ deleted {args.run_id}")
        return 0
    print(f"run {args.run_id} not found", file=sys.stderr)
    return 1


def main() -> None:
    p = argparse.ArgumentParser(description="floatshare ML 训练记录管理")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="列出 run")
    p_list.add_argument("--trainer", default=None, help="过滤: PopTrainer / GRPOTrainer / ...")
    p_list.add_argument("--limit", type=int, default=30)
    p_list.set_defaults(func=_cmd_list)

    p_show = sub.add_parser("show", help="看单 run 的 epoch 曲线")
    p_show.add_argument("run_id")
    p_show.set_defaults(func=_cmd_show)

    p_note = sub.add_parser("note", help="给 run 加/改备注")
    p_note.add_argument("run_id")
    p_note.add_argument("text")
    p_note.set_defaults(func=_cmd_note)

    p_del = sub.add_parser("delete", help="删 run + 其 metrics")
    p_del.add_argument("run_id")
    p_del.set_defaults(func=_cmd_delete)

    args = p.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
