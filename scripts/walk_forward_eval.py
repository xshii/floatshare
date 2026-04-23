"""Walk-Forward Validation (WFV) for POP/PPO trainers.

业界 OOS 评估标准: 每月滚动重训 + 评估在下 1 个月, 聚合所有 OOS metric 看分布.

两种模式:
  [默认] warm-start 链: 每步从上步 ckpt 续训 3 epochs, 产线等价
  [--cold] cold-retrain: 每步从零训 N epochs, 研究/论文标准

用法:
  # warm-start WFV, 评估 2024-01 ~ 2026-03 每月
  python scripts/walk_forward_eval.py --start 2024-01 --end 2026-03

  # cold retrain WFV, 30 epochs per step (耗时 ~30x)
  python scripts/walk_forward_eval.py --start 2024-01 --end 2026-03 --cold --epochs 30

  # 3 月滚动窗 (3 month horizon)
  python scripts/walk_forward_eval.py --start 2024-01 --end 2026-03 --window-months 3

输出:
  data/ml/wfv_results/<run_id>/
      metrics.csv    每步一行: train_end, val_start, val_end, val_auc, val_p@10, ckpt
      summary.json   聚合指标: mean_auc, std_auc, mean_p@10, n_steps

设计:
    anchor 固定 DataConfig.train_start (2018-01-01); train_end 滚动推进.
    每步 val_start..val_end = 当月, train_end = val_start - 1 day.
    默认 warm-start: --resume-from = 上一步 best ckpt (链式).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import uuid
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from floatshare.ml.config import DataConfig


@dataclass
class StepResult:
    step_idx: int
    train_end: str
    val_start: str
    val_end: str
    val_auc: float | None
    val_p_at_10: float | None
    best_metric: float | None
    ckpt_path: str | None
    returncode: int


def _add_months(d: date, months: int) -> date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    return date(y, m, 1)


def _last_day_of_month(d: date) -> date:
    next_month = _add_months(d, 1)
    return next_month - timedelta(days=1)


def _iter_months(start_ym: str, end_ym: str, window_months: int) -> list[tuple[date, date]]:
    """生成 (val_start, val_end) 月份对列表."""
    ys, ms = map(int, start_ym.split("-"))
    ye, me = map(int, end_ym.split("-"))
    start = date(ys, ms, 1)
    end = date(ye, me, 1)
    out: list[tuple[date, date]] = []
    cur = start
    while cur <= end:
        val_start = cur
        val_end = _last_day_of_month(_add_months(cur, window_months - 1))
        out.append((val_start, val_end))
        cur = _add_months(cur, window_months)
    return out


def _parse_final_metric(stdout: str) -> tuple[float | None, float | None, float | None]:
    """从 trainer 最后 log 里抓 val_auc / val_p@10 / best metric.

    trainer 末尾日志格式 (base.py:150):
        '完成 ✓ best val auc=0.543'
    和每个 epoch 结束 eval 行:
        'val_auc=0.541 val_p@10=0.312 base=0.050 auc=0.541'
    """
    best = None
    m = re.search(r"best val \w+=([0-9.]+)", stdout)
    if m:
        best = float(m.group(1))

    val_auc = None
    val_p10 = None
    # 取最后一行 val_auc=...
    for line in reversed(stdout.splitlines()):
        ma = re.search(r"val_auc=([0-9.]+)", line)
        if ma and val_auc is None:
            val_auc = float(ma.group(1))
        mp = re.search(r"val_p@10=([0-9.]+)", line)
        if mp and val_p10 is None:
            val_p10 = float(mp.group(1))
        if val_auc is not None and val_p10 is not None:
            break
    return val_auc, val_p10, best


def _find_produced_ckpt(ckpt_dir: Path, phase_suffix: str = "phase3") -> Path | None:
    """返回 ckpt_dir 下最新修改的 phase{N}_best.pt. 找不到返回 None."""
    candidates = list(ckpt_dir.glob(f"{phase_suffix}*.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _run_one_step(
    anchor: str,
    train_end: date,
    val_start: date,
    val_end: date,
    epochs: int,
    resume_from: str | None,
    ckpt_dir: Path,
) -> StepResult:
    cmd = [
        "floatshare-train-pop",
        "--epochs",
        str(epochs),
        "--start",
        anchor,
        "--end",
        train_end.isoformat(),
        "--val-start",
        val_start.isoformat(),
        "--val-end",
        val_end.isoformat(),
    ]
    if resume_from:
        cmd += ["--resume-from", resume_from]
    print(f"  → 执行: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600 * 6, check=False)
    val_auc, val_p10, best = _parse_final_metric(r.stdout + "\n" + r.stderr)
    produced = _find_produced_ckpt(ckpt_dir)
    return StepResult(
        step_idx=-1,  # 由 caller 填
        train_end=train_end.isoformat(),
        val_start=val_start.isoformat(),
        val_end=val_end.isoformat(),
        val_auc=val_auc,
        val_p_at_10=val_p10,
        best_metric=best,
        ckpt_path=str(produced) if produced else None,
        returncode=r.returncode,
    )


def _write_outputs(run_dir: Path, results: list[StepResult]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "step",
                "train_end",
                "val_start",
                "val_end",
                "val_auc",
                "val_p_at_10",
                "best_metric",
                "ckpt",
                "returncode",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.step_idx,
                    r.train_end,
                    r.val_start,
                    r.val_end,
                    r.val_auc if r.val_auc is not None else "",
                    r.val_p_at_10 if r.val_p_at_10 is not None else "",
                    r.best_metric if r.best_metric is not None else "",
                    r.ckpt_path or "",
                    r.returncode,
                ]
            )

    # summary
    import statistics

    aucs = [r.val_auc for r in results if r.val_auc is not None]
    p10s = [r.val_p_at_10 for r in results if r.val_p_at_10 is not None]
    summary: dict[str, object] = {
        "n_steps": len(results),
        "n_with_auc": len(aucs),
        "mean_auc": statistics.mean(aucs) if aucs else None,
        "std_auc": statistics.stdev(aucs) if len(aucs) > 1 else None,
        "mean_p_at_10": statistics.mean(p10s) if p10s else None,
        "std_p_at_10": statistics.stdev(p10s) if len(p10s) > 1 else None,
        "n_failed": sum(1 for r in results if r.returncode != 0),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n✓ WFV 输出 → {run_dir}")
    print(f"  metrics.csv  — {len(results)} 步")
    print(f"  summary.json — mean_auc={summary['mean_auc']}, std={summary['std_auc']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", required=True, help="WFV 起始月 YYYY-MM (val)")
    p.add_argument("--end", required=True, help="WFV 终止月 YYYY-MM (val)")
    p.add_argument("--window-months", type=int, default=1, help="每步 val 窗口月数")
    p.add_argument("--anchor", default=None, help="train_start anchor; 默认 DataConfig.train_start")
    p.add_argument(
        "--cold",
        action="store_true",
        help="每步冷启动 (不 resume), 与 anchored WFV from-scratch 一致",
    )
    p.add_argument(
        "--epochs", type=int, default=None, help="per-step epochs (warm 默认 3, cold 30)"
    )
    p.add_argument("--ckpt-dir", default="data/ml/ckpts", help="trainer ckpt 输出目录")
    p.add_argument("--out-dir", default=None, help="WFV 输出目录, 默认 data/ml/wfv_results/<uuid>")
    args = p.parse_args()

    anchor = args.anchor or DataConfig().train_start
    epochs = args.epochs or (30 if args.cold else 3)
    run_id = uuid.uuid4().hex[:8]
    run_dir = Path(args.out_dir) if args.out_dir else Path("data/ml/wfv_results") / run_id
    ckpt_dir = Path(args.ckpt_dir)

    steps = _iter_months(args.start, args.end, args.window_months)
    print(f"=== Walk-Forward {len(steps)} 步 ===")
    print(f"anchor      : {anchor}")
    print(f"mode        : {'cold retrain' if args.cold else 'warm-start chain'}")
    print(f"epochs/step : {epochs}")
    print(f"window      : {args.window_months} 月")
    print(f"run_dir     : {run_dir}\n")

    results: list[StepResult] = []
    prev_ckpt: str | None = None
    for i, (val_start, val_end) in enumerate(steps, start=1):
        train_end = val_start - timedelta(days=1)
        resume = None if args.cold else prev_ckpt
        print(f"[{i}/{len(steps)}] train=..{train_end} val={val_start}..{val_end}")
        r = _run_one_step(anchor, train_end, val_start, val_end, epochs, resume, ckpt_dir)
        r.step_idx = i
        results.append(r)
        if r.returncode != 0:
            print(f"  ⚠ step {i} failed rc={r.returncode}, 继续下一步")
        else:
            print(f"  ✓ val_auc={r.val_auc} p@10={r.val_p_at_10} ckpt={r.ckpt_path}")
        prev_ckpt = r.ckpt_path if not args.cold else None

    _write_outputs(run_dir, results)


if __name__ == "__main__":
    main()
