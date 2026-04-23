"""pipeline 时间片探测 — 真跑一次 S1/S2a/S2b/S3, 落盘每 stage 实际耗时.

目的:
    - 验证 S1 (sync) / S2b (tushare 对拍) / S3 (feature audit canary proxy)
      在真数据下的实际耗时, 校准 launchd plist 时间表
    - 记录落 logs/pipeline-timing-YYYY-MM-DD.json, 给后续 pipeline_run 表做基准

用法:
    python scripts/pipeline_timing_probe.py [--trade-date YYYY-MM-DD]

输出:
    console: 每 stage 的 status + 耗时
    logs/pipeline-timing-YYYY-MM-DD.json: 结构化记录
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sqlite3
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import floatshare  # noqa: F401  触发 _load_dotenv
from floatshare.domain.enums import PipelineStage, StageStatus
from floatshare.observability import logger, notify

DB_PATH = "data/floatshare.db"


@dataclass(frozen=True, slots=True)
class StageResult:
    """单个 pipeline stage 的执行记录. 属性访问 + IDE 补全, 替代裸 dict."""

    stage: str  # PipelineStage 的 .value
    started_at: str
    duration_s: float
    status: str  # StageStatus 的 .value
    error: str | None
    extras: dict[str, Any] = field(default_factory=dict)  # stage-specific 计数 / 指标

    def to_json(self) -> dict[str, Any]:
        """JSON 序列化: extras 展平到顶层 (保留 pipeline-timing-*.json 既有 shape)."""
        d = dataclasses.asdict(self)
        extras = d.pop("extras")
        return {**d, **extras}


def _stage(
    records: list[StageResult],
    stage: PipelineStage,
    fn: Callable[[], dict[str, Any] | None],
    *,
    push: bool = True,
) -> StageResult:
    from floatshare.observability.metrics import Metric, record_counter, scope

    t0 = time.monotonic()
    started = datetime.now().isoformat(timespec="seconds")
    status, error, extras = StageStatus.OK, None, {}
    try:
        extras = fn() or {}
    except Exception as exc:
        status = StageStatus.FAIL
        error = f"{type(exc).__name__}: {exc}"
    dur = time.monotonic() - t0
    rec = StageResult(
        stage=stage.value,
        started_at=started,
        duration_s=round(dur, 2),
        status=status.value,
        error=error,
        extras=extras,
    )
    records.append(rec)

    # 持久化: duration_s + ok bool + 每个数值 extra 各一条 counter
    stage_scope = scope(Metric.Domain.PIPELINE, stage.value)
    record_counter(stage_scope, Metric.Counter.DURATION_S, dur, status=status.value)
    record_counter(
        stage_scope,
        Metric.Counter.OK,
        1.0 if status == StageStatus.OK else 0.0,
        error=error,
    )
    for k, v in extras.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            record_counter(stage_scope, k, float(v))

    msg = f"[{stage.value}] {status.value} in {dur:.1f}s"
    if error:
        msg += f" — {error}"
    if extras:
        msg += f"  {extras}"
    logger.info(msg)
    if push:
        icon = "✓" if status == StageStatus.OK else "⚠️"
        title = f"{icon} [{stage.value}] {dur:.0f}s"
        body_bits = [f"{k}={v}" for k, v in extras.items()]
        if error:
            body_bits.insert(0, error)
        notify(title, " | ".join(body_bits) if body_bits else status.value)
    return rec


def main() -> None:
    from floatshare.application.bootstrap import cli_metrics_run

    p = argparse.ArgumentParser(description="pipeline timing probe")
    p.add_argument("--trade-date", default=date.today().isoformat(), help="YYYY-MM-DD (T)")
    p.add_argument("--spot-sample", type=int, default=50, help="tushare 对拍抽样股数")
    p.add_argument(
        "--force-sync",
        action="store_true",
        help="S1 加 --force 重拉已有数据 (历史日测耗时用, 不落盘 watermark)",
    )
    args = p.parse_args()
    with cli_metrics_run():
        _run_probe(args)


def _run_probe(args) -> None:
    from floatshare.observability.metrics import Metric, record_kpi

    trade_date = args.trade_date
    log_path = Path(f"logs/pipeline-timing-{trade_date}.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[StageResult] = []
    ctx: dict[str, Any] = {}  # 跨 stage 共享上下文 (feats/panel/codes)

    # === S1: sync T 日行情 4 表 (by-date mode) ===
    def s1() -> dict[str, Any]:
        cmd = [
            "floatshare-sync",
            "--by-date",
            "--start",
            trade_date,
            "--end",
            trade_date,
            "--include",
            "raw_daily",
            "daily_basic",
            "moneyflow",
            "adj_factor",
        ]
        if args.force_sync:
            cmd.append("--force")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, check=False)
        # 用 DB 直接抽检当天 raw_daily 行数
        with sqlite3.connect(DB_PATH) as conn:
            n = conn.execute(
                f"SELECT COUNT(*) FROM raw_daily WHERE trade_date LIKE '{trade_date}%'"
            ).fetchone()[0]
        return {"returncode": r.returncode, "raw_daily_rows_today": n}

    _stage(records, PipelineStage.S1_SYNC, s1)

    # === S2a: DB 完整性 (3 个 check, 复用 db_integrity.py) ===
    def s2a() -> dict[str, Any]:
        from floatshare.application.db_integrity import (
            check_cross_table_alignment,
            check_daily_row_count_stability,
            check_trade_date_duplicates,
        )

        dups = check_trade_date_duplicates(DB_PATH)
        cross = check_cross_table_alignment(DB_PATH)
        jumps = check_daily_row_count_stability(DB_PATH)
        return {
            "duplicates": len(dups),
            "cross_missing_pairs": sum(m.missing_pairs for m in cross),
            "row_count_jumps": len(jumps),
        }

    _stage(records, PipelineStage.S2A_DB_INTEGRITY, s2a)

    # === S_prep: 算今日 features (S2b/S3 共同前置) ===
    def prep() -> dict[str, Any]:
        # universe: 今日有行情 + 排除 BJ + 排除 ST (name LIKE 'ST%' / '*ST%')
        from floatshare.domain.enums import ExchangeSuffix
        from floatshare.ml.data.loader import load_panel
        from floatshare.ml.features import compute_features

        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                f"""
                SELECT DISTINCT r.code
                FROM raw_daily r
                LEFT JOIN stock_lifecycle l ON l.code = r.code
                WHERE r.trade_date LIKE '{trade_date}%'
                  AND r.code NOT LIKE '%{ExchangeSuffix.BJ}'
                  AND (l.name IS NULL OR (l.name NOT LIKE 'ST%' AND l.name NOT LIKE '*ST%'))
                """
            ).fetchall()
        codes = [r[0] for r in rows]
        # 窗口: 过去 ~150 个日历日 (确保 120 交易日, MACD 26 EMA + 安全余量)
        t = date.fromisoformat(trade_date)
        start = (t.replace(year=t.year - 1) if t.month > 2 else date(t.year - 1, 1, 1)).isoformat()
        panel = load_panel(DB_PATH, codes, start, trade_date)
        feats = compute_features(panel)
        ctx["panel"] = panel
        ctx["feats"] = feats
        ctx["codes"] = codes
        return {
            "n_codes": len(codes),
            "panel_rows": len(panel),
            "feats_rows": len(feats),
        }

    _stage(records, PipelineStage.S2B_PREP_FEATURES, prep)

    # === S2b: tushare stk_factor 对拍 ===
    def s2b() -> dict[str, Any]:
        from floatshare.infrastructure.data_sources.tushare import TushareSource
        from floatshare.ml.audit_tushare import run_tushare_spot_check

        ts = TushareSource()
        mm = run_tushare_spot_check(ctx["feats"], trade_date, ts, sample_codes=args.spot_sample)
        return {
            "sample_size": args.spot_sample,
            "n_mismatches": len(mm),
            "features": sorted({m.feature for m in mm}) if mm else [],
        }

    _stage(records, PipelineStage.S3A_TUSHARE_CHECK, s2b)

    # === S3: feature audit (rolling 252 漂移 = canary proxy) ===
    def s3() -> dict[str, Any]:
        from floatshare.ml.audit import run_feature_audit

        report, _ = run_feature_audit(ctx["feats"], trade_date, panel=None, raise_on_error=False)
        return {
            "n_alerts": len(report.alerts),
            "n_features_checked": report.n_features_checked,
            "has_errors": report.has_errors(),
        }

    _stage(records, PipelineStage.S3B_FEATURE_AUDIT, s3)

    log_path.write_text(json.dumps([r.to_json() for r in records], indent=2, ensure_ascii=False))
    total = sum(r.duration_s for r in records)
    n_ok = sum(1 for r in records if r.status == StageStatus.OK.value)
    logger.info(f"✓ 完成 {n_ok}/{len(records)} stages, 总耗时 {total:.1f}s → {log_path}")

    # 顶层 KPI: 跨日趋势用 — 总耗时 + 成功率 (今天这次 = n_ok / n_stages)
    record_kpi(
        Metric.Domain.PIPELINE,
        trade_date,
        Metric.Kpi.TOTAL_DURATION_S,
        total,
        n_stages=len(records),
        n_ok=n_ok,
    )
    success_rate = n_ok / len(records) if records else 0.0
    record_kpi(
        Metric.Domain.PIPELINE,
        trade_date,
        Metric.Kpi.SUCCESS_RATE_7D,  # 当前只写单次; 7d 滚动由 web 层从 counter_event 聚合
        success_rate,
    )


if __name__ == "__main__":
    main()
