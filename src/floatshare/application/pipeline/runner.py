"""Pipeline 编排器 — preflight → gates → stages 逐个跑, 按 fail 策略中止或继续.

设计:
    - StagePolicy = (stage_enum, fail_policy, fn)
    - run_pipeline(trade_date, stages, ...) → PipelineSummary
    - 过 gate: skip=True → 幂等跳; ok=False → 按 fail_policy 决定 abort/continue
    - 每 stage 自动 time_scope + record_counter (duration_s, ok) + 数值 extras
    - 结束时 record_kpi 总耗时 + 成功率, 并 notify summary
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from floatshare.application.pipeline.gates import STAGE_GATES, GateResult
from floatshare.application.pipeline.preflight import (
    PreflightResult,
    has_fatal_failure,
    run_preflight,
)
from floatshare.domain.enums import FailPolicy, PipelineStage, StageStatus
from floatshare.domain.pipeline import StageContext
from floatshare.observability import logger, notify
from floatshare.observability.metrics import (
    Metric,
    record_counter,
    record_kpi,
    scope,
)

StageFn = Callable[[StageContext], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class StagePolicy:
    """stage 的编排策略: 用哪个 fn 跑, 失败怎么办."""

    stage: PipelineStage
    fail_policy: FailPolicy
    fn: StageFn


@dataclass(slots=True)
class StageOutcome:
    """单 stage 执行结果 — 供 summary + 测试断言用."""

    stage: PipelineStage
    status: StageStatus
    duration_s: float
    error: str | None = None
    gate_block_reason: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PipelineSummary:
    """pipeline 一次执行的完整结果."""

    run_id: str
    trade_date: str
    preflight: list[PreflightResult]
    stages: list[StageOutcome]
    total_duration_s: float
    success_rate: float
    aborted: bool


# 默认 stage 队列组装在 cli/run_pipeline.py — 因为引入 ml 层 stage fn,
# 而 application 不能依赖 ml (lint-imports 契约). CLI 层聚合两边没问题.


# === 内部执行 ================================================================


def _check_gates(
    stage: PipelineStage, trade_date: str, db_path: str | Path
) -> tuple[GateResult | None, bool]:
    """过该 stage 所有 gate. 返回 (first_block 或 None, 是否建议跳过).

    - 任一 gate ok=False → 返回 (gate, False): 前置未满足
    - 任一 gate skip=True → 返回 (gate, True): 幂等跳过
    - 全过 → 返回 (None, False)
    """
    for gate_fn in STAGE_GATES.get(stage, ()):
        r = gate_fn(trade_date, db_path)
        if not r.ok:
            return r, False
        if r.skip:
            return r, True
    return None, False


def _run_stage(policy: StagePolicy, ctx: StageContext) -> StageOutcome:
    """time + 调 fn + 捕异常 + emit counter. 不处理 fail 策略 (留给 run_pipeline)."""
    stage_scope = scope(Metric.Domain.PIPELINE, policy.stage.value)
    t0 = time.monotonic()
    status: StageStatus = StageStatus.OK
    error: str | None = None
    extras: dict[str, Any] = {}
    try:
        extras = policy.fn(ctx) or {}
    except Exception as exc:
        status = StageStatus.FAIL
        error = f"{type(exc).__name__}: {exc}"
        logger.exception(f"[{policy.stage.value}] 异常")
    dur = time.monotonic() - t0

    # Counter 埋点 — 每条带 trade_date tag, gate 以此识别 (不用墙钟 ts)
    record_counter(
        stage_scope,
        Metric.Counter.DURATION_S,
        dur,
        trade_date=ctx.trade_date,
        status=status.value,
    )
    record_counter(
        stage_scope,
        Metric.Counter.OK,
        1.0 if status == StageStatus.OK else 0.0,
        trade_date=ctx.trade_date,
        error=error,
    )
    for k, v in extras.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            record_counter(stage_scope, k, float(v), trade_date=ctx.trade_date)

    return StageOutcome(
        stage=policy.stage,
        status=status,
        duration_s=round(dur, 2),
        error=error,
        extras=extras,
    )


def _emit_summary_kpis(trade_date: str, outcomes: list[StageOutcome]) -> tuple[float, float]:
    """聚合 KPI: 总耗时 + 成功率. 返回 (total, success_rate)."""
    total = sum(o.duration_s for o in outcomes)
    n_ok = sum(1 for o in outcomes if o.status == StageStatus.OK)
    n_real = sum(1 for o in outcomes if o.status != StageStatus.SKIPPED)
    success_rate = n_ok / n_real if n_real else 0.0

    record_kpi(
        Metric.Domain.PIPELINE,
        trade_date,
        Metric.Kpi.TOTAL_DURATION_S,
        total,
        n_stages=len(outcomes),
        n_ok=n_ok,
        n_real=n_real,
    )
    record_kpi(Metric.Domain.PIPELINE, trade_date, Metric.Kpi.SUCCESS_RATE_7D, success_rate)
    return total, success_rate


def _notify_summary(
    trade_date: str, outcomes: list[StageOutcome], total: float, aborted: bool
) -> None:
    n_ok = sum(1 for o in outcomes if o.status == StageStatus.OK)
    icon = "✓" if (not aborted and n_ok == len(outcomes)) else "⚠️"
    title = f"{icon} Pipeline {trade_date} {n_ok}/{len(outcomes)} OK ({total:.0f}s)"
    body_lines = [
        f"{o.stage.value}: {o.status.value} {o.duration_s:.1f}s"
        + (f" — {o.error[:60]}" if o.error else "")
        + (f" — {o.gate_block_reason[:60]}" if o.gate_block_reason else "")
        for o in outcomes
    ]
    if aborted:
        body_lines.append("⛔ ABORTED by fail-fast")
    notify(title, "\n".join(body_lines))


# === 主入口 ==================================================================


def run_pipeline(
    trade_date: str,
    stages: tuple[StagePolicy, ...],
    *,
    db_path: str | Path = "data/floatshare.db",
    skip_preflight: bool = False,
) -> PipelineSummary:
    """跑一次 pipeline. Caller 负责 setup_metrics() / run_context (如需).

    stages 必传 — 由 CLI 层组装 (application 不可依赖 ml, 无法在此直接组).

    Args:
        trade_date: T 日期 YYYY-MM-DD
        stages: 要跑的 stage 策略序列
        db_path: 本地 DB 路径
        skip_preflight: 测试里可跳过 env 检查

    Returns:
        PipelineSummary 含 preflight + 每 stage outcome + 聚合指标
    """
    started_at = datetime.now().isoformat(timespec="seconds")
    logger.info(f"▶ pipeline 启动 trade_date={trade_date} 共 {len(stages)} stages")

    preflight: list[PreflightResult] = []
    if not skip_preflight:
        preflight = run_preflight()
        if has_fatal_failure(preflight):
            logger.error("preflight 致命失败, pipeline 中止")
            return PipelineSummary(
                run_id=_current_run_id_or_empty(),
                trade_date=trade_date,
                preflight=preflight,
                stages=[],
                total_duration_s=0.0,
                success_rate=0.0,
                aborted=True,
            )

    ctx = StageContext(trade_date=trade_date, db_path=db_path)
    outcomes: list[StageOutcome] = []
    aborted = False

    for policy in stages:
        # Gate 前置
        block, should_skip = _check_gates(policy.stage, trade_date, db_path)
        if should_skip:
            logger.info(f"⟳ [{policy.stage.value}] {block.message if block else 'skip'}")
            outcomes.append(
                StageOutcome(
                    stage=policy.stage,
                    status=StageStatus.SKIPPED,
                    duration_s=0.0,
                    gate_block_reason=block.message if block else None,
                )
            )
            continue
        if block is not None:  # ok=False 阻塞
            logger.warning(f"✗ [{policy.stage.value}] gate: {block.message}")
            outcomes.append(
                StageOutcome(
                    stage=policy.stage,
                    status=StageStatus.GATE_BLOCKED,
                    duration_s=0.0,
                    gate_block_reason=block.message,
                )
            )
            if policy.fail_policy == FailPolicy.FAST:
                aborted = True
                break
            continue

        # 真正执行
        outcome = _run_stage(policy, ctx)
        outcomes.append(outcome)
        logger.info(
            f"  [{outcome.stage.value}] {outcome.status.value} {outcome.duration_s:.1f}s "
            + (f"— {outcome.error}" if outcome.error else f"— {outcome.extras}")
        )
        if outcome.status == StageStatus.FAIL and policy.fail_policy == FailPolicy.FAST:
            aborted = True
            logger.error(f"fail-fast: [{outcome.stage.value}] 失败 → pipeline abort")
            break

    total, success_rate = _emit_summary_kpis(trade_date, outcomes)
    _notify_summary(trade_date, outcomes, total, aborted)
    logger.info(
        f"◉ pipeline 结束 {started_at} → 耗时 {total:.1f}s "
        f"成功率 {success_rate:.0%} aborted={aborted}"
    )

    return PipelineSummary(
        run_id=_current_run_id_or_empty(),
        trade_date=trade_date,
        preflight=preflight,
        stages=outcomes,
        total_duration_s=round(total, 2),
        success_rate=round(success_rate, 4),
        aborted=aborted,
    )


def _current_run_id_or_empty() -> str:
    from floatshare.observability.metrics import get_run_id

    return get_run_id() or ""


# === stage 组装 helper ======================================================


def stages_with_policies(
    policies: dict[PipelineStage, FailPolicy],
    fns: dict[PipelineStage, StageFn],
) -> tuple[StagePolicy, ...]:
    """根据 stage→policy 映射表组装 StagePolicy tuple — 方便测试 / 自定义编排."""
    return tuple(StagePolicy(s, policies[s], fns[s]) for s in policies if s in fns)
