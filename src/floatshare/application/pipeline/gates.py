"""Stage gate — 每个 stage 跑之前的前置依赖 + 幂等检查.

设计:
    - Gate = (name, check_fn) 组合, 返回 GateResult(ok, skip, message)
    - skip=True 表示已跑过 (幂等), 应当直接跳过 stage
    - ok=False 表示前置未满足, 按 stage 策略决定 fail-fast / fail-soft

与 preflight.py 区别:
    preflight = pipeline 启动一次 (全局环境)
    gate = 每个 stage 自己检查前置 (局部依赖)
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from floatshare.domain.enums import PipelineStage


@dataclass(frozen=True, slots=True)
class GateResult:
    """Stage gate 检查结果."""

    ok: bool  # 前置满足, 可以跑
    skip: bool = False  # 已跑过 (幂等), 建议跳过
    message: str = ""


def _count_stage_ok_for_trade_date(
    db_path: str | Path, stage: PipelineStage, trade_date: str
) -> int | None:
    """查 counter_event 里该 stage + trade_date tag 的 ok=1.0 行数.

    依据 runner.py 约定: 每条 counter 都带 tags_json 里 trade_date.
    查不到表 / DB 不存在 → 返回 None.
    """
    p = Path(db_path)
    if not p.exists():
        return None
    with sqlite3.connect(str(p)) as conn:
        try:
            # tags_json 形如 '{"trade_date": "2026-04-21", ...}', LIKE 精确匹配
            row = conn.execute(
                """
                SELECT COUNT(*) FROM counter_event
                WHERE scope = ? AND name = 'ok' AND value = 1.0
                  AND tags_json LIKE ?
                """,
                (f"pipeline/{stage.value}", f'%"trade_date": "{trade_date}"%'),
            ).fetchone()
        except sqlite3.OperationalError:
            return None
    return row[0] if row else 0


def already_ran_today(
    stage: PipelineStage,
    trade_date: str,
    db_path: str | Path = "data/floatshare.db",
) -> GateResult:
    """幂等检查: 该 stage 是否已经为这个 trade_date 成功跑过.

    "today" 不是墙钟, 而是指 trade_date — 允许回补历史日期.
    """
    n_ok = _count_stage_ok_for_trade_date(db_path, stage, trade_date)
    if n_ok is None:
        return GateResult(ok=True, skip=False, message="DB / counter_event 不存在, 首次运行")
    if n_ok > 0:
        return GateResult(
            ok=True,
            skip=True,
            message=f"{stage.value} T={trade_date} 已成功跑过 {n_ok} 次, 跳过 (幂等)",
        )
    return GateResult(ok=True, skip=False, message="未跑过, 可执行")


def prior_stage_succeeded(
    prior: PipelineStage,
    trade_date: str,
    db_path: str | Path = "data/floatshare.db",
) -> GateResult:
    """依赖检查: 前置 stage 对这个 trade_date 必须 OK 过.

    e.g. S2a 要求 S1_SYNC 先 OK (同 trade_date).
    """
    n_ok = _count_stage_ok_for_trade_date(db_path, prior, trade_date)
    if n_ok is None:
        return GateResult(ok=False, message=f"前置 {prior.value} 未跑 (DB / 表不存在)")
    if n_ok > 0:
        return GateResult(ok=True, message=f"前置 {prior.value} T={trade_date} OK")
    return GateResult(ok=False, message=f"前置 {prior.value} T={trade_date} 未成功, 本 stage 跳过")


def raw_daily_has_today(
    trade_date: str,
    db_path: str | Path = "data/floatshare.db",
    min_rows: int = 4500,
) -> GateResult:
    """S2a / S_prep 前置: raw_daily 今日行数 >= 阈值 (A 股一般 5500±50)."""
    p = Path(db_path)
    if not p.exists():
        return GateResult(ok=False, message="DB 不存在")
    with sqlite3.connect(str(p)) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM raw_daily WHERE trade_date LIKE ?", (f"{trade_date}%",)
        ).fetchone()
    n = row[0] if row else 0
    if n < min_rows:
        return GateResult(
            ok=False,
            message=f"raw_daily T={trade_date} 仅 {n} 行 (<{min_rows}), S1 可能未完成",
        )
    return GateResult(ok=True, message=f"raw_daily T={trade_date} {n} 行")


def ckpt_exists(ckpt_path: str | Path) -> GateResult:
    """S5 recommend 前置: 训好的 ckpt 存在."""
    p = Path(ckpt_path)
    if not p.exists() or p.stat().st_size < 1024:
        return GateResult(ok=False, message=f"ckpt 不存在或过小: {p}")
    return GateResult(ok=True, message=f"ckpt 就绪 {p.name} ({p.stat().st_size / 1_000_000:.1f}MB)")


# === stage → gate 声明 =======================================================
# 每个 stage 跑之前要过哪些 gate. runner.py 按这张表遍历, 调用 (trade_date, db_path) → GateResult.
# 用 functools.partial 预绑定静态参数 (如前置 stage 标识), 取代字符串 dispatch —
# 单一真相来源, mypy 可检查, 改名不会悄悄失效.

# Gate 统一签名: (trade_date: str, db_path: str | Path) → GateResult
GateFn = Callable[..., GateResult]


STAGE_GATES: dict[PipelineStage, tuple[GateFn, ...]] = {
    PipelineStage.S1_SYNC: (
        # S1 无 pipeline 内前置 (preflight 已校验 env/DB)
    ),
    PipelineStage.S1C_NEWS_INGEST: (
        # S1C 与 S1 并行, 无 pipeline 内前置 (仅依赖 preflight TUSHARE_TOKEN)
    ),
    PipelineStage.S2A_DB_INTEGRITY: (
        partial(prior_stage_succeeded, PipelineStage.S1_SYNC),
        raw_daily_has_today,
    ),
    PipelineStage.S2B_PREP_FEATURES: (
        partial(prior_stage_succeeded, PipelineStage.S1_SYNC),
        raw_daily_has_today,
    ),
    PipelineStage.S3A_TUSHARE_CHECK: (
        partial(prior_stage_succeeded, PipelineStage.S2B_PREP_FEATURES),
    ),
    PipelineStage.S3B_FEATURE_AUDIT: (
        partial(prior_stage_succeeded, PipelineStage.S2B_PREP_FEATURES),
    ),
    PipelineStage.S4_TRAIN: (partial(prior_stage_succeeded, PipelineStage.S2B_PREP_FEATURES),),
    PipelineStage.S5_RECOMMEND: (
        partial(prior_stage_succeeded, PipelineStage.S4_TRAIN),
        # TODO: ckpt_exists 需要具体 ckpt 路径, 组装 stage 时 partial 绑定
    ),
}
