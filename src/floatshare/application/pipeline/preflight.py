"""Pipeline preflight — 启动前环境/依赖/权限检查, 失败立即 fail-fast.

设计:
    - 每个 check 是 PreflightCheck dataclass, 有 name + run() -> PreflightResult
    - DEFAULT_CHECKS 是生产默认组, run_preflight() 逐个跑并聚合
    - 非致命项用 severity="warn" (默认 fatal)

用法:
    from floatshare.application.pipeline.preflight import run_preflight
    results = run_preflight()         # 含打印 + notify
    if any(r.severity == "fatal" and not r.ok for r in results):
        raise SystemExit(1)
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from floatshare.observability import logger, notify


@dataclass(frozen=True, slots=True)
class PreflightResult:
    name: str
    ok: bool
    severity: str  # "fatal" | "warn"
    message: str


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    """可调用的前置检查, name 唯一标识, run() 无副作用 (除非注明)."""

    name: str
    run: Callable[[], PreflightResult]


# === 具体检查函数 ============================================================


def _check_tushare_token() -> PreflightResult:
    tok = os.environ.get("TUSHARE_TOKEN", "")
    if len(tok) < 20:
        return PreflightResult(
            "tushare_token",
            ok=False,
            severity="fatal",
            message=f"TUSHARE_TOKEN 缺失或过短 (len={len(tok)}, 期望 ≥20). 检查 .env",
        )
    return PreflightResult(
        "tushare_token",
        ok=True,
        severity="fatal",
        message=f"token 长度 {len(tok)}, 前缀 {tok[:4]}***",
    )


def _check_db_writable(db_path: str | Path = "data/floatshare.db") -> PreflightResult:
    """副作用提示: 会 mkdir 父目录 (幂等, 为保证后续 sync 能写入)."""
    p = Path(db_path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        return PreflightResult(
            "db_writable", ok=False, severity="fatal", message=f"父目录创建失败: {exc}"
        )
    if p.exists() and not os.access(p, os.W_OK):
        return PreflightResult(
            "db_writable",
            ok=False,
            severity="fatal",
            message=f"{p} 不可写 (权限 / 占用)",
        )
    if not os.access(p.parent, os.W_OK):
        return PreflightResult(
            "db_writable",
            ok=False,
            severity="fatal",
            message=f"父目录 {p.parent} 不可写",
        )
    size_mb = p.stat().st_size / 1_000_000 if p.exists() else 0
    return PreflightResult(
        "db_writable",
        ok=True,
        severity="fatal",
        message=f"{p} 可写, 当前 {size_mb:.1f}MB",
    )


def _check_disk_space(path: str | Path = ".", min_free_gb: float = 2.0) -> PreflightResult:
    total, used, free = shutil.disk_usage(Path(path).resolve())
    free_gb = free / 1_000_000_000
    if free_gb < min_free_gb:
        return PreflightResult(
            "disk_space",
            ok=False,
            severity="fatal",
            message=f"磁盘剩余 {free_gb:.1f}GB < 最低 {min_free_gb}GB",
        )
    low_headroom = free_gb < min_free_gb * 2
    return PreflightResult(
        "disk_space",
        ok=True,
        severity="fatal",  # fail 时的 severity; 这里是 ok=True 所以不参与判定
        message=(
            f"剩余 {free_gb:.1f}GB (使用 {used / total:.1%})"
            + (" ⚠ 余量偏低" if low_headroom else "")
        ),
    )


def _check_trade_calendar_fresh(
    db_path: str | Path = "data/floatshare.db",
    max_stale_days: int = 30,
) -> PreflightResult:
    """trade_calendar 最新日期不应太旧 — 否则 sync 的日期推断会偏."""
    p = Path(db_path)
    if not p.exists():
        return PreflightResult(
            "trade_calendar_fresh",
            ok=False,
            severity="warn",
            message=f"{p} 不存在, 跳过 (首次运行)",
        )
    try:
        with sqlite3.connect(str(p)) as conn:
            row = conn.execute("SELECT MAX(trade_date) FROM trade_calendar").fetchone()
    except sqlite3.OperationalError:
        return PreflightResult(
            "trade_calendar_fresh",
            ok=False,
            severity="warn",
            message="trade_calendar 表不存在, 跳过 (首次 init)",
        )
    if not row or not row[0]:
        return PreflightResult(
            "trade_calendar_fresh",
            ok=False,
            severity="warn",
            message="trade_calendar 空, 请先 sync lifecycle",
        )
    max_d = datetime.fromisoformat(row[0]).date() if "T" in row[0] else date.fromisoformat(row[0])
    age = (date.today() - max_d).days
    if age > max_stale_days:
        return PreflightResult(
            "trade_calendar_fresh",
            ok=False,
            severity="fatal",
            message=f"trade_calendar 最新 {max_d} 距今 {age} 天, 超阈值 {max_stale_days}",
        )
    return PreflightResult(
        "trade_calendar_fresh",
        ok=True,
        severity="fatal",
        message=f"最新 {max_d} (距今 {age} 天)",
    )


# === 默认检查组 + 调度 ======================================================


DEFAULT_CHECKS: tuple[PreflightCheck, ...] = (
    PreflightCheck("tushare_token", _check_tushare_token),
    PreflightCheck("db_writable", _check_db_writable),
    PreflightCheck("disk_space", _check_disk_space),
    PreflightCheck("trade_calendar_fresh", _check_trade_calendar_fresh),
)


def run_preflight(
    checks: tuple[PreflightCheck, ...] = DEFAULT_CHECKS,
    *,
    notify_on_fail: bool = True,
) -> list[PreflightResult]:
    """跑所有 preflight 检查, 聚合结果 + 打印 + notify."""
    results: list[PreflightResult] = []
    for chk in checks:
        try:
            r = chk.run()
        except Exception as exc:
            r = PreflightResult(
                chk.name, ok=False, severity="fatal", message=f"check 异常: {exc!r}"
            )
        results.append(r)
        icon = "✓" if r.ok else ("⚠️ " if r.severity == "warn" else "✗")
        logger.info(f"  {icon} [{r.name}] {r.message}")

    fatal_fails = [r for r in results if not r.ok and r.severity == "fatal"]
    if fatal_fails and notify_on_fail:
        body = "\n".join(f"- {r.name}: {r.message}" for r in fatal_fails)
        notify(f"❌ Preflight fail {len(fatal_fails)} 致命", body)
    return results


def has_fatal_failure(results: list[PreflightResult]) -> bool:
    return any(not r.ok and r.severity == "fatal" for r in results)
