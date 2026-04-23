"""Sync 进度快照 — JSON 序列化对象，cli 写 / web 读。

替代散在 cli/web 之间的 `dict[str, Any]` + 字符串 key 模式 (Python Cookbook Recipe 8.10/9.20)。
- 写: SyncProgress(...).write()
- 读: SyncProgress.read() → SyncProgress | None
- 派生属性: percent / elapsed_seconds / eta_seconds / is_running
"""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import ClassVar


class SyncStatus(StrEnum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    STALE = "stale"  # 状态文件停在 RUNNING 但 pid 已不存在 (kill / crash)


class SyncStage(StrEnum):
    BATCH = "batch"  # 市场级 batch 任务阶段 (lifecycle/index_weight/macro 等)
    PER_CODE = "per-code"  # 每只股票循环阶段 (大头)
    PER_DAY = "per-day"  # 按交易日批量阶段 (top_list/top_inst)
    DONE = "done"  # 全部结束


@dataclass
class SyncProgress:
    """sync 一次执行的进度快照。"""

    pid: int
    started_at: str  # ISO 时间戳
    selected_types: list[str]
    codes_total: int
    codes_done: int = 0
    current_code: str | None = None
    status: str = SyncStatus.RUNNING.value
    stage: str = SyncStage.BATCH.value
    errors: int = 0
    finished_at: str | None = None

    PATH: ClassVar[Path] = Path("logs/sync-progress.json")

    # ------------------------------------------------------------------ I/O

    def write(self) -> None:
        """原子写 JSON，失败静默 (sync 主流程不被 IO 异常打断)。"""
        with contextlib.suppress(Exception):
            self.PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.PATH.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2))
            tmp.replace(self.PATH)

    @classmethod
    def read(cls) -> SyncProgress | None:
        """读 JSON 反序列化，文件缺失或损坏返回 None。"""
        if not cls.PATH.exists():
            return None
        try:
            data = json.loads(cls.PATH.read_text())
            return cls(**data)
        except Exception:
            return None

    # ----------------------------------------------------- 派生属性 (UI 用)

    @property
    def percent(self) -> float:
        if self.codes_total == 0:
            return 0.0
        return round(100 * self.codes_done / self.codes_total, 1)

    @property
    def elapsed_seconds(self) -> int:
        if not self.started_at:
            return 0
        delta = datetime.now() - datetime.fromisoformat(self.started_at)
        return max(int(delta.total_seconds()), 0)

    @property
    def eta_seconds(self) -> int | None:
        """剩余时间估算 — 只有 running 且 done > 0 时才有效。"""
        if self.codes_done <= 0 or self.codes_total <= self.codes_done:
            return None
        if self.status != SyncStatus.RUNNING.value:
            return None
        per_code = self.elapsed_seconds / self.codes_done
        return int(per_code * (self.codes_total - self.codes_done))

    @property
    def is_running(self) -> bool:
        return self.status == SyncStatus.RUNNING.value and self.is_pid_alive

    @property
    def is_pid_alive(self) -> bool:
        """探测 pid 是否还在 — 用 signal 0 (POSIX): 不发实际信号, 仅查存在性。"""
        if self.pid <= 0:
            return False
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return False  # 进程不存在 (kill 过 / crash)
        except PermissionError:
            return True  # 进程存在但属于别的用户 (非典型)
        except Exception:
            return False
        return True

    @property
    def effective_status(self) -> str:
        """实际状态: 如果声称 RUNNING 但 pid 已死, 返回 STALE。

        web/cli 都用这个判断, 不要直接读 self.status。
        """
        if self.status == SyncStatus.RUNNING.value and not self.is_pid_alive:
            return SyncStatus.STALE.value
        return self.status


# 默认空实例工厂 — sync 启动时用
def empty_progress(pid: int, selected: list[str], codes_total: int) -> SyncProgress:
    return SyncProgress(
        pid=pid,
        started_at=datetime.now().isoformat(timespec="seconds"),
        selected_types=selected,
        codes_total=codes_total,
    )


_ = field  # 抑制 dataclasses.field 未使用的 lint (留给未来字段扩展)
