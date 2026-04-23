"""记录类共享的元信息类型。

FieldSpec 驱动前端"渐进式披露" — label + primary (主要字段默认勾选) + unit。
RecordSchema Protocol 给类型检查工具提供 dataclass 必备 ClassVar 约束。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """字段在 UI 中如何呈现。"""

    label: str  # 中文名
    primary: bool = False  # 默认勾选 (True) / "展开更多" 才显示 (False)
    unit: str | None = None  # "元" / "%" / "股" / None


class RecordSchema(Protocol):
    """记录 dataclass 的最小契约 — 让 mypy 知道 ClassVar 存在。

    替代散在各处的 `# type: ignore[attr-defined]` 注释。
    """

    TABLE: ClassVar[str]
    PK: ClassVar[tuple[str, ...]]
