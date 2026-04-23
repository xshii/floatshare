"""Pipeline 跨层共享值对象 — 放 domain 层, application 和 ml 都可依赖.

这里只放纯数据结构 (StageContext), 不含逻辑 (orchestration 在 application/pipeline/).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class StageContext:
    """跨 stage 的共享上下文 — DB 路径 + 交易日 + stage-shared 对象.

    shared: 前一个 stage 产出, 后续 stage 消费 (e.g. feats / panel / codes).
    """

    trade_date: str
    db_path: str | Path
    shared: dict[str, Any] = field(default_factory=dict)
