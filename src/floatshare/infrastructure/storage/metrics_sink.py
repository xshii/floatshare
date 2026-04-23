"""SQLite 实现的 metrics sink — 把 CounterEvent/KpiSnapshot 写入对应表.

分层: 仅本模块可依赖 DatabaseStorage. observability.metrics 只知道 MetricsSink
Protocol, 不知道 SQLite.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pandas as pd

from floatshare.domain.records import CounterEvent, KpiSnapshot

if TYPE_CHECKING:
    from collections.abc import Sequence

    from floatshare.infrastructure.storage.database import DatabaseStorage


class SqliteMetricsSink:
    """实现 observability.metrics.MetricsSink Protocol, 写入 DatabaseStorage."""

    def __init__(self, db: DatabaseStorage) -> None:
        self._db = db

    def write_counter(self, events: Sequence[CounterEvent]) -> None:
        if not events:
            return
        df = pd.DataFrame([dataclasses.asdict(e) for e in events])
        self._db.save(CounterEvent, df)

    def write_kpi(self, snapshots: Sequence[KpiSnapshot]) -> None:
        if not snapshots:
            return
        df = pd.DataFrame([dataclasses.asdict(s) for s in snapshots])
        self._db.save(KpiSnapshot, df)
