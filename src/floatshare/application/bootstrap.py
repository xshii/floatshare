"""组合根 — 把 observability.metrics 的 sink 插到 infrastructure 上.

分层:
    observability.metrics 只知道 MetricsSink Protocol  (不依赖 infrastructure)
    infrastructure.storage.metrics_sink 提供 SqliteMetricsSink 实现
    本模块 (application 层) 在入口处把两者拧起来, 组合根唯一位置

用法 (cli / scripts 入口):
    from floatshare.application.bootstrap import setup_metrics
    from floatshare.observability.metrics import run_context

    def main() -> None:
        db = setup_metrics()                # 构造 db + init_tables + 注入 sink
        with run_context() as run_id:        # 所有 record_counter/record_kpi 带这个 id
            ...
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from floatshare.observability.metrics import run_context, set_sink

if TYPE_CHECKING:
    from floatshare.infrastructure.storage.database import DatabaseStorage


def setup_metrics(
    db_path: str | Path = "data/floatshare.db",
    *,
    existing_db: DatabaseStorage | None = None,
) -> DatabaseStorage:
    """Wire SqliteMetricsSink 到 observability.metrics. 返回 db 供后续使用.

    Args:
        db_path: DB 路径, 默认 data/floatshare.db
        existing_db: 已构造好的 DatabaseStorage 时直接复用 (避免双重 init_tables)

    Returns:
        DatabaseStorage 实例 (入口可直接用它 save/load 数据).
    """
    from floatshare.infrastructure.storage.database import DatabaseStorage
    from floatshare.infrastructure.storage.metrics_sink import SqliteMetricsSink

    db = existing_db if existing_db is not None else DatabaseStorage(db_path)
    if existing_db is None:
        db.init_tables()
    set_sink(SqliteMetricsSink(db))
    return db


@contextmanager
def cli_metrics_run(
    db_path: str | Path = "data/floatshare.db",
    *,
    existing_db: DatabaseStorage | None = None,
    run_id: str | None = None,
) -> Iterator[tuple[DatabaseStorage, str]]:
    """CLI 入口合并模板: setup_metrics + run_context — 消除 main 拆 _run 的重复.

    用法:
        def main() -> None:
            with cli_metrics_run() as (db, run_id):
                ...                     # 直接写主逻辑, 不必拆 _run 函数

    Yields:
        (db, run_id): DatabaseStorage + 当次执行的 uuid (12 位 hex)
    """
    db = setup_metrics(db_path, existing_db=existing_db)
    with run_context(run_id) as rid:
        yield db, rid
