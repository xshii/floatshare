"""SQL 自动生成 — 从 dataclass schema 推导 DDL/INSERT/列名。

避免在 database.py 里手写 3 处 (DDL + INSERT SQL + params dict) 字符串列名。

用法:
    from floatshare.domain.records import DailyBasic
    from floatshare.infrastructure.storage import schema_sql

    schema_sql.ddl(DailyBasic)            # CREATE TABLE IF NOT EXISTS ...
    schema_sql.upsert_sql(DailyBasic)     # INSERT OR REPLACE INTO ...
    schema_sql.row_to_params(DailyBasic, df_row)  # dict for execute()
"""

from __future__ import annotations

import dataclasses
import typing
from typing import Any, Protocol


class _RecordSchema(Protocol):
    """记录 dataclass 的最小契约 — 必须有 TABLE 和 PK ClassVar。"""

    TABLE: str
    PK: tuple[str, ...]


def _is_optional(type_hint: Any) -> bool:
    """判断字段是否可空 (X | None)。"""
    args = typing.get_args(type_hint)
    return type(None) in args


def _to_sql_type(type_hint: Any) -> str:
    """Python 类型注解 → SQLite 类型。"""
    # 解 Optional 包装
    args = typing.get_args(type_hint)
    inner = next((a for a in args if a is not type(None)), type_hint) if args else type_hint

    if inner is str:
        return "TEXT"
    if inner is int:
        return "INTEGER"
    if inner is float:
        return "REAL"
    return "REAL"  # 默认数值


def _columns(record_cls: type) -> list[dataclasses.Field]:
    """返回 dataclass 的所有字段 (跳过 ClassVar)。"""
    return list(dataclasses.fields(record_cls))


def column_names(record_cls: type) -> tuple[str, ...]:
    """返回所有列名元组。"""
    return tuple(f.name for f in _columns(record_cls))


def ddl(record_cls: type[_RecordSchema]) -> str:
    """从 dataclass 生成 CREATE TABLE IF NOT EXISTS。"""
    type_hints = typing.get_type_hints(record_cls)

    cols_sql = []
    for f in _columns(record_cls):
        sql_type = _to_sql_type(type_hints[f.name])
        nullable = (
            " NOT NULL" if (f.name in record_cls.PK or not _is_optional(type_hints[f.name])) else ""
        )
        cols_sql.append(f"{f.name} {sql_type}{nullable}")

    pk_clause = f"PRIMARY KEY ({', '.join(record_cls.PK)})"
    body = ",\n    ".join([*cols_sql, pk_clause])
    return f"CREATE TABLE IF NOT EXISTS {record_cls.TABLE} (\n    {body}\n)"


def upsert_sql(record_cls: type[_RecordSchema]) -> str:
    """从 dataclass 生成 INSERT OR REPLACE INTO ... VALUES (:col1, :col2, ...)。"""
    cols = column_names(record_cls)
    cols_clause = ", ".join(cols)
    placeholders = ", ".join(f":{c}" for c in cols)
    return f"INSERT OR REPLACE INTO {record_cls.TABLE} ({cols_clause}) VALUES ({placeholders})"


def from_row(record_cls: type, row):  # type: ignore[no-untyped-def]
    """通用 SQL row (dict 或 SQLAlchemy RowMapping) → dataclass 实例.

    用法:
        lc = from_row(StockLifecycle, row)   # 等价 StockLifecycle(**{f.name: row[f.name] ...})

    替代散落的 `{f.name: row[f.name] for f in fields(cls)}` 模板 (Cookbook 9.20:
    委托给可复用工具函数).
    """
    import dataclasses as _dc

    return record_cls(**{f.name: row[f.name] for f in _dc.fields(record_cls)})


def row_to_params(record_cls: type[_RecordSchema], row: dict) -> dict:
    """从 DataFrame row dict 提取 dataclass 字段对应的参数 dict。

    - 缺失字段补 None；额外列丢弃（避免 SQL 参数化失败）
    - 主键列若 None 则补空串避免 SQL NOT NULL 报错
    - pandas Timestamp / datetime / date 自动转 ISO 字符串（SQLite 不支持）
    """
    import datetime as _dt

    import pandas as _pd

    out: dict[str, Any] = {}
    for col in column_names(record_cls):
        v = row.get(col)
        if v is None and col in record_cls.PK:
            out[col] = ""
        elif isinstance(v, (_pd.Timestamp, _dt.datetime, _dt.date)):
            out[col] = v.isoformat()
        else:
            out[col] = v
    return out
