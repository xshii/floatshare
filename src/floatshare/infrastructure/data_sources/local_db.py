"""本地数据库数据源 — 把 DatabaseStorage 包装为 Protocol 兼容的只读数据源。

作为降级链中的一级：命中本地库直接返回，未命中抛 DataSourceError 让链继续。
优先读 raw_daily (新表)，stock_list 读 stock_info。
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from floatshare.domain.enums import AdjustType
from floatshare.domain.records import RawDaily
from floatshare.infrastructure.storage.database import DatabaseStorage
from floatshare.interfaces.data_source import DataSourceError


class LocalDbSource:
    """SQLite 本地数据源，仅支持 daily 和 stock_list。"""

    def __init__(self, db: DatabaseStorage | None = None) -> None:
        self._db = db or DatabaseStorage()

    @property
    def db(self) -> DatabaseStorage:
        return self._db

    def get_daily(
        self,
        code: str,
        start: date | None = None,
        end: date | None = None,
        adj: AdjustType = AdjustType.QFQ,
    ) -> pd.DataFrame:
        df = self._db.load(RawDaily.TABLE, code, start, end)
        if df.empty:
            raise DataSourceError(f"local db miss: daily {code}")
        return df

    def get_stock_list(self) -> pd.DataFrame:
        df = self._db.load_stock_list()
        if df.empty:
            raise DataSourceError("local db miss: stock_list")
        return df
