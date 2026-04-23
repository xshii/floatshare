"""数据加载器 — 链式降级 + 缓存回写。

对外只依赖 `interfaces` 中定义的 Protocol，构造时由 cli/工厂注入实现。
降级顺序: Cache → SQLite → Tushare(付费主源) → AKShare(免费备份) → EastMoney(末选)
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

from floatshare.domain.enums import AdjustType, TimeFrame
from floatshare.interfaces.data_source import (
    CalendarSource,
    DailyDataSource,
    DataSourceError,
    IndexDataSource,
    MinuteDataSource,
    StockListSource,
)
from floatshare.observability import logger

if TYPE_CHECKING:
    from typing import Any


class AllSourcesFailed(DataSourceError):
    """所有候选源都失败时抛出。"""


class DataLoader:
    """统一数据访问入口。

    每种能力维护一个有序的 source 列表，按顺序尝试，第一个成功的胜出。
    成功后可通过 on_daily_fetched / on_stock_list_fetched 回调写入本地缓存/数据库。
    """

    def __init__(
        self,
        daily: list[DailyDataSource] | None = None,
        minute: list[MinuteDataSource] | None = None,
        index: list[IndexDataSource] | None = None,
        calendar: list[CalendarSource] | None = None,
        stock_list: list[StockListSource] | None = None,
        on_daily_fetched: Callable[[str, pd.DataFrame], None] | None = None,
        on_stock_list_fetched: Callable[[pd.DataFrame], None] | None = None,
    ) -> None:
        self.daily = daily or []
        self.minute = minute or []
        self.index = index or []
        self.calendar = calendar or []
        self.stock_list = stock_list or []
        self._on_daily_fetched = on_daily_fetched
        self._on_stock_list_fetched = on_stock_list_fetched

    @staticmethod
    def _try_chain(
        chain: list[Any],
        op_name: str,
        invoke: Callable[[Any], Any],
    ) -> Any:
        last_exc: Exception | None = None
        for src in chain:
            try:
                return invoke(src)
            except DataSourceError as exc:
                logger.warning(f"{type(src).__name__}.{op_name} failed: {exc}")
                last_exc = exc
        raise AllSourcesFailed(f"All sources failed for {op_name}; last error: {last_exc}")

    def get_daily(
        self,
        code: str,
        start: date | None = None,
        end: date | None = None,
        adj: AdjustType = AdjustType.QFQ,
    ) -> pd.DataFrame:
        df: pd.DataFrame = self._try_chain(
            self.daily,
            "get_daily",
            lambda s: s.get_daily(code, start, end, adj),
        )
        if self._on_daily_fetched and not df.empty:
            try:
                self._on_daily_fetched(code, df)
            except Exception as exc:
                logger.debug(f"daily 回写失败 (非致命): {exc}")
        return df

    def get_minute(
        self,
        code: str,
        start: date | None = None,
        end: date | None = None,
        freq: TimeFrame = TimeFrame.MIN_5,
    ) -> pd.DataFrame:
        return self._try_chain(
            self.minute,
            "get_minute",
            lambda s: s.get_minute(code, start, end, freq),
        )

    def get_index_daily(
        self,
        code: str,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        return self._try_chain(
            self.index,
            "get_index_daily",
            lambda s: s.get_index_daily(code, start, end),
        )

    def get_trade_calendar(
        self,
        start: date | None = None,
        end: date | None = None,
    ) -> list[date]:
        return self._try_chain(
            self.calendar,
            "get_trade_calendar",
            lambda s: s.get_trade_calendar(start, end),
        )

    def get_stock_list(self) -> pd.DataFrame:
        df: pd.DataFrame = self._try_chain(
            self.stock_list,
            "get_stock_list",
            lambda s: s.get_stock_list(),
        )
        if self._on_stock_list_fetched and not df.empty:
            try:
                self._on_stock_list_fetched(df)
            except Exception as exc:
                logger.debug(f"stock_list 回写失败 (非致命): {exc}")
        return df


def create_default_loader() -> DataLoader:
    """工厂函数 — 组装完整降级链。

    日线降级: Cache → DataSyncer(local raw+adj 增量同步) → AKShare → EastMoney
    其它数据: Cache → Tushare → AKShare → EastMoney

    DataSyncer 内部的远程源顺序: Tushare(付费) → AKShare(免费)

    这是 application 层允许 import infrastructure 的唯一例外（composition root）。
    用户可以完全绕过这个函数，自己注入想要的 source 组合。
    """
    from floatshare.application.data_syncer import DataSyncer
    from floatshare.infrastructure.data_sources.akshare import AKShareSource
    from floatshare.infrastructure.data_sources.cached import (
        TTL_STOCK_LIST,
        CachedSource,
    )
    from floatshare.infrastructure.data_sources.eastmoney import EastMoneySource
    from floatshare.infrastructure.data_sources.local_db import LocalDbSource
    from floatshare.infrastructure.data_sources.tushare import TushareSource
    from floatshare.infrastructure.storage.database import DatabaseStorage

    db = DatabaseStorage()
    db.init_tables()

    cache_src = CachedSource()
    db_src = LocalDbSource(db)
    ts = TushareSource()
    ak = AKShareSource()
    em = EastMoneySource()

    # DataSyncer: 增量同步 raw_daily + adj_factor，读时复权
    syncer = DataSyncer(
        db=db,
        raw_sources=[ts, ak],  # Tushare 优先，AKShare 兜底
        adj_sources=[ts],  # 复权因子只有 Tushare 提供
        calendar_sources=[ts, ak],
    )

    def _writeback_stock_list(df: pd.DataFrame) -> None:
        cache_src.put("stock_list", df, ttl=TTL_STOCK_LIST)
        db.save_stock_list(df)

    return DataLoader(
        # 日线: cache → syncer(增量同步+读时复权) → AKShare(纯远程兜底) → EastMoney
        daily=[cache_src, syncer, ak, em],
        minute=[ak],
        index=[ts, ak],
        calendar=[ts, ak],
        stock_list=[cache_src, db_src, ts, ak, em],
        on_stock_list_fetched=_writeback_stock_list,
    )
