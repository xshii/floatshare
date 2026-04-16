"""数据加载器 — 链式降级 + 可选缓存。

对外只依赖 `interfaces` 中定义的 Protocol，构造时由 cli/工厂注入实现。
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from floatshare.domain.enums import AdjustType, ReportType, TimeFrame
from floatshare.interfaces.data_source import (
    CalendarSource,
    DailyDataSource,
    DataSourceError,
    FinancialDataSource,
    IndexDataSource,
    MinuteDataSource,
    StockListSource,
)
from floatshare.observability import logger


class AllSourcesFailed(DataSourceError):
    """所有候选源都失败时抛出。"""


class DataLoader:
    """统一数据访问入口。

    每种能力维护一个有序的 source 列表，按顺序尝试，第一个成功的胜出。
    """

    def __init__(
        self,
        daily: list[DailyDataSource] | None = None,
        minute: list[MinuteDataSource] | None = None,
        index: list[IndexDataSource] | None = None,
        financial: list[FinancialDataSource] | None = None,
        calendar: list[CalendarSource] | None = None,
        stock_list: list[StockListSource] | None = None,
    ) -> None:
        self.daily = daily or []
        self.minute = minute or []
        self.index = index or []
        self.financial = financial or []
        self.calendar = calendar or []
        self.stock_list = stock_list or []

    @staticmethod
    def _try_chain(
        chain: list,
        op_name: str,
        invoke,
    ):
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
        return self._try_chain(
            self.daily,
            "get_daily",
            lambda s: s.get_daily(code, start, end, adj),
        )

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

    def get_financial(
        self,
        code: str,
        report_type: ReportType = ReportType.QUARTERLY,
    ) -> pd.DataFrame:
        return self._try_chain(
            self.financial,
            "get_financial",
            lambda s: s.get_financial(code, report_type),
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
        return self._try_chain(
            self.stock_list,
            "get_stock_list",
            lambda s: s.get_stock_list(),
        )


def create_default_loader() -> DataLoader:
    """工厂函数 — 默认 AKShare 主源。

    这是 application 层允许 import infrastructure 的唯一例外（composition root）。
    用户可以完全绕过这个函数，自己注入想要的 source 组合。
    """
    from floatshare.infrastructure.data_sources.akshare import AKShareSource

    ak = AKShareSource()
    return DataLoader(
        daily=[ak],
        minute=[ak],
        index=[ak],
        financial=[ak],
        calendar=[ak],
        stock_list=[ak],
    )
