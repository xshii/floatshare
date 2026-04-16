"""数据源协议 — 按 ISP 拆分，每个能力独立 Protocol。"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd

from floatshare.domain.enums import AdjustType, ReportType, TimeFrame


class DataSourceError(RuntimeError):
    """数据源失败异常。链式降级时由 application 层捕获。"""


@runtime_checkable
class DailyDataSource(Protocol):
    """提供日线行情。"""

    def get_daily(
        self,
        code: str,
        start: date | None = None,
        end: date | None = None,
        adj: AdjustType = AdjustType.QFQ,
    ) -> pd.DataFrame:
        """返回包含 trade_date/open/high/low/close/volume 列的 DataFrame。"""
        ...


@runtime_checkable
class MinuteDataSource(Protocol):
    """提供分钟线行情。"""

    def get_minute(
        self,
        code: str,
        start: date | None = None,
        end: date | None = None,
        freq: TimeFrame = TimeFrame.MIN_5,
    ) -> pd.DataFrame: ...


@runtime_checkable
class IndexDataSource(Protocol):
    """提供指数日线。"""

    def get_index_daily(
        self,
        code: str,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame: ...


@runtime_checkable
class FinancialDataSource(Protocol):
    """提供财务数据。"""

    def get_financial(
        self,
        code: str,
        report_type: ReportType = ReportType.QUARTERLY,
    ) -> pd.DataFrame: ...


@runtime_checkable
class CalendarSource(Protocol):
    """提供交易日历。"""

    def get_trade_calendar(
        self,
        start: date | None = None,
        end: date | None = None,
    ) -> list[date]: ...


@runtime_checkable
class StockListSource(Protocol):
    """提供股票列表。"""

    def get_stock_list(self) -> pd.DataFrame: ...
