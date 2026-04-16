"""抽象接口层 — 仅 Protocol/ABC，不含实现。"""

from floatshare.interfaces.data_source import (
    CalendarSource,
    DailyDataSource,
    DataSourceError,
    FinancialDataSource,
    IndexDataSource,
    MinuteDataSource,
    StockListSource,
)

__all__ = [
    "CalendarSource",
    "DailyDataSource",
    "DataSourceError",
    "FinancialDataSource",
    "IndexDataSource",
    "MinuteDataSource",
    "StockListSource",
]
