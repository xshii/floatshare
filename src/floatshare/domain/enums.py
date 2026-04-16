"""枚举常量 — 替换全部魔法字符串。"""

from __future__ import annotations

from enum import Enum, StrEnum


class AdjustType(StrEnum):
    """复权类型。"""

    NONE = ""
    QFQ = "qfq"  # 前复权
    HFQ = "hfq"  # 后复权


class Market(StrEnum):
    """交易所市场。"""

    SH = "SH"  # 上海
    SZ = "SZ"  # 深圳
    BJ = "BJ"  # 北京


class Direction(StrEnum):
    """交易方向。"""

    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    """订单类型。"""

    MARKET = "market"
    LIMIT = "limit"


class DataSourceKind(StrEnum):
    """数据源标识。"""

    AKSHARE = "akshare"
    TUSHARE = "tushare"
    EASTMONEY = "eastmoney"
    BAOSTOCK = "baostock"
    LOCAL = "local"


class ReportType(StrEnum):
    """财报频次。"""

    QUARTERLY = "quarterly"
    SEMIANNUAL = "semiannual"
    ANNUAL = "annual"


# 非字符串语义的纯枚举
class TimeFrame(Enum):
    """K 线频次。"""

    DAY = "D"
    WEEK = "W"
    MONTH = "M"
    MIN_1 = "1min"
    MIN_5 = "5min"
    MIN_15 = "15min"
    MIN_30 = "30min"
    MIN_60 = "60min"
