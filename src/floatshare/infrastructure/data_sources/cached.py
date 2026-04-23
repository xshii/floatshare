"""缓存数据源 — 把 CacheManager 包装为 Protocol 兼容的只读数据源。

作为降级链第一级：命中缓存直接返回，未命中抛 DataSourceError 让链继续。

TTL 策略 (DDIA "已封闭数据 vs 活跃数据" 原则):
- 历史日线 (date < today): 不可变 → TTL = 30天
- 当日数据: 可能盘中变化  → TTL = 60s
- 参考数据 (股票列表/日历): 变化极少 → TTL = 24h / 7天
- 技术因子: 派生数据     → 不缓存 (每次现算)
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from floatshare.domain.enums import AdjustType, TimeFrame
from floatshare.infrastructure.storage.cache import CacheManager
from floatshare.interfaces.data_source import DataSourceError

# TTL 常量 (秒)
TTL_HISTORICAL = 30 * 86400  # 30 天 — 已封闭的历史数据
TTL_INTRADAY = 60  # 60 秒 — 当日盘中数据
TTL_STOCK_LIST = 86400  # 24 小时 — 股票列表
TTL_CALENDAR = 7 * 86400  # 7 天 — 交易日历


def smart_daily_ttl(end: date | None = None) -> int:
    """根据查询的结束日期判断 TTL。

    - 查询包含当日数据 (end=None 或 end >= today) → 短 TTL (60s)
    - 查询纯历史数据 (end < today)               → 长 TTL (30天)
    """
    today = date.today()
    if end is None or end >= today:
        return TTL_INTRADAY
    return TTL_HISTORICAL


def _cache_key(*parts: object) -> str:
    return ":".join(str(p) for p in parts)


class CachedSource:
    """内存+pickle 两级缓存，实现全部 6 个 Protocol。"""

    def __init__(self, cache: CacheManager | None = None) -> None:
        self._cache = cache or CacheManager()

    @property
    def cache(self) -> CacheManager:
        return self._cache

    def _get_or_miss(self, key: str) -> pd.DataFrame:
        result = self._cache.get(key)
        if result is None:
            raise DataSourceError(f"cache miss: {key}")
        return result

    def put(self, key: str, value: object, ttl: int | None = None) -> None:
        """供上层写入缓存（远程数据拉到后回写）。"""
        self._cache.set(key, value, ttl)

    # --- DailyDataSource -----------------------------------------------------
    def get_daily(
        self,
        code: str,
        start: date | None = None,
        end: date | None = None,
        adj: AdjustType = AdjustType.QFQ,
    ) -> pd.DataFrame:
        return self._get_or_miss(_cache_key("daily", code, start, end, adj))

    # --- MinuteDataSource ----------------------------------------------------
    def get_minute(
        self,
        code: str,
        start: date | None = None,
        end: date | None = None,
        freq: TimeFrame = TimeFrame.MIN_5,
    ) -> pd.DataFrame:
        return self._get_or_miss(_cache_key("minute", code, start, end, freq))

    # --- IndexDataSource -----------------------------------------------------
    def get_index_daily(
        self,
        code: str,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        return self._get_or_miss(_cache_key("index", code, start, end))

    # --- CalendarSource ------------------------------------------------------
    def get_trade_calendar(
        self,
        start: date | None = None,
        end: date | None = None,
    ) -> list[date]:
        result = self._cache.get(_cache_key("calendar", start, end))
        if result is None:
            raise DataSourceError(f"cache miss: calendar:{start}:{end}")
        return result

    # --- StockListSource -----------------------------------------------------
    def get_stock_list(self) -> pd.DataFrame:
        return self._get_or_miss("stock_list")


# --- 缓存 key 构造，供 DataLoader 回写时使用 --------------------------------


def daily_key(
    code: str,
    start: date | None = None,
    end: date | None = None,
    adj: AdjustType = AdjustType.QFQ,
) -> str:
    return _cache_key("daily", code, start, end, adj)
