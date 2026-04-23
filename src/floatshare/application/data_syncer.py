"""数据同步器 — watermark 增量拉取 + 读时复权。

核心设计 (DDIA "真相来源 + 派生数据" 原则):
- 存储: raw_daily (不复权) + adj_factor 分离，是 Source of Truth
- 计算: 复权价 = f(raw, factor, adj_type)，是纯函数派生，不持久化
- 同步: 用 sync_watermark 记录每只股票的最新同步日期，增量追加
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
from typing import TYPE_CHECKING, cast

import pandas as pd

from floatshare.domain.enums import AdjustType
from floatshare.domain.records import AdjFactor, RawDaily
from floatshare.interfaces.data_source import DataSourceError
from floatshare.observability import logger

if TYPE_CHECKING:
    from floatshare.infrastructure.storage.database import DatabaseStorage
    from floatshare.interfaces.data_source import (
        AdjFactorSource,
        CalendarSource,
        RawDailySource,
    )


class DataSyncer:
    """增量数据同步 + 读时复权 — 实现 DailyDataSource Protocol，可插入降级链。

    工作流:
    1. get_daily(code) → 检查 watermark
    2. watermark 过期 → 增量拉取 raw + adj_factor → 写入 SQLite
    3. 从本地 raw_daily 读取 + apply_adjustment → 返回复权后 DataFrame
    """

    def __init__(
        self,
        db: DatabaseStorage,
        raw_sources: list[RawDailySource],
        adj_sources: list[AdjFactorSource],
        calendar_sources: list[CalendarSource] | None = None,
    ) -> None:
        self._db = db
        self._raw_sources = raw_sources
        self._adj_sources = adj_sources
        self._calendar_sources = calendar_sources or []

    # --- DailyDataSource Protocol 实现 --------------------------------------

    def get_daily(
        self,
        code: str,
        start: date | None = None,
        end: date | None = None,
        adj: AdjustType = AdjustType.QFQ,
    ) -> pd.DataFrame:
        """先同步再读取，保证本地数据是最新的。"""
        self._sync_if_stale(code)

        raw = self._db.load(RawDaily.TABLE, code, start, end)
        if raw.empty:
            raise DataSourceError(f"syncer: 本地无 {code} 数据 (sync 后仍为空)")

        if adj == AdjustType.NONE:
            return raw

        factors = self._db.load("adj_factor", code, columns="trade_date, adj_factor")
        if factors.empty:
            logger.warning(f"{code} 无复权因子，返回不复权数据")
            return raw

        return apply_adjustment(raw, factors, adj)

    # --- 增量同步核心 -------------------------------------------------------

    def _sync_if_stale(self, code: str) -> None:
        """检查水位线，仅在过期时同步。"""
        watermark = self._db.get_watermark(code)
        today = date.today()

        if watermark is not None and watermark >= today - timedelta(days=1):
            return  # 水位线是昨天或今天，不需要同步

        self._do_sync(code, since=watermark)

    def _do_sync(self, code: str, since: date | None) -> None:
        """从远程增量拉取 raw daily + adj_factor，写入本地。"""
        fetch_start = since + timedelta(days=1) if since else None

        # 1. 拉取 raw daily (增量) — 用 callable 而非反射，保持类型安全
        raw = _try_sources(
            self._raw_sources,
            lambda src: src.get_raw_daily(code=code, start=fetch_start),
        )
        if raw is not None and not raw.empty:
            saved = self._db.save(RawDaily, raw)
            logger.info(f"[sync] {code} raw_daily +{saved} rows (since {fetch_start})")

        # 2. 拉取 adj_factor (全量 — 因子可能追溯修正)
        adj = _try_sources(
            self._adj_sources,
            lambda src: src.get_adj_factor(code=code),
        )
        if adj is not None and not adj.empty:
            self._db.save(AdjFactor, adj)
            logger.debug(f"[sync] {code} adj_factor updated ({len(adj)} rows)")

        # 3. 更新水位线
        latest = self._db.latest_date(RawDaily.TABLE, code)
        if latest is not None:
            source_name = (
                self._raw_sources[0].__class__.__name__ if self._raw_sources else "unknown"
            )
            self._db.update_watermark(code, latest, source_name)

    # --- 交易日历同步 -------------------------------------------------------

    def sync_calendar(self) -> None:
        """同步交易日历到本地（全量覆盖）。"""
        for src in self._calendar_sources:
            try:
                dates = src.get_trade_calendar()
                if dates:
                    self._db.save_trade_calendar(dates)
                    logger.info(f"[sync] trade_calendar updated ({len(dates)} days)")
                    return
            except DataSourceError as exc:
                logger.warning(f"{type(src).__name__}.get_trade_calendar failed: {exc}")


# --- 模块级纯函数 -----------------------------------------------------------


def _try_sources(
    sources: list,
    invoke: Callable,
) -> pd.DataFrame | None:
    """从 source 链中逐个尝试 invoke(src)，返回第一个成功的结果。"""
    for src in sources:
        try:
            return invoke(src)
        except Exception as exc:
            logger.warning(f"{type(src).__name__} failed: {exc}")
    return None


def apply_adjustment(
    raw: pd.DataFrame,
    factors: pd.DataFrame,
    adj: AdjustType,
) -> pd.DataFrame:
    """纯函数：原始价 × 复权因子 → 复权价。

    前复权 (QFQ): factor_i / factor_latest  — 最近的价格不变，历史价格向下调整
    后复权 (HFQ): factor_i / factor_earliest — 最早的价格不变，后续价格向上调整
    """
    if adj == AdjustType.NONE or factors.empty:
        return raw

    df = raw.copy()
    factor_map = cast(pd.Series, factors.set_index("trade_date")["adj_factor"])
    df["_adj"] = cast(pd.Series, df["trade_date"]).map(factor_map)

    has_factor = cast(pd.Series, df["_adj"].notna())
    if not has_factor.any():
        return df.drop(columns=["_adj"])

    if adj == AdjustType.QFQ:
        reference = df.loc[has_factor, "_adj"].iloc[-1]
    else:  # HFQ
        reference = df.loc[has_factor, "_adj"].iloc[0]

    ratio = df["_adj"] / reference
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df.loc[has_factor, col] = df.loc[has_factor, col] * ratio[has_factor]

    return df.drop(columns=["_adj"])
