"""AKShare 适配器 — 实现 Daily/Index/Financial/Calendar/StockList Protocol。"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, cast

import pandas as pd

from floatshare.domain.enums import AdjustType, TimeFrame
from floatshare.domain.schema import normalize_ohlcv
from floatshare.interfaces.data_source import DataSourceError

if TYPE_CHECKING:
    from types import ModuleType


def _to_market(ticker: str) -> str:
    if ticker.startswith("6"):
        return f"{ticker}.SH"
    if ticker.startswith(("0", "3")):
        return f"{ticker}.SZ"
    if ticker.startswith(("4", "8")):
        return f"{ticker}.BJ"
    return ticker


def _strip_market(code: str) -> str:
    return code.split(".")[0] if "." in code else code


class AKShareSource:
    """AKShare 免费数据源。"""

    def __init__(self) -> None:
        self._ak: ModuleType | None = None

    @property
    def ak(self) -> ModuleType:
        if self._ak is None:
            try:
                import akshare as ak
            except ImportError as exc:  # pragma: no cover
                raise DataSourceError("请先安装 akshare: pip install akshare") from exc
            self._ak = ak
        return self._ak

    # --- StockListSource -----------------------------------------------------
    def get_stock_list(self) -> pd.DataFrame:
        df = self.ak.stock_info_a_code_name()
        df = df.rename(columns={"code": "ticker"})
        df["code"] = df["ticker"].apply(_to_market)
        return df

    # --- DailyDataSource -----------------------------------------------------
    def get_daily(
        self,
        code: str,
        start: date | None = None,
        end: date | None = None,
        adj: AdjustType = AdjustType.QFQ,
    ) -> pd.DataFrame:
        ticker = _strip_market(code)
        try:
            df = self.ak.stock_zh_a_hist(
                symbol=ticker,
                period="daily",
                start_date=start.strftime("%Y%m%d") if start else "19900101",
                end_date=end.strftime("%Y%m%d") if end else None,
                adjust=adj.value,
            )
        except Exception as exc:  # pragma: no cover
            raise DataSourceError(f"akshare get_daily 失败: {exc}") from exc

        if df.empty:
            return df
        df = df.rename(
            columns={
                "日期": "trade_date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "涨跌幅": "pct_change",
                "涨跌额": "change",
                "换手率": "turnover",
            }
        )
        df["code"] = code
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values("trade_date").reset_index(drop=True)
        return normalize_ohlcv(df)

    # --- RawDailySource ------------------------------------------------------
    def get_raw_daily(
        self,
        code: str,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        """返回未复权日线（adjust=""）。"""
        return self.get_daily(code, start, end, adj=AdjustType.NONE)

    # --- MinuteDataSource ----------------------------------------------------
    def get_minute(
        self,
        code: str,
        start: date | None = None,
        end: date | None = None,
        freq: TimeFrame = TimeFrame.MIN_5,
    ) -> pd.DataFrame:
        ticker = _strip_market(code)
        period_map = {
            TimeFrame.MIN_1: "1",
            TimeFrame.MIN_5: "5",
            TimeFrame.MIN_15: "15",
            TimeFrame.MIN_30: "30",
            TimeFrame.MIN_60: "60",
        }
        period = period_map.get(freq, "5")
        try:
            df = self.ak.stock_zh_a_hist_min_em(symbol=ticker, period=period)
        except Exception as exc:  # pragma: no cover
            raise DataSourceError(f"akshare get_minute 失败: {exc}") from exc
        if df.empty:
            return df
        df = df.rename(
            columns={
                "时间": "datetime",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
            }
        )
        df["code"] = code
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df.reset_index(drop=True)

    # --- IndexDataSource -----------------------------------------------------
    def get_index_daily(
        self,
        code: str,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        ticker = _strip_market(code)
        for prefix in ("sh", "sz"):
            try:
                df = self.ak.stock_zh_index_daily(symbol=f"{prefix}{ticker}")
                break
            except Exception:
                df = pd.DataFrame()
        if df.empty:
            return df
        df = df.rename(columns={"date": "trade_date"})
        df["code"] = code
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        if start is not None:
            df = cast(pd.DataFrame, df[df["trade_date"] >= pd.Timestamp(start)])
        if end is not None:
            df = cast(pd.DataFrame, df[df["trade_date"] <= pd.Timestamp(end)])
        return normalize_ohlcv(df.reset_index(drop=True))

    # --- CalendarSource ------------------------------------------------------
    def get_trade_calendar(
        self,
        start: date | None = None,
        end: date | None = None,
    ) -> list[date]:
        try:
            df = self.ak.tool_trade_date_hist_sina()
        except Exception as exc:  # pragma: no cover
            raise DataSourceError(f"akshare 交易日历失败: {exc}") from exc
        dates = pd.to_datetime(df["trade_date"]).dt.date.tolist()
        if start:
            dates = [d for d in dates if d >= start]
        if end:
            dates = [d for d in dates if d <= end]
        return sorted(dates)
