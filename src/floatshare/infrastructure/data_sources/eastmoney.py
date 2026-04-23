"""东方财富 — 走 akshare 的 em 接口，仅日线 + 股票列表。"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

from floatshare.domain.enums import AdjustType
from floatshare.domain.schema import normalize_ohlcv
from floatshare.interfaces.data_source import DataSourceError

if TYPE_CHECKING:
    from types import ModuleType


class EastMoneySource:
    """东方财富数据源（仅 daily/list）。"""

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

    def get_stock_list(self) -> pd.DataFrame:
        try:
            df = self.ak.stock_zh_a_spot_em()
        except Exception as exc:  # pragma: no cover
            raise DataSourceError(f"eastmoney get_stock_list 失败: {exc}") from exc
        df = df[["代码", "名称"]].copy()
        df = df.rename(columns={"代码": "ticker", "名称": "name"})
        df["code"] = df["ticker"].apply(self._to_market)
        return df

    @staticmethod
    def _to_market(ticker: str) -> str:
        if ticker.startswith("6"):
            return f"{ticker}.SH"
        if ticker.startswith(("0", "3")):
            return f"{ticker}.SZ"
        if ticker.startswith(("4", "8")):
            return f"{ticker}.BJ"
        return ticker

    def get_daily(
        self,
        code: str,
        start: date | None = None,
        end: date | None = None,
        adj: AdjustType = AdjustType.QFQ,
    ) -> pd.DataFrame:
        ticker = code.split(".")[0] if "." in code else code
        try:
            df = self.ak.stock_zh_a_hist(
                symbol=ticker,
                period="daily",
                start_date=start.strftime("%Y%m%d") if start else "19900101",
                end_date=end.strftime("%Y%m%d") if end else None,
                adjust=adj.value,
            )
        except Exception as exc:  # pragma: no cover
            raise DataSourceError(f"eastmoney get_daily 失败: {exc}") from exc
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
            }
        )
        df["code"] = code
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values("trade_date").reset_index(drop=True)
        return normalize_ohlcv(df)
