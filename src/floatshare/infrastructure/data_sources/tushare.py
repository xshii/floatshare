"""Tushare 适配器 — token 通过构造参数或环境变量注入，不再依赖外部 settings。"""

from __future__ import annotations

import os
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

from floatshare.domain.enums import AdjustType, ReportType, TimeFrame
from floatshare.interfaces.data_source import DataSourceError

if TYPE_CHECKING:
    from typing import Any


class TushareSource:
    """Tushare Pro 数据源。"""

    def __init__(self, token: str | None = None) -> None:
        self.token = token or os.getenv("TUSHARE_TOKEN")
        self._pro: Any = None

    @property
    def pro(self) -> Any:
        if self._pro is None:
            try:
                import tushare as ts  # pyright: ignore[reportMissingImports]  # 可选依赖
            except ImportError as exc:  # pragma: no cover
                raise DataSourceError("请先安装 tushare: pip install tushare") from exc
            if not self.token:
                raise DataSourceError("Tushare 缺少 token，请设置 TUSHARE_TOKEN 环境变量")
            ts.set_token(self.token)
            self._pro = ts.pro_api()
        return self._pro

    # --- StockListSource -----------------------------------------------------
    def get_stock_list(self) -> pd.DataFrame:
        df = self.pro.stock_basic(
            exchange="",
            list_status="L",
            fields="ts_code,symbol,name,area,industry,market,list_date",
        )
        return df.rename(columns={"ts_code": "code", "symbol": "ticker"})

    # --- DailyDataSource -----------------------------------------------------
    def get_daily(
        self,
        code: str,
        start: date | None = None,
        end: date | None = None,
        adj: AdjustType = AdjustType.QFQ,
    ) -> pd.DataFrame:
        start_str = start.strftime("%Y%m%d") if start else None
        end_str = end.strftime("%Y%m%d") if end else None
        df = self.pro.daily(ts_code=code, start_date=start_str, end_date=end_str)
        if df.empty:
            return df
        df = df.rename(columns={"ts_code": "code", "vol": "volume"})
        df["trade_date"] = pd.to_datetime(df["trade_date"])

        if adj != AdjustType.NONE:
            adj_df = self.pro.adj_factor(ts_code=code, start_date=start_str, end_date=end_str)
            if not adj_df.empty:
                df = df.merge(adj_df[["trade_date", "adj_factor"]], on="trade_date")
                df = self._apply_adj(df, adj)

        return df.sort_values("trade_date").reset_index(drop=True)

    @staticmethod
    def _apply_adj(df: pd.DataFrame, adj: AdjustType) -> pd.DataFrame:
        if adj == AdjustType.QFQ:
            factor = df["adj_factor"] / df["adj_factor"].iloc[-1]
        elif adj == AdjustType.HFQ:
            factor = df["adj_factor"] / df["adj_factor"].iloc[0]
        else:
            return df
        for col in ("open", "high", "low", "close"):
            df[col] = df[col] * factor
        return df

    # --- MinuteDataSource (high tier only) -----------------------------------
    def get_minute(
        self,
        code: str,
        start: date | None = None,
        end: date | None = None,
        freq: TimeFrame = TimeFrame.MIN_5,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    # --- IndexDataSource -----------------------------------------------------
    def get_index_daily(
        self,
        code: str,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        start_str = start.strftime("%Y%m%d") if start else None
        end_str = end.strftime("%Y%m%d") if end else None
        df = self.pro.index_daily(ts_code=code, start_date=start_str, end_date=end_str)
        if df.empty:
            return df
        df = df.rename(columns={"ts_code": "code", "vol": "volume"})
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df.sort_values("trade_date").reset_index(drop=True)

    # --- FinancialDataSource -------------------------------------------------
    def get_financial(
        self,
        code: str,
        report_type: ReportType = ReportType.QUARTERLY,
    ) -> pd.DataFrame:
        income = self.pro.income(ts_code=code)
        if income.empty:
            return pd.DataFrame()
        df = income[["ts_code", "ann_date", "end_date", "revenue", "n_income"]].copy()
        df = df.rename(columns={"ts_code": "code", "n_income": "net_profit"})
        indicator = self.pro.fina_indicator(ts_code=code)
        if not indicator.empty:
            indicator = indicator[["end_date", "eps", "bps", "roe", "roa"]]
            df = df.merge(indicator, on="end_date", how="left")
        return df

    # --- CalendarSource ------------------------------------------------------
    def get_trade_calendar(
        self,
        start: date | None = None,
        end: date | None = None,
    ) -> list[date]:
        start_str = start.strftime("%Y%m%d") if start else None
        end_str = end.strftime("%Y%m%d") if end else None
        df = self.pro.trade_cal(exchange="SSE", start_date=start_str, end_date=end_str, is_open=1)
        return sorted(pd.to_datetime(df["cal_date"]).dt.date.tolist())
