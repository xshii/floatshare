"""AKShare数据源适配器"""

from datetime import date
from typing import List, Optional
import pandas as pd

from ..loader import BaseDataSource


class AKShareSource(BaseDataSource):
    """AKShare数据源（免费）"""

    def __init__(self):
        self._ak = None

    @property
    def ak(self):
        """延迟初始化akshare"""
        if self._ak is None:
            try:
                import akshare as ak

                self._ak = ak
            except ImportError:
                raise ImportError("请先安装akshare: pip install akshare")
        return self._ak

    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        df = self.ak.stock_info_a_code_name()
        df = df.rename(columns={"code": "ticker"})
        # 添加code列（带市场后缀）
        df["code"] = df["ticker"].apply(self._add_market_suffix)
        return df

    def _add_market_suffix(self, ticker: str) -> str:
        """添加市场后缀"""
        if ticker.startswith("6"):
            return f"{ticker}.SH"
        elif ticker.startswith(("0", "3")):
            return f"{ticker}.SZ"
        elif ticker.startswith(("4", "8")):
            return f"{ticker}.BJ"
        return ticker

    def _remove_market_suffix(self, code: str) -> str:
        """移除市场后缀"""
        return code.split(".")[0] if "." in code else code

    def get_daily(
        self,
        code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        adj: str = "qfq",
    ) -> pd.DataFrame:
        """获取日线数据"""
        ticker = self._remove_market_suffix(code)

        # 复权类型映射
        adj_map = {"qfq": "qfq", "hfq": "hfq", None: ""}
        adjust = adj_map.get(adj, "qfq")

        try:
            df = self.ak.stock_zh_a_hist(
                symbol=ticker,
                period="daily",
                start_date=start_date.strftime("%Y%m%d") if start_date else "19900101",
                end_date=end_date.strftime("%Y%m%d") if end_date else None,
                adjust=adjust,
            )
        except Exception:
            return pd.DataFrame()

        if df.empty:
            return df

        # 重命名列
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
        df = df.sort_values("trade_date")

        return df.reset_index(drop=True)

    def get_minute(
        self,
        code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        freq: str = "5min",
    ) -> pd.DataFrame:
        """获取分钟线数据"""
        ticker = self._remove_market_suffix(code)

        # 频率映射
        period_map = {"1min": "1", "5min": "5", "15min": "15", "30min": "30", "60min": "60"}
        period = period_map.get(freq, "5")

        try:
            df = self.ak.stock_zh_a_hist_min_em(symbol=ticker, period=period)
        except Exception:
            return pd.DataFrame()

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

    def get_index_daily(
        self,
        code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """获取指数日线数据"""
        ticker = self._remove_market_suffix(code)

        try:
            df = self.ak.stock_zh_index_daily(symbol=f"sh{ticker}")
        except Exception:
            try:
                df = self.ak.stock_zh_index_daily(symbol=f"sz{ticker}")
            except Exception:
                return pd.DataFrame()

        if df.empty:
            return df

        df = df.rename(
            columns={
                "date": "trade_date",
            }
        )

        df["code"] = code
        df["trade_date"] = pd.to_datetime(df["trade_date"])

        if start_date:
            df = df[df["trade_date"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["trade_date"] <= pd.Timestamp(end_date)]

        return df.reset_index(drop=True)

    def get_financial(
        self,
        code: str,
        report_type: str = "quarterly",
    ) -> pd.DataFrame:
        """获取财务数据"""
        ticker = self._remove_market_suffix(code)

        try:
            # 获取财务指标
            df = self.ak.stock_financial_analysis_indicator(symbol=ticker)
        except Exception:
            return pd.DataFrame()

        return df

    def get_trade_calendar(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[date]:
        """获取交易日历"""
        try:
            df = self.ak.tool_trade_date_hist_sina()
            dates = pd.to_datetime(df["trade_date"]).dt.date.tolist()

            if start_date:
                dates = [d for d in dates if d >= start_date]
            if end_date:
                dates = [d for d in dates if d <= end_date]

            return sorted(dates)
        except Exception:
            return []
