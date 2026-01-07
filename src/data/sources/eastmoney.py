"""东方财富数据源适配器"""

from datetime import date
from typing import List, Optional
import pandas as pd

from ..loader import BaseDataSource


class EastMoneySource(BaseDataSource):
    """东方财富数据源"""

    def __init__(self):
        self._ak = None

    @property
    def ak(self):
        """使用akshare获取东方财富数据"""
        if self._ak is None:
            try:
                import akshare as ak

                self._ak = ak
            except ImportError:
                raise ImportError("请先安装akshare: pip install akshare")
        return self._ak

    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        try:
            df = self.ak.stock_zh_a_spot_em()
            df = df[["代码", "名称"]].copy()
            df = df.rename(columns={"代码": "ticker", "名称": "name"})
            df["code"] = df["ticker"].apply(self._add_market_suffix)
            return df
        except Exception:
            return pd.DataFrame()

    def _add_market_suffix(self, ticker: str) -> str:
        """添加市场后缀"""
        if ticker.startswith("6"):
            return f"{ticker}.SH"
        elif ticker.startswith(("0", "3")):
            return f"{ticker}.SZ"
        elif ticker.startswith(("4", "8")):
            return f"{ticker}.BJ"
        return ticker

    def get_daily(
        self,
        code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        adj: str = "qfq",
    ) -> pd.DataFrame:
        """获取日线数据（使用东方财富接口）"""
        ticker = code.split(".")[0] if "." in code else code

        adj_map = {"qfq": "qfq", "hfq": "hfq", None: ""}

        try:
            df = self.ak.stock_zh_a_hist(
                symbol=ticker,
                period="daily",
                start_date=start_date.strftime("%Y%m%d") if start_date else "19900101",
                end_date=end_date.strftime("%Y%m%d") if end_date else None,
                adjust=adj_map.get(adj, "qfq"),
            )
        except Exception:
            return pd.DataFrame()

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

        return df.sort_values("trade_date").reset_index(drop=True)

    def get_minute(
        self,
        code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        freq: str = "5min",
    ) -> pd.DataFrame:
        """获取分钟线数据"""
        return pd.DataFrame()

    def get_index_daily(
        self,
        code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """获取指数日线数据"""
        return pd.DataFrame()

    def get_financial(
        self,
        code: str,
        report_type: str = "quarterly",
    ) -> pd.DataFrame:
        """获取财务数据"""
        return pd.DataFrame()

    def get_trade_calendar(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[date]:
        """获取交易日历"""
        return []
