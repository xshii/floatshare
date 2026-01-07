"""东方财富数据源适配器"""

from datetime import date
from typing import List, Optional
import pandas as pd

from src.data.loader import BaseDataSource


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
        adj: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取日线数据（使用东方财富接口）

        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            adj: 复权类型 (None-不复权[默认], qfq-前复权, hfq-后复权)

        Returns:
            DataFrame，价格默认为不复权，包含 adj_factor 列
        """
        ticker = code.split(".")[0] if "." in code else code
        start_str = start_date.strftime("%Y%m%d") if start_date else "19900101"
        end_str = end_date.strftime("%Y%m%d") if end_date else None

        try:
            # 获取不复权数据
            df = self.ak.stock_zh_a_hist(
                symbol=ticker,
                period="daily",
                start_date=start_str,
                end_date=end_str,
                adjust="",  # 不复权
            )

            # 获取后复权数据来计算复权因子
            df_hfq = self.ak.stock_zh_a_hist(
                symbol=ticker,
                period="daily",
                start_date=start_str,
                end_date=end_str,
                adjust="hfq",
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

        # 计算复权因子
        if not df_hfq.empty and "收盘" in df_hfq.columns:
            df_hfq = df_hfq.rename(columns={"日期": "trade_date", "收盘": "close_hfq"})
            df_hfq["trade_date"] = pd.to_datetime(df_hfq["trade_date"])
            df = df.merge(df_hfq[["trade_date", "close_hfq"]], on="trade_date", how="left")
            df["adj_factor"] = df["close_hfq"] / df["close"]
            df["adj_factor"] = df["adj_factor"].fillna(1.0)
            df = df.drop(columns=["close_hfq"])
        else:
            df["adj_factor"] = 1.0

        df = df.sort_values("trade_date")

        # 如果请求复权数据，动态计算
        if adj == "hfq":
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col] * df["adj_factor"]
        elif adj == "qfq":
            latest_factor = df["adj_factor"].iloc[-1]
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col] * df["adj_factor"] / latest_factor

        return df.reset_index(drop=True)

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

    def get_dividend(
        self,
        code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """获取分红送股数据（使用AKShare）"""
        ticker = code.split(".")[0] if "." in code else code

        try:
            df = self.ak.stock_fhps_em(symbol=ticker)
        except Exception:
            return pd.DataFrame()

        if df.empty:
            return df

        # 重命名列
        column_mapping = {
            "报告期": "report_period",
            "除权除息日": "ex_date",
            "股权登记日": "record_date",
            "派息日": "pay_date",
            "送股比例": "bonus_ratio",
            "转增比例": "transfer_ratio",
            "派息比例": "cash_div",
        }
        rename_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=rename_cols)

        df["code"] = code

        # 转换日期
        if "ex_date" in df.columns:
            df["ex_date"] = pd.to_datetime(df["ex_date"], errors="coerce")
            df = df.dropna(subset=["ex_date"])

        # 转换比例
        for col in ["bonus_ratio", "transfer_ratio"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0) / 10

        if "cash_div" in df.columns:
            df["cash_div"] = pd.to_numeric(df["cash_div"], errors="coerce").fillna(0) / 10

        return df.reset_index(drop=True)
