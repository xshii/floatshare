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
        adj: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取日线数据

        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            adj: 复权类型 (None-不复权[默认], qfq-前复权, hfq-后复权)

        Returns:
            DataFrame，价格默认为不复权，包含 adj_factor 列
        """
        ticker = self._remove_market_suffix(code)
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

            # 同时获取后复权数据来计算复权因子
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

        # 计算复权因子: adj_factor = 后复权价格 / 不复权价格
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

    def get_dividend(
        self,
        code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        获取分红送股数据

        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame with columns:
            - code: 股票代码
            - ex_date: 除权除息日
            - record_date: 股权登记日
            - pay_date: 派息日
            - cash_div: 每股现金分红（元）
            - bonus_ratio: 每股送股比例
            - transfer_ratio: 每股转增比例
            - allot_ratio: 每股配股比例
            - allot_price: 配股价格
        """
        ticker = self._remove_market_suffix(code)

        try:
            # 获取分红送配数据
            df = self.ak.stock_fhps_em(symbol=ticker)
        except Exception:
            return pd.DataFrame()

        if df.empty:
            return df

        # 重命名列（东方财富数据格式）
        column_mapping = {
            "报告期": "report_period",
            "业绩披露日期": "ann_date",
            "除权除息日": "ex_date",
            "股权登记日": "record_date",
            "派息日": "pay_date",
            "送股比例": "bonus_ratio",
            "转增比例": "transfer_ratio",
            "派息比例": "cash_div",
            "配股比例": "allot_ratio",
            "配股价格": "allot_price",
            "方案进度": "progress",
        }

        # 只重命名存在的列
        rename_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=rename_cols)

        df["code"] = code

        # 转换日期格式
        date_cols = ["ex_date", "record_date", "pay_date", "ann_date"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # 转换比例（东方财富的格式是"10送3"，需要转换为0.3）
        ratio_cols = ["bonus_ratio", "transfer_ratio", "allot_ratio"]
        for col in ratio_cols:
            if col in df.columns:
                # 如果是数值型，除以10转换为比例
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0) / 10

        # 转换现金分红（东方财富的格式是"10派X元"，X就是每10股分红）
        if "cash_div" in df.columns:
            df["cash_div"] = pd.to_numeric(df["cash_div"], errors="coerce").fillna(0) / 10

        # 过滤日期
        if "ex_date" in df.columns:
            df = df.dropna(subset=["ex_date"])
            if start_date:
                df = df[df["ex_date"] >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df["ex_date"] <= pd.Timestamp(end_date)]

        return df.reset_index(drop=True)
