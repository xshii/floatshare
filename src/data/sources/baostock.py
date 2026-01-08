"""Baostock 数据源

免费数据源，无需注册，适合作为备用数据源。
http://baostock.com/

特点：
- 完全免费
- 无需 token
- 数据延迟约 1 天
- 支持日线、周线、月线、5/15/30/60分钟线
"""

import logging
from datetime import date
from typing import List, Optional

import pandas as pd

from src.data.loader import BaseDataSource

logger = logging.getLogger(__name__)


class BaostockSource(BaseDataSource):
    """Baostock 数据源"""

    def __init__(self):
        self._logged_in = False

    def _login(self):
        """登录 baostock"""
        if self._logged_in:
            return

        try:
            import baostock as bs
            lg = bs.login()
            if lg.error_code != "0":
                raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")
            self._logged_in = True
            logger.debug("baostock 登录成功")
        except ImportError:
            raise ImportError("请安装 baostock: pip install baostock")

    def _logout(self):
        """登出 baostock"""
        if self._logged_in:
            import baostock as bs
            bs.logout()
            self._logged_in = False

    def _convert_code(self, code: str) -> str:
        """
        转换股票代码格式
        000001.SZ -> sz.000001
        600000.SH -> sh.600000
        """
        if "." in code:
            ticker, market = code.split(".")
            market = market.lower()
            return f"{market}.{ticker}"
        return code

    def _convert_code_back(self, bs_code: str) -> str:
        """
        转换回标准格式
        sz.000001 -> 000001.SZ
        sh.600000 -> 600000.SH
        """
        if "." in bs_code:
            market, ticker = bs_code.split(".")
            return f"{ticker}.{market.upper()}"
        return bs_code

    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        import baostock as bs

        self._login()

        rs = bs.query_stock_basic()
        if rs.error_code != "0":
            logger.error(f"获取股票列表失败: {rs.error_msg}")
            return pd.DataFrame()

        data = []
        while rs.next():
            data.append(rs.get_row_data())

        df = pd.DataFrame(data, columns=rs.fields)

        # 只保留 A 股
        df = df[df["type"] == "1"]  # 1=股票

        # 转换格式
        df["code"] = df["code"].apply(self._convert_code_back)
        df = df.rename(columns={"code_name": "name"})

        return df[["code", "name"]].reset_index(drop=True)

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
            code: 股票代码 (如 000001.SZ)
            start_date: 开始日期
            end_date: 结束日期
            adj: 复权类型 (None/qfq/hfq)，baostock 默认不复权
        """
        import baostock as bs

        self._login()

        bs_code = self._convert_code(code)

        # 日期格式转换
        start_str = start_date.strftime("%Y-%m-%d") if start_date else "1990-01-01"
        end_str = end_date.strftime("%Y-%m-%d") if end_date else date.today().strftime("%Y-%m-%d")

        # 复权参数
        adjust_flag = "3"  # 默认不复权
        if adj == "qfq":
            adjust_flag = "2"  # 前复权
        elif adj == "hfq":
            adjust_flag = "1"  # 后复权

        # 查询数据
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume,amount,adjustflag,turn,pctChg",
            start_date=start_str,
            end_date=end_str,
            frequency="d",
            adjustflag=adjust_flag,
        )

        if rs.error_code != "0":
            logger.error(f"获取 {code} 日线数据失败: {rs.error_msg}")
            return pd.DataFrame()

        data = []
        while rs.next():
            data.append(rs.get_row_data())

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=rs.fields)

        # 转换数据类型
        df = df.rename(columns={
            "date": "trade_date",
            "turn": "turnover",
            "pctChg": "pct_change",
        })

        numeric_cols = ["open", "high", "low", "close", "volume", "amount", "pct_change", "turnover"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["code"] = code

        # 添加复权因子列
        df["adj_factor"] = 1.0

        return df[["code", "trade_date", "open", "high", "low", "close", "volume", "amount", "adj_factor"]].reset_index(drop=True)

    def get_minute(
        self,
        code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        freq: str = "5min",
    ) -> pd.DataFrame:
        """获取分钟线数据"""
        import baostock as bs

        self._login()

        bs_code = self._convert_code(code)

        start_str = start_date.strftime("%Y-%m-%d") if start_date else "1990-01-01"
        end_str = end_date.strftime("%Y-%m-%d") if end_date else date.today().strftime("%Y-%m-%d")

        # 频率映射
        freq_map = {"5min": "5", "15min": "15", "30min": "30", "60min": "60"}
        bs_freq = freq_map.get(freq, "5")

        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,time,open,high,low,close,volume,amount",
            start_date=start_str,
            end_date=end_str,
            frequency=bs_freq,
        )

        if rs.error_code != "0":
            logger.error(f"获取 {code} 分钟线数据失败: {rs.error_msg}")
            return pd.DataFrame()

        data = []
        while rs.next():
            data.append(rs.get_row_data())

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=rs.fields)

        # 转换数据类型
        numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
        df["code"] = code

        return df[["code", "datetime", "open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)

    def get_index_daily(
        self,
        code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """获取指数日线数据"""
        # baostock 指数代码格式: sh.000001 (上证指数)
        return self.get_daily(code, start_date, end_date)

    def get_financial(
        self,
        code: str,
        report_type: str = "quarterly",
    ) -> pd.DataFrame:
        """获取财务数据"""
        import baostock as bs

        self._login()

        bs_code = self._convert_code(code)

        # 获取季度财务数据
        rs = bs.query_profit_data(code=bs_code, year=date.today().year, quarter=4)

        if rs.error_code != "0":
            logger.error(f"获取 {code} 财务数据失败: {rs.error_msg}")
            return pd.DataFrame()

        data = []
        while rs.next():
            data.append(rs.get_row_data())

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=rs.fields)
        df["code"] = code

        return df

    def get_trade_calendar(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[date]:
        """获取交易日历"""
        import baostock as bs

        self._login()

        start_str = start_date.strftime("%Y-%m-%d") if start_date else "1990-01-01"
        end_str = end_date.strftime("%Y-%m-%d") if end_date else date.today().strftime("%Y-%m-%d")

        rs = bs.query_trade_dates(start_date=start_str, end_date=end_str)

        if rs.error_code != "0":
            logger.error(f"获取交易日历失败: {rs.error_msg}")
            return []

        dates = []
        while rs.next():
            row = rs.get_row_data()
            if row[1] == "1":  # is_trading_day
                dates.append(date.fromisoformat(row[0]))

        return dates

    def get_dividend(
        self,
        code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """获取分红送股数据"""
        import baostock as bs

        self._login()

        bs_code = self._convert_code(code)

        rs = bs.query_dividend_data(code=bs_code, year="", yearType="report")

        if rs.error_code != "0":
            logger.error(f"获取 {code} 分红数据失败: {rs.error_msg}")
            return pd.DataFrame()

        data = []
        while rs.next():
            data.append(rs.get_row_data())

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=rs.fields)
        df["code"] = code

        # 日期过滤
        if "dividOperateDate" in df.columns:
            df["ex_date"] = pd.to_datetime(df["dividOperateDate"])
            if start_date:
                df = df[df["ex_date"] >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df["ex_date"] <= pd.Timestamp(end_date)]

        return df

    def __del__(self):
        """析构时登出"""
        self._logout()
