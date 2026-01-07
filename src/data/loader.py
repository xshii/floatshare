"""数据加载器"""

from abc import ABC, abstractmethod
from datetime import date
from typing import List, Optional, Dict, Any

import pandas as pd

from src.data.models import StockDaily, StockInfo


class DataLoader:
    """数据加载器 - 统一数据访问接口"""

    def __init__(self, source: str = "tushare"):
        """
        初始化数据加载器

        Args:
            source: 数据源名称 (tushare, akshare, local)
        """
        self.source = source
        self._adapter = self._get_adapter(source)

    def _get_adapter(self, source: str) -> "BaseDataSource":
        """获取数据源适配器"""
        from .sources.tushare import TushareSource
        from .sources.akshare import AKShareSource
        from .sources.baostock import BaostockSource

        adapters = {
            "tushare": TushareSource,
            "akshare": AKShareSource,
            "baostock": BaostockSource,
        }

        if source not in adapters:
            raise ValueError(f"不支持的数据源: {source}")

        return adapters[source]()

    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        return self._adapter.get_stock_list()

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
                 推荐存储不复权数据，使用时动态复权

        Returns:
            DataFrame with columns: trade_date, open, high, low, close, volume, amount, adj_factor
            价格为不复权原始价格，adj_factor 为累计复权因子
        """
        return self._adapter.get_daily(code, start_date, end_date, adj)

    def get_minute(
        self,
        code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        freq: str = "5min",
    ) -> pd.DataFrame:
        """获取分钟线数据"""
        return self._adapter.get_minute(code, start_date, end_date, freq)

    def get_index_daily(
        self,
        code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """获取指数日线数据"""
        return self._adapter.get_index_daily(code, start_date, end_date)

    def get_financial(
        self,
        code: str,
        report_type: str = "quarterly",
    ) -> pd.DataFrame:
        """获取财务数据"""
        return self._adapter.get_financial(code, report_type)

    def get_trade_calendar(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[date]:
        """获取交易日历"""
        return self._adapter.get_trade_calendar(start_date, end_date)

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
        return self._adapter.get_dividend(code, start_date, end_date)


class BaseDataSource(ABC):
    """数据源基类"""

    @abstractmethod
    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        pass

    @abstractmethod
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
            adj: 复权类型 (None-不复权[默认], qfq-前复权, hfq-后复权)

        Returns:
            DataFrame，价格默认为不复权，包含 adj_factor 列
        """
        pass

    @abstractmethod
    def get_minute(
        self,
        code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        freq: str = "5min",
    ) -> pd.DataFrame:
        """获取分钟线数据"""
        pass

    @abstractmethod
    def get_index_daily(
        self,
        code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """获取指数日线数据"""
        pass

    @abstractmethod
    def get_financial(
        self,
        code: str,
        report_type: str = "quarterly",
    ) -> pd.DataFrame:
        """获取财务数据"""
        pass

    @abstractmethod
    def get_trade_calendar(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[date]:
        """获取交易日历"""
        pass

    @abstractmethod
    def get_dividend(
        self,
        code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        获取分红送股数据

        Returns:
            DataFrame with columns: code, ex_date, cash_div, bonus_ratio, transfer_ratio, ...
        """
        pass
