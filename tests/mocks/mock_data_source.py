"""Mock 数据源 - 用于单元测试，避免真实网络请求"""

import json
from datetime import date
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.data.loader import BaseDataSource


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class MockDataSource(BaseDataSource):
    """Mock 数据源 - 从本地 fixtures 文件加载数据"""

    def __init__(self, fixtures_dir: Optional[Path] = None):
        self.fixtures_dir = fixtures_dir or FIXTURES_DIR
        self._daily_bars = None
        self._stock_list = None
        self._dividends = None
        self._hs300 = None

    def _load_json(self, filename: str) -> dict:
        """加载 JSON fixture"""
        filepath = self.fixtures_dir / filename
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    @property
    def daily_bars(self) -> dict:
        if self._daily_bars is None:
            self._daily_bars = self._load_json("daily_bars.json")
        return self._daily_bars

    @property
    def stock_list_data(self) -> list:
        if self._stock_list is None:
            self._stock_list = self._load_json("stock_list.json")
        return self._stock_list

    @property
    def dividends_data(self) -> dict:
        if self._dividends is None:
            self._dividends = self._load_json("dividends.json")
        return self._dividends

    @property
    def hs300_data(self) -> list:
        if self._hs300 is None:
            self._hs300 = self._load_json("hs300_constituents.json")
        return self._hs300

    def get_daily(
        self,
        code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        adj: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取日线数据"""
        bars = self.daily_bars.get(code, [])
        if not bars:
            return pd.DataFrame()

        df = pd.DataFrame(bars)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["code"] = code

        # 日期过滤
        if start_date:
            df = df[df["trade_date"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["trade_date"] <= pd.Timestamp(end_date)]

        return df.reset_index(drop=True)

    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        if not self.stock_list_data:
            return pd.DataFrame()
        return pd.DataFrame(self.stock_list_data)

    def get_dividend(
        self,
        code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """获取分红数据"""
        divs = self.dividends_data.get(code, [])
        if not divs:
            return pd.DataFrame()

        df = pd.DataFrame(divs)
        df["ex_date"] = pd.to_datetime(df["ex_date"])
        df["record_date"] = pd.to_datetime(df["record_date"])
        df["code"] = code

        if start_date:
            df = df[df["ex_date"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["ex_date"] <= pd.Timestamp(end_date)]

        return df.reset_index(drop=True)

    def get_index_constituents(self, index_code: str) -> pd.DataFrame:
        """获取指数成分股"""
        if index_code in ("000300.SH", "000300"):
            return pd.DataFrame(self.hs300_data)
        return pd.DataFrame()


class MockDataLoader:
    """Mock DataLoader - 直接替换 DataLoader 使用"""

    def __init__(self, source: str = "mock"):
        self.source = source
        self._adapter = MockDataSource()

    def get_daily(
        self,
        code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        adj: Optional[str] = None,
    ) -> pd.DataFrame:
        return self._adapter.get_daily(code, start_date, end_date, adj)

    def get_stock_list(self) -> pd.DataFrame:
        return self._adapter.get_stock_list()

    def get_dividend(
        self,
        code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        return self._adapter.get_dividend(code, start_date, end_date)


def patch_data_loader():
    """
    Pytest fixture 用法:

    @pytest.fixture
    def mock_loader():
        with patch_data_loader():
            yield

    或者直接在 conftest.py 中:

    @pytest.fixture(autouse=True)
    def auto_mock_loader(monkeypatch):
        monkeypatch.setattr("src.data.loader.DataLoader", MockDataLoader)
    """
    from unittest.mock import patch
    return patch("src.data.loader.DataLoader", MockDataLoader)
