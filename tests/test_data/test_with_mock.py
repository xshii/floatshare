"""使用 Mock 数据源的测试"""

import pytest
import pandas as pd
from datetime import date
from unittest.mock import patch

from tests.mocks.mock_data_source import MockDataSource, MockDataLoader


class TestMockDataSource:
    """测试 Mock 数据源"""

    @pytest.fixture
    def mock_source(self):
        return MockDataSource()

    def test_get_stock_list(self, mock_source):
        """测试获取股票列表"""
        df = mock_source.get_stock_list()

        assert not df.empty
        assert "code" in df.columns
        assert "name" in df.columns
        assert len(df) >= 5

    def test_get_daily(self, mock_source):
        """测试获取日线数据"""
        df = mock_source.get_daily("000001.SZ")

        assert not df.empty
        assert "trade_date" in df.columns
        assert "open" in df.columns
        assert "close" in df.columns
        assert "adj_factor" in df.columns

    def test_get_daily_with_date_filter(self, mock_source):
        """测试日期过滤"""
        df = mock_source.get_daily(
            "000001.SZ",
            start_date=date(2024, 1, 3),
            end_date=date(2024, 1, 5),
        )

        assert not df.empty
        assert len(df) <= 3

    def test_get_daily_nonexistent_code(self, mock_source):
        """测试不存在的股票代码"""
        df = mock_source.get_daily("999999.SZ")
        assert df.empty

    def test_get_dividend(self, mock_source):
        """测试获取分红数据"""
        df = mock_source.get_dividend("000001.SZ")

        assert not df.empty
        assert "ex_date" in df.columns
        assert "cash_div" in df.columns

    def test_get_index_constituents(self, mock_source):
        """测试获取指数成分股"""
        df = mock_source.get_index_constituents("000300.SH")

        assert not df.empty
        assert "code" in df.columns
        assert "weight" in df.columns


class TestMockDataLoader:
    """测试 Mock DataLoader"""

    @pytest.fixture
    def mock_loader(self):
        return MockDataLoader()

    def test_loader_interface(self, mock_loader):
        """测试 DataLoader 接口兼容性"""
        # 股票列表
        stock_list = mock_loader.get_stock_list()
        assert not stock_list.empty

        # 日线数据
        daily = mock_loader.get_daily("000001.SZ")
        assert not daily.empty

        # 分红数据
        dividend = mock_loader.get_dividend("000001.SZ")
        assert not dividend.empty


class TestWithPatchedLoader:
    """使用 patch 替换真实 DataLoader 的测试"""

    def test_syncer_with_mock(self):
        """测试 DataSyncer 使用 Mock 数据"""
        with patch("src.data.syncer.DataLoader", MockDataLoader):
            from src.data.syncer import SourcePool

            # 创建使用 mock 的 SourcePool
            pool = SourcePool.__new__(SourcePool)
            pool.sources = ["mock"]
            pool.parallel = False
            pool.max_workers = 1
            pool.loaders = {"mock": MockDataLoader()}
            pool.health = {}

            from src.data.syncer import SourceHealth
            pool.health["mock"] = SourceHealth(name="mock")

            # 测试获取数据
            df, source = pool.fetch_daily(
                code="000001.SZ",
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 10),
            )

            assert source == "mock"
            assert not df.empty
