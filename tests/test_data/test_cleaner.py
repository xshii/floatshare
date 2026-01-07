"""数据清洗测试"""

import pytest
import pandas as pd
import numpy as np
from src.data.cleaner import DataCleaner


class TestDataCleaner:
    """DataCleaner测试类"""

    def test_clean_daily_data_removes_duplicates(self, sample_daily_data):
        """测试去重"""
        # 添加重复数据
        df = pd.concat([sample_daily_data, sample_daily_data.head(5)])

        cleaned = DataCleaner.clean_daily_data(df)

        assert len(cleaned) == len(sample_daily_data)

    def test_clean_daily_data_sorts_by_date(self, sample_daily_data):
        """测试排序"""
        # 打乱顺序
        df = sample_daily_data.sample(frac=1)

        cleaned = DataCleaner.clean_daily_data(df)

        dates = cleaned["trade_date"].tolist()
        assert dates == sorted(dates)

    def test_fill_missing_prices(self):
        """测试缺失值填充"""
        df = pd.DataFrame({
            "code": ["000001.SZ"] * 5,
            "trade_date": pd.date_range("2023-01-01", periods=5),
            "open": [10.0, np.nan, 10.5, np.nan, 11.0],
            "high": [10.5, 10.6, np.nan, 11.0, 11.2],
            "low": [9.8, np.nan, 10.2, 10.5, np.nan],
            "close": [10.2, 10.4, 10.6, np.nan, 11.0],
            "volume": [1000, np.nan, 1200, 1100, 1300],
        })

        filled = DataCleaner._fill_missing_prices(df)

        assert not filled["open"].isna().any()
        assert not filled["close"].isna().any()
        assert filled["volume"].isna().sum() == 0 or filled["volume"].iloc[1] == 0

    def test_resample_to_weekly(self, sample_daily_data):
        """测试重采样到周线"""
        weekly = DataCleaner.resample(sample_daily_data, freq="W")

        assert len(weekly) < len(sample_daily_data)
        assert "open" in weekly.columns
        assert "close" in weekly.columns
