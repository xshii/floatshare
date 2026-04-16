"""数据清洗模块测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from floatshare.infrastructure.data_cleaner import (
    clean_daily,
    fill_missing_prices,
    resample,
)


class TestCleaner:
    def test_clean_removes_duplicates(self, sample_daily_data):
        df = pd.concat([sample_daily_data, sample_daily_data.head(5)])
        cleaned = clean_daily(df)
        assert len(cleaned) == len(sample_daily_data)

    def test_clean_sorts_by_date(self, sample_daily_data):
        df = sample_daily_data.sample(frac=1, random_state=0)
        cleaned = clean_daily(df)
        assert cleaned["trade_date"].is_monotonic_increasing

    def test_fill_missing_prices(self):
        df = pd.DataFrame(
            {
                "code": ["000001.SZ"] * 5,
                "trade_date": pd.date_range("2023-01-01", periods=5),
                "open": [10.0, np.nan, 10.5, np.nan, 11.0],
                "high": [10.5, 10.6, np.nan, 11.0, 11.2],
                "low": [9.8, np.nan, 10.2, 10.5, np.nan],
                "close": [10.2, 10.4, 10.6, np.nan, 11.0],
                "volume": [1000, np.nan, 1200, 1100, 1300],
            }
        )
        filled = fill_missing_prices(df)
        assert not bool(filled["open"].isna().any())
        assert not bool(filled["close"].isna().any())
        assert not bool(filled["volume"].isna().any())

    def test_resample_to_weekly(self, sample_daily_data):
        weekly = resample(sample_daily_data, freq="W")
        assert len(weekly) < len(sample_daily_data)
        assert {"open", "close"} <= set(weekly.columns)
