"""Pytest配置和fixtures"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta
import sys
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_daily_data() -> pd.DataFrame:
    """生成示例日线数据"""
    dates = pd.date_range("2023-01-01", periods=100, freq="B")

    np.random.seed(42)
    n = len(dates)

    # 生成价格数据
    base_price = 10.0
    returns = np.random.randn(n) * 0.02
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        "code": "000001.SZ",
        "trade_date": dates,
        "open": prices * (1 + np.random.randn(n) * 0.005),
        "high": prices * (1 + np.abs(np.random.randn(n) * 0.01)),
        "low": prices * (1 - np.abs(np.random.randn(n) * 0.01)),
        "close": prices,
        "volume": np.random.randint(100000, 1000000, n),
        "amount": prices * np.random.randint(100000, 1000000, n),
    })

    return df


@pytest.fixture
def multi_stock_data() -> pd.DataFrame:
    """生成多只股票数据"""
    codes = ["000001.SZ", "000002.SZ", "600000.SH"]
    all_data = []

    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="B")

    for code in codes:
        n = len(dates)
        base_price = np.random.uniform(5, 50)
        returns = np.random.randn(n) * 0.02
        prices = base_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            "code": code,
            "trade_date": dates,
            "open": prices * (1 + np.random.randn(n) * 0.005),
            "high": prices * (1 + np.abs(np.random.randn(n) * 0.01)),
            "low": prices * (1 - np.abs(np.random.randn(n) * 0.01)),
            "close": prices,
            "volume": np.random.randint(100000, 1000000, n),
            "amount": prices * np.random.randint(100000, 1000000, n),
        })
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)


@pytest.fixture
def sample_returns() -> pd.Series:
    """生成示例收益率序列"""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    returns = pd.Series(
        np.random.randn(len(dates)) * 0.01 + 0.0003,  # 均值略大于0
        index=dates,
    )
    return returns


@pytest.fixture
def sample_portfolio():
    """创建示例组合"""
    from src.account.portfolio import Portfolio, Position

    portfolio = Portfolio(
        name="test",
        initial_capital=1_000_000,
    )

    # 添加一些持仓
    portfolio.add_position("000001.SZ", 1000, 10.0)
    portfolio.update_price("000001.SZ", 11.0)

    portfolio.add_position("000002.SZ", 500, 20.0)
    portfolio.update_price("000002.SZ", 19.0)

    return portfolio
