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


@pytest.fixture
def sample_daily_with_adj() -> pd.DataFrame:
    """生成包含复权因子的日线数据（模拟送股除权）"""
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    n = len(dates)

    np.random.seed(42)

    # 基础价格序列
    base_price = 20.0
    returns = np.random.randn(n) * 0.02
    prices = base_price * np.exp(np.cumsum(returns))

    # 模拟在第50天发生10送5（1:1.5拆分）
    adj_factor = np.ones(n)
    split_idx = 50
    adj_factor[:split_idx] = 1.0
    adj_factor[split_idx:] = 1.5  # 除权后复权因子变大

    # 不复权价格需要在除权日后除以1.5
    unadj_prices = prices.copy()
    unadj_prices[split_idx:] = prices[split_idx:] / 1.5

    df = pd.DataFrame({
        "code": "000001.SZ",
        "trade_date": dates,
        "open": unadj_prices * (1 + np.random.randn(n) * 0.005),
        "high": unadj_prices * (1 + np.abs(np.random.randn(n) * 0.01)),
        "low": unadj_prices * (1 - np.abs(np.random.randn(n) * 0.01)),
        "close": unadj_prices,
        "volume": np.random.randint(100000, 1000000, n),
        "amount": unadj_prices * np.random.randint(100000, 1000000, n),
        "adj_factor": adj_factor,
    })

    return df


@pytest.fixture
def sample_dividend_data() -> pd.DataFrame:
    """生成分红送股数据"""
    data = [
        {
            "code": "000001.SZ",
            "ex_date": date(2023, 6, 15),
            "record_date": date(2023, 6, 14),
            "pay_date": date(2023, 6, 16),
            "cash_div": 0.5,  # 每股0.5元
            "bonus_ratio": 0.0,
            "transfer_ratio": 0.5,  # 10转5
            "report_period": "2022-12-31",
        },
        {
            "code": "000001.SZ",
            "ex_date": date(2022, 6, 20),
            "record_date": date(2022, 6, 17),
            "pay_date": date(2022, 6, 21),
            "cash_div": 0.3,
            "bonus_ratio": 0.0,
            "transfer_ratio": 0.0,
            "report_period": "2021-12-31",
        },
        {
            "code": "000002.SZ",
            "ex_date": date(2023, 5, 10),
            "record_date": date(2023, 5, 9),
            "pay_date": date(2023, 5, 11),
            "cash_div": 0.8,
            "bonus_ratio": 0.3,  # 10送3
            "transfer_ratio": 0.0,
            "report_period": "2022-12-31",
        },
    ]
    return pd.DataFrame(data)


@pytest.fixture
def sample_stock_list() -> pd.DataFrame:
    """生成股票列表数据"""
    data = [
        {"code": "000001.SZ", "ticker": "000001", "name": "平安银行", "market": "SZ", "industry": "银行"},
        {"code": "000002.SZ", "ticker": "000002", "name": "万科A", "market": "SZ", "industry": "房地产"},
        {"code": "600000.SH", "ticker": "600000", "name": "浦发银行", "market": "SH", "industry": "银行"},
        {"code": "600036.SH", "ticker": "600036", "name": "招商银行", "market": "SH", "industry": "银行"},
        {"code": "000858.SZ", "ticker": "000858", "name": "五粮液", "market": "SZ", "industry": "白酒"},
        {"code": "600519.SH", "ticker": "600519", "name": "贵州茅台", "market": "SH", "industry": "白酒"},
    ]
    return pd.DataFrame(data)


@pytest.fixture
def temp_db_path(tmp_path) -> str:
    """生成临时数据库路径"""
    return str(tmp_path / "test.db")


@pytest.fixture
def temp_state_path(tmp_path) -> str:
    """生成临时状态文件路径"""
    return str(tmp_path / "sync_state.json")


# ============================================================
# Mock 数据源 Fixtures
# ============================================================


@pytest.fixture
def mock_data_source():
    """Mock 数据源实例"""
    from tests.mocks.mock_data_source import MockDataSource
    return MockDataSource()


@pytest.fixture
def mock_data_loader():
    """Mock DataLoader 实例"""
    from tests.mocks.mock_data_source import MockDataLoader
    return MockDataLoader()


@pytest.fixture
def patch_loader(monkeypatch):
    """自动替换 DataLoader 为 Mock 版本"""
    from tests.mocks.mock_data_source import MockDataLoader
    monkeypatch.setattr("src.data.loader.DataLoader", MockDataLoader)
    monkeypatch.setattr("src.data.syncer.DataLoader", MockDataLoader)
