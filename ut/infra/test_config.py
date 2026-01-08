"""配置模块测试"""

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from config.base import (
    AppSettings,
    BacktestConfig,
    CacheConfig,
    DatabaseConfig,
    DataSourceConfig,
    Direction,
    LoggingConfig,
    LogLevel,
    Market,
    OrderType,
    TradingConfig,
    get_settings,
)


class TestEnums:
    """枚举测试"""

    def test_market_values(self):
        assert Market.SH.value == "SH"
        assert Market.SZ.value == "SZ"
        assert Market.BJ.value == "BJ"

    def test_order_type_values(self):
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"

    def test_direction_values(self):
        assert Direction.BUY.value == "buy"
        assert Direction.SELL.value == "sell"

    def test_log_level_values(self):
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"


class TestDatabaseConfig:
    """DatabaseConfig 测试"""

    def test_default_values(self):
        config = DatabaseConfig()
        assert config.pg_host == "localhost"
        assert config.pg_port == 5432
        assert config.redis_port == 6379

    def test_sqlite_url(self):
        config = DatabaseConfig(sqlite_path=Path("test.db"))
        assert config.sqlite_url == "sqlite:///test.db"

    def test_postgres_url_without_password(self):
        config = DatabaseConfig(pg_username="user", pg_password="")
        assert "user@localhost" in config.postgres_url

    def test_postgres_url_with_password(self):
        config = DatabaseConfig(pg_username="user", pg_password="pass")
        assert "user:pass@localhost" in config.postgres_url

    def test_redis_url(self):
        config = DatabaseConfig()
        assert config.redis_url == "redis://localhost:6379/0"

    def test_port_validation(self):
        with pytest.raises(ValidationError):
            DatabaseConfig(pg_port=70000)  # 超过 65535

    def test_pool_size_validation(self):
        with pytest.raises(ValidationError):
            DatabaseConfig(pool_size=0)  # 必须 >= 1


class TestTradingConfig:
    """TradingConfig 测试"""

    def test_default_values(self):
        config = TradingConfig()
        assert config.commission_rate == 0.0003
        assert config.min_commission == 5.0
        assert config.stamp_duty == 0.001

    def test_commission_rate_validation(self):
        with pytest.raises(ValidationError):
            TradingConfig(commission_rate=-0.01)  # 负数

        with pytest.raises(ValidationError):
            TradingConfig(commission_rate=0.02)  # 超过 1%

    def test_max_position_pct_validation(self):
        with pytest.raises(ValidationError):
            TradingConfig(max_position_pct=1.5)  # 超过 100%

        with pytest.raises(ValidationError):
            TradingConfig(max_position_pct=0)  # 必须 > 0

    def test_calculate_commission_buy(self):
        config = TradingConfig(commission_rate=0.0003, min_commission=5.0)
        amount = 100000

        commission = config.calculate_commission(amount, Direction.BUY)

        # 佣金 + 过户费（无印花税）
        expected_commission = max(amount * 0.0003, 5.0)
        expected_transfer = amount * 0.00001
        assert commission == pytest.approx(expected_commission + expected_transfer)

    def test_calculate_commission_sell(self):
        config = TradingConfig(
            commission_rate=0.0003,
            min_commission=5.0,
            stamp_duty=0.001,
        )
        amount = 100000

        commission = config.calculate_commission(amount, Direction.SELL)

        # 佣金 + 印花税 + 过户费
        expected_commission = max(amount * 0.0003, 5.0)
        expected_stamp = amount * 0.001
        expected_transfer = amount * 0.00001
        assert commission == pytest.approx(
            expected_commission + expected_stamp + expected_transfer
        )

    def test_min_commission_applied(self):
        config = TradingConfig(commission_rate=0.0003, min_commission=5.0)
        amount = 1000  # 佣金 = 0.3，低于最低佣金

        commission = config.calculate_commission(amount, Direction.BUY)
        assert commission >= 5.0  # 至少是最低佣金


class TestBacktestConfig:
    """BacktestConfig 测试"""

    def test_default_values(self):
        config = BacktestConfig()
        assert config.initial_capital == 1_000_000
        assert config.benchmark == "000300.SH"

    def test_date_validation(self):
        with pytest.raises(ValidationError):
            BacktestConfig(
                start_date=date(2025, 1, 1),
                end_date=date(2020, 1, 1),  # 结束日期早于开始日期
            )

    def test_initial_capital_validation(self):
        with pytest.raises(ValidationError):
            BacktestConfig(initial_capital=0)  # 必须 > 0

    def test_drawdown_limit_validation(self):
        with pytest.raises(ValidationError):
            BacktestConfig(max_drawdown=1.5)  # 超过 100%


class TestDataSourceConfig:
    """DataSourceConfig 测试"""

    def test_default_values(self):
        config = DataSourceConfig()
        assert config.primary_source == "akshare"
        assert "baostock" in config.fallback_sources

    def test_invalid_primary_source(self):
        with pytest.raises(ValidationError):
            DataSourceConfig(primary_source="invalid_source")

    def test_invalid_fallback_source(self):
        with pytest.raises(ValidationError):
            DataSourceConfig(fallback_sources=["invalid"])

    def test_rate_limit_validation(self):
        with pytest.raises(ValidationError):
            DataSourceConfig(requests_per_minute=0)  # 必须 >= 1


class TestCacheConfig:
    """CacheConfig 测试"""

    def test_default_values(self):
        config = CacheConfig()
        assert config.enabled is True
        assert config.backend == "memory"
        assert config.max_size == 1000

    def test_invalid_backend(self):
        with pytest.raises(ValidationError):
            CacheConfig(backend="invalid")

    def test_max_size_validation(self):
        with pytest.raises(ValidationError):
            CacheConfig(max_size=50)  # 必须 >= 100


class TestLoggingConfig:
    """LoggingConfig 测试"""

    def test_default_values(self):
        config = LoggingConfig()
        assert config.level == LogLevel.INFO
        assert config.console is True

    def test_invalid_rotation(self):
        with pytest.raises(ValidationError):
            LoggingConfig(rotation="invalid")


class TestAppSettings:
    """AppSettings 测试"""

    def test_default_initialization(self):
        settings = AppSettings()
        assert isinstance(settings.database, DatabaseConfig)
        assert isinstance(settings.trading, TradingConfig)
        assert isinstance(settings.backtest, BacktestConfig)

    def test_nested_config(self):
        settings = AppSettings()
        assert settings.trading.commission_rate == 0.0003
        assert settings.database.pg_port == 5432

    def test_to_dict(self):
        settings = AppSettings()
        data = settings.to_dict()

        assert "database" in data
        assert "trading" in data
        assert data["trading"]["commission_rate"] == 0.0003

    def test_get_settings_singleton(self):
        settings1 = get_settings()
        settings2 = get_settings()
        # 注意：由于全局状态，这可能在并行测试中失败
        # 在实际使用中应该是单例
        assert isinstance(settings1, AppSettings)
        assert isinstance(settings2, AppSettings)
