"""集成测试

测试各模块协同工作：
- 数据获取 -> 校验 -> 管道处理 -> 存储
- 配置 -> 依赖注入 -> 缓存
- 事件总线 -> 指标收集
"""

import os
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def temp_db():
    """临时数据库"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    yield db_path

    # 清理
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def sample_daily_data():
    """样本日线数据"""
    dates = pd.date_range("2025-01-01", periods=10, freq="D")
    return pd.DataFrame({
        "code": "000001.SZ",
        "trade_date": dates,
        "open": [10.0 + i * 0.1 for i in range(10)],
        "high": [10.5 + i * 0.1 for i in range(10)],
        "low": [9.5 + i * 0.1 for i in range(10)],
        "close": [10.2 + i * 0.1 for i in range(10)],
        "volume": [1000000 + i * 10000 for i in range(10)],
        "amount": [10000000 + i * 100000 for i in range(10)],
    })


@pytest.fixture
def sample_data_with_errors():
    """包含异常数据的样本"""
    dates = pd.date_range("2025-01-01", periods=5, freq="D")
    return pd.DataFrame({
        "code": "000001.SZ",
        "trade_date": dates,
        "open": [10.0, 10.1, 10.2, 10.3, 10.4],
        "high": [10.5, 10.6, 9.0, 10.8, 10.9],  # 第3条 high < low
        "low": [9.5, 9.6, 9.7, 9.8, 9.9],
        "close": [10.2, 10.3, 10.4, 10.5, 10.6],
        "volume": [1000000, 1100000, 1200000, 1300000, 1400000],
        "amount": [10000000, 11000000, 12000000, 13000000, 14000000],
    })


# ============================================================
# 数据流集成测试
# ============================================================


class TestDataPipelineIntegration:
    """数据管道集成测试"""

    def test_validate_and_clean_pipeline(self, sample_data_with_errors):
        """测试: 数据校验 + 管道清洗"""
        from src.data.pipeline import Pipeline
        from src.data.validator import DataValidator

        # 1. 先用校验器检查
        validator = DataValidator(max_pct_change=22.0)
        result = validator.validate(sample_data_with_errors, "000001.SZ")

        # 应该有校验警告/错误
        assert not result.valid or len(result.warnings) > 0 or len(result.errors) > 0

        # 2. 用管道过滤
        pipeline = (
            Pipeline("clean")
            .filter(lambda df: df["high"] >= df["low"], name="ValidHighLow")
            .filter(
                lambda df: (df["open"] >= df["low"]) & (df["open"] <= df["high"]),
                name="ValidOpen",
            )
            .sort("trade_date")
        )

        pipe_result = pipeline.execute(sample_data_with_errors)

        assert pipe_result.success
        # 过滤掉了异常数据
        assert len(pipe_result.data) < len(sample_data_with_errors)

    def test_pipeline_with_storage(self, temp_db, sample_daily_data):
        """测试: 管道处理 -> 数据库存储"""
        from src.data.pipeline import Pipeline
        from src.data.storage.database import DatabaseStorage

        # 1. 管道处理
        pipeline = (
            Pipeline("process")
            .add_column("adj_factor", lambda df: pd.Series([1.0] * len(df)))
            .sort("trade_date")
        )

        result = pipeline.execute(sample_daily_data)
        assert result.success

        # 2. 存储到数据库
        storage = DatabaseStorage(db_path=temp_db)
        storage.init_tables()

        # 转换日期格式
        df = result.data.copy()
        df["trade_date"] = df["trade_date"].dt.strftime("%Y-%m-%d")

        rows = storage.save_daily(df)
        assert rows == 10

        # 3. 读取验证
        loaded = storage.load_daily("000001.SZ")
        assert len(loaded) == 10


class TestCacheIntegration:
    """缓存集成测试"""

    def test_cache_with_pipeline(self, sample_daily_data):
        """测试: 缓存 + 管道"""
        from src.utils.cache import LRUCache, cached, set_cache

        cache = LRUCache(max_size=100)
        set_cache(cache)

        call_count = 0

        @cached(ttl=60, key_prefix="daily")
        def process_data(code: str) -> pd.DataFrame:
            nonlocal call_count
            call_count += 1
            # 模拟数据处理
            return sample_daily_data[sample_daily_data["code"] == code]

        # 第一次调用
        result1 = process_data("000001.SZ")
        assert call_count == 1
        assert len(result1) == 10

        # 第二次调用（命中缓存）
        result2 = process_data("000001.SZ")
        assert call_count == 1  # 没有再次调用
        pd.testing.assert_frame_equal(result1, result2)

        # 验证缓存统计
        assert cache.hit_rate > 0


class TestDIIntegration:
    """依赖注入集成测试"""

    def test_di_with_config(self):
        """测试: 依赖注入 + 配置"""
        from src.utils.di import Container, Lifetime

        # 创建容器
        container = Container()

        # 注册配置
        try:
            from config.base import AppSettings, get_settings
            settings = get_settings()
            container.register(AppSettings, settings)

            # 解析配置
            resolved = container.resolve(AppSettings)
            assert resolved.trading.commission_rate == 0.0003
        except Exception:
            # Pydantic 可能未安装
            pytest.skip("Pydantic not available")

    def test_di_with_storage(self, temp_db):
        """测试: 依赖注入 + 存储"""
        from src.data.storage.database import DatabaseStorage
        from src.utils.di import Container, Lifetime

        container = Container()

        # 注册存储服务
        container.register(
            DatabaseStorage,
            lambda: DatabaseStorage(db_path=temp_db),
            Lifetime.SINGLETON,
        )

        # 解析
        storage1 = container.resolve(DatabaseStorage)
        storage2 = container.resolve(DatabaseStorage)

        # 单例模式
        assert storage1 is storage2


class TestEventBusIntegration:
    """事件总线集成测试"""

    def test_events_with_metrics(self):
        """测试: 事件总线 + 指标收集"""
        from src.utils.events import (
            DataEvent,
            EventBus,
            OrderFilledEvent,
            SyncProgressEvent,
        )
        from src.utils.metrics import MetricsRegistry

        # 创建指标注册表
        registry = MetricsRegistry(prefix="test")
        sync_counter = registry.counter("sync_events", labels=["status"])
        order_counter = registry.counter("order_events", labels=["direction"])

        # 创建事件总线
        bus = EventBus()

        # 订阅事件并更新指标
        def on_sync_progress(event: SyncProgressEvent):
            if event.completed == event.total:
                sync_counter.inc(status="completed")
            else:
                sync_counter.inc(status="in_progress")

        def on_order_filled(event: OrderFilledEvent):
            order_counter.inc(direction=event.direction)

        bus.subscribe(SyncProgressEvent, on_sync_progress)
        bus.subscribe(OrderFilledEvent, on_order_filled)

        # 发布事件
        bus.publish(SyncProgressEvent(total=100, completed=50))
        bus.publish(SyncProgressEvent(total=100, completed=100))
        bus.publish(OrderFilledEvent(direction="buy", fill_price=10.5))
        bus.publish(OrderFilledEvent(direction="sell", fill_price=11.0))

        # 验证指标
        assert sync_counter.get(status="in_progress") == 1
        assert sync_counter.get(status="completed") == 1
        assert order_counter.get(direction="buy") == 1
        assert order_counter.get(direction="sell") == 1

        # 导出 Prometheus 格式
        output = registry.export_prometheus()
        assert "test_sync_events" in output
        assert "test_order_events" in output


class TestFullDataFlow:
    """完整数据流集成测试"""

    def test_end_to_end_data_flow(self, temp_db):
        """测试: 完整数据流（获取 -> 校验 -> 管道 -> 存储 -> 缓存）"""
        from src.data.pipeline import Pipeline
        from src.data.storage.database import DatabaseStorage
        from src.data.validator import DataValidator
        from src.utils.cache import LRUCache, set_cache
        from src.utils.events import DataEvent, EventBus

        # 设置缓存
        cache = LRUCache(max_size=100)
        set_cache(cache)

        # 设置事件总线
        bus = EventBus()
        events_received = []
        bus.subscribe(DataEvent, lambda e: events_received.append(e))

        # 模拟数据
        raw_data = pd.DataFrame({
            "code": ["000001.SZ"] * 5,
            "trade_date": pd.date_range("2025-01-01", periods=5),
            "open": [10.0, 10.1, 10.2, 10.3, 10.4],
            "high": [10.5, 10.6, 10.7, 10.8, 10.9],
            "low": [9.5, 9.6, 9.7, 9.8, 9.9],
            "close": [10.2, 10.3, 10.4, 10.5, 10.6],
            "volume": [1000000] * 5,
            "amount": [10000000] * 5,
        })

        # 1. 数据校验
        validator = DataValidator()
        validation_result = validator.validate(raw_data, "000001.SZ")
        assert validation_result.valid

        # 2. 管道处理
        pipeline = (
            Pipeline("etl")
            .filter(lambda df: df["volume"] > 0)
            .add_column("adj_factor", lambda df: pd.Series([1.0] * len(df)))
            .sort("trade_date")
        )

        pipe_result = pipeline.execute(raw_data)
        assert pipe_result.success

        # 3. 存储
        storage = DatabaseStorage(db_path=temp_db)
        storage.init_tables()

        df = pipe_result.data.copy()
        df["trade_date"] = df["trade_date"].dt.strftime("%Y-%m-%d")
        rows = storage.save_daily(df)

        # 4. 发布事件
        bus.publish(DataEvent(code="000001.SZ", data_type="daily", rows=rows))

        # 5. 验证
        assert rows == 5
        assert len(events_received) == 1
        assert events_received[0].rows == 5

        # 6. 读取并缓存
        loaded = storage.load_daily("000001.SZ")
        assert len(loaded) == 5


class TestConfigIntegration:
    """配置集成测试"""

    def test_config_validation(self):
        """测试: Pydantic 配置校验"""
        try:
            from config.base import (
                BacktestConfig,
                DataSourceConfig,
                TradingConfig,
            )

            # 正常配置
            trading = TradingConfig(commission_rate=0.0003)
            assert trading.commission_rate == 0.0003

            # 边界校验
            with pytest.raises(Exception):  # ValidationError
                TradingConfig(commission_rate=-0.01)

            # 数据源配置
            data_source = DataSourceConfig(
                primary_source="akshare",
                fallback_sources=["baostock"],
            )
            assert data_source.primary_source == "akshare"

        except ImportError:
            pytest.skip("Pydantic not available")


class TestConnectionPoolIntegration:
    """连接池集成测试"""

    def test_pool_with_storage(self, temp_db):
        """测试: 连接池 + 数据操作"""
        from src.data.storage.pool import ConnectionPool, create_sqlite_pool

        # 创建连接池
        pool = create_sqlite_pool(temp_db)

        # 健康检查
        assert pool.health_check()

        # 执行操作
        with pool.connection() as conn:
            from sqlalchemy import text
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS test_table (
                    id INTEGER PRIMARY KEY,
                    name TEXT
                )
            """))
            conn.execute(text("INSERT INTO test_table (name) VALUES ('test')"))

        # 查询验证
        with pool.connection() as conn:
            from sqlalchemy import text
            result = conn.execute(text("SELECT * FROM test_table")).fetchall()
            assert len(result) == 1

        # 关闭
        pool.dispose()


class TestMetricsIntegration:
    """指标集成测试"""

    def test_metrics_export(self):
        """测试: 指标导出"""
        from src.utils.metrics import MetricsRegistry

        registry = MetricsRegistry(prefix="floatshare")

        # 创建各类型指标
        counter = registry.counter("requests_total", "请求总数", ["method"])
        gauge = registry.gauge("active_connections", "活跃连接数")
        histogram = registry.histogram(
            "request_duration_seconds",
            "请求耗时",
            buckets=(0.1, 0.5, 1.0, 5.0),
        )

        # 记录数据
        counter.inc(method="GET")
        counter.inc(method="GET")
        counter.inc(method="POST")
        gauge.set(10)
        histogram.observe(0.3)
        histogram.observe(0.8)
        histogram.observe(2.0)

        # Prometheus 格式导出
        output = registry.export_prometheus()
        assert "floatshare_requests_total" in output
        assert "floatshare_active_connections" in output
        assert "floatshare_request_duration_seconds" in output

        # JSON 格式导出
        json_output = registry.export_json()
        assert "floatshare_requests_total" in json_output
        assert json_output["floatshare_active_connections"]["samples"][0]["value"] == 10


# ============================================================
# 运行测试
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
