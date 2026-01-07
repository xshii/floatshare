"""数据同步器测试"""

import json
import pytest
import pandas as pd
from datetime import date, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.data.syncer import (
    DataSyncer,
    SyncPriority,
    SyncStatus,
    SyncTask,
    SyncState,
    RateLimiter,
    SourcePool,
    SourceHealth,
)


class TestRateLimiter:
    """速率限制器测试"""

    def test_basic_rate_limit(self):
        """测试基本速率限制"""
        limiter = RateLimiter(
            requests_per_minute=60,  # 1 request per second
            batch_size=100,
            batch_pause=0,
        )

        import time

        start = time.time()

        for _ in range(3):
            limiter.wait()

        elapsed = time.time() - start

        # 3次请求应该至少花费2秒（第一次立即执行）
        assert elapsed >= 2.0


class TestSyncState:
    """同步状态测试"""

    def test_state_serialization(self, temp_state_path):
        """测试状态序列化"""
        state = SyncState(
            session_id="test_123",
            created_at="2023-01-01T00:00:00",
            updated_at="2023-01-01T00:00:00",
            source="akshare",
            data_type="daily",
            total_tasks=10,
            completed_tasks=5,
        )

        # 保存
        with open(temp_state_path, "w") as f:
            from dataclasses import asdict

            json.dump(asdict(state), f)

        # 加载
        with open(temp_state_path, "r") as f:
            data = json.load(f)

        loaded = SyncState(**data)

        assert loaded.session_id == "test_123"
        assert loaded.total_tasks == 10
        assert loaded.completed_tasks == 5


class TestDataSyncer:
    """数据同步器测试"""

    def test_init(self, temp_db_path, temp_state_path):
        """测试初始化"""
        syncer = DataSyncer(
            source="akshare",
            db_path=temp_db_path,
            state_path=temp_state_path,
        )

        assert syncer.source == "akshare"

    def test_set_rate_limit(self, temp_db_path, temp_state_path):
        """测试设置速率限制"""
        syncer = DataSyncer(
            source="akshare",
            db_path=temp_db_path,
            state_path=temp_state_path,
        )

        syncer.set_rate_limit(
            requests_per_minute=10,
            batch_size=5,
            batch_pause=1.0,
        )

        assert syncer.rate_limiter.requests_per_minute == 10
        assert syncer.rate_limiter.batch_size == 5

    def test_add_market_suffix(self, temp_db_path, temp_state_path):
        """测试市场后缀添加"""
        syncer = DataSyncer(
            source="akshare",
            db_path=temp_db_path,
            state_path=temp_state_path,
        )

        assert syncer._add_market_suffix("600000") == "600000.SH"
        assert syncer._add_market_suffix("000001") == "000001.SZ"
        assert syncer._add_market_suffix("300001") == "300001.SZ"
        assert syncer._add_market_suffix("430001") == "430001.BJ"

    def test_clear_state(self, temp_db_path, temp_state_path):
        """测试清除状态"""
        syncer = DataSyncer(
            source="akshare",
            db_path=temp_db_path,
            state_path=temp_state_path,
        )

        # 创建状态文件
        from pathlib import Path

        Path(temp_state_path).write_text("{}")

        syncer.clear_state()

        assert not Path(temp_state_path).exists()

    def test_get_sync_progress_empty(self, temp_db_path, temp_state_path):
        """测试获取进度（无状态文件）"""
        syncer = DataSyncer(
            source="akshare",
            db_path=temp_db_path,
            state_path=temp_state_path,
        )

        progress = syncer.get_sync_progress()
        assert progress is None

    def test_get_sync_progress(self, temp_db_path, temp_state_path):
        """测试获取进度"""
        # 创建状态文件
        state_data = {
            "session_id": "test_123",
            "created_at": "2023-01-01T00:00:00",
            "updated_at": "2023-01-01T12:00:00",
            "source": "akshare",
            "data_type": "daily",
            "total_tasks": 100,
            "completed_tasks": 50,
            "failed_tasks": 5,
            "current_priority": "hs300",
            "tasks": {},
        }

        from pathlib import Path

        Path(temp_state_path).write_text(json.dumps(state_data))

        syncer = DataSyncer(
            source="akshare",
            db_path=temp_db_path,
            state_path=temp_state_path,
        )

        progress = syncer.get_sync_progress()

        assert progress is not None
        assert progress["total"] == 100
        assert progress["completed"] == 50
        assert progress["failed"] == 5
        assert progress["pending"] == 45


class TestDataSyncerWithMock:
    """使用Mock的同步器测试"""

    @patch("src.data.syncer.DataLoader")
    def test_sync_single_stock(self, mock_loader_class, temp_db_path, temp_state_path):
        """测试同步单个股票"""
        # 准备mock数据
        mock_df = pd.DataFrame({
            "code": ["000001.SZ"] * 5,
            "trade_date": pd.date_range("2023-01-01", periods=5),
            "open": [10.0, 10.1, 10.2, 10.3, 10.4],
            "high": [10.5, 10.6, 10.7, 10.8, 10.9],
            "low": [9.5, 9.6, 9.7, 9.8, 9.9],
            "close": [10.2, 10.3, 10.4, 10.5, 10.6],
            "volume": [1000000] * 5,
            "amount": [10000000] * 5,
            "adj_factor": [1.0] * 5,
        })

        mock_loader = MagicMock()
        mock_loader.get_daily.return_value = mock_df
        mock_loader_class.return_value = mock_loader

        syncer = DataSyncer(
            source="akshare",
            db_path=temp_db_path,
            state_path=temp_state_path,
        )
        syncer.loader = mock_loader

        # 初始化表
        syncer._init_daily_table()

        # 同步
        rows = syncer._sync_single_stock(
            code="000001.SZ",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 10),
        )

        assert rows == 5

    @patch("src.data.syncer.DataLoader")
    def test_incremental_sync(self, mock_loader_class, temp_db_path, temp_state_path):
        """测试增量同步"""
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader

        syncer = DataSyncer(
            source="akshare",
            db_path=temp_db_path,
            state_path=temp_state_path,
        )
        syncer.loader = mock_loader

        # 初始化表并插入一些数据
        syncer._init_daily_table()

        from sqlalchemy import text

        with syncer.storage.engine.connect() as conn:
            conn.execute(
                text(
                    """
                INSERT INTO stock_daily (code, trade_date, open, high, low, close, volume, amount, adj_factor)
                VALUES ('000001.SZ', '2023-01-05', 10.0, 10.5, 9.5, 10.2, 1000000, 10000000, 1.0)
                """
                )
            )
            conn.commit()

        # 模拟返回新数据
        mock_df = pd.DataFrame({
            "code": ["000001.SZ"] * 3,
            "trade_date": pd.date_range("2023-01-06", periods=3),
            "open": [10.3, 10.4, 10.5],
            "high": [10.8, 10.9, 11.0],
            "low": [10.0, 10.1, 10.2],
            "close": [10.5, 10.6, 10.7],
            "volume": [1000000] * 3,
            "amount": [10000000] * 3,
            "adj_factor": [1.0] * 3,
        })
        mock_loader.get_daily.return_value = mock_df

        # 同步应该从 2023-01-06 开始
        rows = syncer._sync_single_stock(
            code="000001.SZ",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 10),
        )

        assert rows == 3

        # 验证调用参数
        call_args = mock_loader.get_daily.call_args
        assert call_args[1]["start_date"] == date(2023, 1, 6)


class TestSyncPriority:
    """同步优先级测试"""

    def test_priority_values(self):
        """测试优先级值"""
        assert SyncPriority.HS300.value == "hs300"
        assert SyncPriority.ZZ500.value == "zz500"
        assert SyncPriority.ALL.value == "all"


class TestSyncTask:
    """同步任务测试"""

    def test_task_creation(self):
        """测试任务创建"""
        task = SyncTask(
            code="000001.SZ",
            name="平安银行",
            priority=SyncPriority.HS300,
        )

        assert task.code == "000001.SZ"
        assert task.status == SyncStatus.PENDING
        assert task.retry_count == 0

    def test_task_to_dict(self):
        """测试任务序列化"""
        from dataclasses import asdict

        task = SyncTask(
            code="000001.SZ",
            name="平安银行",
            priority=SyncPriority.HS300,
            status=SyncStatus.COMPLETED,
            synced_rows=100,
        )

        d = asdict(task)

        assert d["code"] == "000001.SZ"
        assert d["synced_rows"] == 100


class TestSourceHealth:
    """数据源健康状态测试"""

    def test_initial_state(self):
        """测试初始状态"""
        health = SourceHealth(name="akshare")

        assert health.success_rate == 1.0
        assert health.is_available
        assert health.consecutive_fails == 0

    def test_record_success(self):
        """测试记录成功"""
        health = SourceHealth(name="akshare")

        health.record_success(response_time=0.5)
        health.record_success(response_time=0.3)

        assert health.success_count == 2
        assert health.fail_count == 0
        assert health.success_rate == 1.0
        assert health.avg_response_time > 0

    def test_record_failure_with_backoff(self):
        """测试失败后的指数退避"""
        health = SourceHealth(name="akshare")

        # 连续失败3次后应该被禁用
        for _ in range(3):
            health.record_failure(backoff_base=1.0)  # 使用1秒的基数便于测试

        assert health.consecutive_fails == 3
        assert health.disabled_until is not None
        assert not health.is_available

    def test_success_resets_consecutive_fails(self):
        """测试成功重置连续失败计数"""
        health = SourceHealth(name="akshare")

        health.record_failure()
        health.record_failure()
        health.record_success(0.5)

        assert health.consecutive_fails == 0


class TestSourcePool:
    """数据源池测试"""

    def test_init_with_sources(self):
        """测试初始化多数据源"""
        # 注意：这会尝试初始化真实的 DataLoader，可能会失败
        # 在实际测试中应该 mock DataLoader
        pass

    @patch("src.data.syncer.DataLoader")
    def test_get_available_sources(self, mock_loader_class):
        """测试获取可用数据源"""
        mock_loader_class.return_value = MagicMock()

        pool = SourcePool(sources=["akshare", "eastmoney"], parallel=False)

        available = pool.get_available_sources()
        assert len(available) == 2

    @patch("src.data.syncer.DataLoader")
    def test_sequential_fetch_fallback(self, mock_loader_class):
        """测试顺序获取的降级机制"""
        # 设置第一个数据源失败，第二个成功
        mock_loader1 = MagicMock()
        mock_loader1.get_daily.side_effect = Exception("网络错误")

        mock_loader2 = MagicMock()
        mock_loader2.get_daily.return_value = pd.DataFrame({
            "code": ["000001.SZ"],
            "trade_date": [pd.Timestamp("2023-01-01")],
            "open": [10.0],
            "high": [10.5],
            "low": [9.5],
            "close": [10.2],
            "volume": [1000000],
            "amount": [10000000],
        })

        mock_loader_class.side_effect = [mock_loader1, mock_loader2]

        pool = SourcePool(sources=["source1", "source2"], parallel=False)
        pool.loaders = {"source1": mock_loader1, "source2": mock_loader2}

        df, source = pool.fetch_daily(
            code="000001.SZ",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 10),
        )

        assert not df.empty
        assert source == "source2"
        assert pool.health["source1"].fail_count == 1

    def test_health_report(self):
        """测试健康报告"""
        health = SourceHealth(name="test")
        health.record_success(0.5)
        health.record_success(0.3)
        health.record_failure()

        pool = SourcePool.__new__(SourcePool)
        pool.health = {"test": health}
        pool.loaders = {}

        report = pool.get_health_report()

        assert "test" in report
        assert "success_rate" in report["test"]
        assert "avg_response_time" in report["test"]


class TestDataSyncerMultiSource:
    """多数据源同步器测试"""

    def test_init_with_sources(self, temp_db_path, temp_state_path):
        """测试多数据源初始化"""
        with patch("src.data.syncer.DataLoader"):
            syncer = DataSyncer(
                sources=["akshare", "eastmoney"],
                parallel=False,
                db_path=temp_db_path,
                state_path=temp_state_path,
            )

            assert syncer.multi_source
            assert syncer.source_pool is not None

    def test_init_with_single_source(self, temp_db_path, temp_state_path):
        """测试单数据源初始化（向后兼容）"""
        with patch("src.data.syncer.DataLoader"):
            syncer = DataSyncer(
                source="akshare",
                db_path=temp_db_path,
                state_path=temp_state_path,
            )

            assert not syncer.multi_source
            assert syncer.source_pool is None

    def test_set_parallel(self, temp_db_path, temp_state_path):
        """测试设置并行模式"""
        with patch("src.data.syncer.DataLoader"):
            syncer = DataSyncer(
                sources=["akshare", "eastmoney"],
                parallel=False,
                db_path=temp_db_path,
                state_path=temp_state_path,
            )

            syncer.set_parallel(True)
            assert syncer.source_pool.parallel

    def test_get_source_health(self, temp_db_path, temp_state_path):
        """测试获取数据源健康状态"""
        with patch("src.data.syncer.DataLoader"):
            syncer = DataSyncer(
                sources=["akshare", "eastmoney"],
                db_path=temp_db_path,
                state_path=temp_state_path,
            )

            health = syncer.get_source_health()
            assert health is not None
            assert "akshare" in health or "eastmoney" in health
