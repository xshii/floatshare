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
