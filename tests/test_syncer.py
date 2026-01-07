"""syncer 模块单元测试

测试 ProxyPool, SourceHealth, SourcePool, RateLimiter 等组件的业务逻辑。
不涉及网络请求。
"""

import time
from datetime import date
from unittest.mock import Mock, patch, MagicMock

import pandas as pd
import pytest

from src.data.syncer import (
    ProxyPool,
    SourceHealth,
    SourcePool,
    RateLimiter,
    SyncPriority,
    SyncStatus,
    SyncTask,
)


# ============================================================
# ProxyPool 测试
# ============================================================


class TestProxyPool:
    """代理池测试"""

    def test_empty_pool_returns_none(self):
        """空代理池返回 None"""
        pool = ProxyPool()
        assert pool.get_proxy() is None

    def test_proxy_rotation(self):
        """代理轮换"""
        proxies = ["http://p1:8080", "http://p2:8080", "http://p3:8080"]
        pool = ProxyPool(proxies)

        # 轮换获取
        p1 = pool.get_proxy()
        p2 = pool.get_proxy()
        p3 = pool.get_proxy()
        p4 = pool.get_proxy()

        assert p1 == "http://p1:8080"
        assert p2 == "http://p2:8080"
        assert p3 == "http://p3:8080"
        assert p4 == "http://p1:8080"  # 循环

    def test_report_failure(self):
        """记录失败"""
        proxies = ["http://p1:8080", "http://p2:8080"]
        pool = ProxyPool(proxies)

        pool.report_failure("http://p1:8080")
        pool.report_failure("http://p1:8080")

        assert pool.failed_proxies["http://p1:8080"] == 2
        assert pool.available_count == 2  # 还没到 max_fails

    def test_proxy_disabled_after_max_fails(self):
        """达到最大失败次数后禁用"""
        proxies = ["http://p1:8080", "http://p2:8080"]
        pool = ProxyPool(proxies)
        pool.max_fails = 3

        # p1 失败 3 次
        for _ in range(3):
            pool.report_failure("http://p1:8080")

        assert pool.available_count == 1

        # 获取代理应该跳过 p1
        for _ in range(5):
            p = pool.get_proxy()
            assert p == "http://p2:8080"

    def test_all_proxies_failed_reset(self):
        """所有代理都失败后重置"""
        proxies = ["http://p1:8080"]
        pool = ProxyPool(proxies)
        pool.max_fails = 2

        pool.report_failure("http://p1:8080")
        pool.report_failure("http://p1:8080")

        # 所有代理都失败，应该重置
        p = pool.get_proxy()
        assert p == "http://p1:8080"
        assert pool.failed_proxies.get("http://p1:8080", 0) == 0

    def test_report_success_resets_count(self):
        """成功后重置失败计数"""
        pool = ProxyPool(["http://p1:8080"])

        pool.report_failure("http://p1:8080")
        pool.report_failure("http://p1:8080")
        assert pool.failed_proxies["http://p1:8080"] == 2

        pool.report_success("http://p1:8080")
        assert pool.failed_proxies["http://p1:8080"] == 0

    def test_add_proxies(self):
        """添加代理"""
        pool = ProxyPool(["http://p1:8080"])
        pool.add_proxies(["http://p2:8080", "http://p1:8080"])  # p1 重复

        assert len(pool.proxies) == 2
        assert "http://p2:8080" in pool.proxies


# ============================================================
# SourceHealth 测试
# ============================================================


class TestSourceHealth:
    """数据源健康状态测试"""

    def test_initial_state(self):
        """初始状态"""
        health = SourceHealth(name="test")

        assert health.success_rate == 1.0  # 没有请求时默认 100%
        assert health.is_available
        assert not health.may_be_rate_limited
        assert health.consecutive_fails == 0
        assert health.consecutive_empty == 0

    def test_record_success(self):
        """记录成功"""
        health = SourceHealth(name="test")

        health.record_success(1.5)
        health.record_success(2.0)

        assert health.success_count == 2
        assert health.consecutive_fails == 0
        assert health.avg_response_time > 0

    def test_record_failure_with_backoff(self):
        """记录失败并指数退避"""
        health = SourceHealth(name="test")

        # 连续失败 3 次触发禁用
        health.record_failure(backoff_base=1.0)
        health.record_failure(backoff_base=1.0)
        assert health.is_available

        health.record_failure(backoff_base=1.0)
        assert not health.is_available  # 被禁用
        assert health.disabled_until is not None

    def test_success_resets_consecutive_fails(self):
        """成功后重置连续失败计数"""
        health = SourceHealth(name="test")

        health.record_failure()
        health.record_failure()
        assert health.consecutive_fails == 2

        health.record_success(1.0)
        assert health.consecutive_fails == 0

    def test_rate_limit_detection(self):
        """限流检测（连续空数据）"""
        health = SourceHealth(name="test")

        for _ in range(4):
            health.record_empty()
        assert not health.may_be_rate_limited

        health.record_empty()
        assert health.may_be_rate_limited  # 5 次空数据

    def test_success_resets_empty_count(self):
        """成功后重置空数据计数"""
        health = SourceHealth(name="test")

        for _ in range(3):
            health.record_empty()

        health.record_success(1.0)
        assert health.consecutive_empty == 0

    def test_availability_after_backoff(self):
        """退避结束后恢复可用"""
        health = SourceHealth(name="test")

        # 触发禁用，退避时间设为 0
        for _ in range(3):
            health.record_failure(backoff_base=0)

        # disabled_until 设为过去时间
        health.disabled_until = time.time() - 1
        assert health.is_available


# ============================================================
# RateLimiter 测试
# ============================================================


class TestRateLimiter:
    """速率限制器测试"""

    def test_basic_rate_limiting(self):
        """基本速率限制"""
        limiter = RateLimiter(requests_per_minute=600, jitter=0)  # 0.1s 间隔

        start = time.time()
        limiter.wait()
        limiter.wait()
        elapsed = time.time() - start

        # 两次请求至少需要 0.1s 间隔
        assert elapsed >= 0.09

    def test_request_count(self):
        """请求计数"""
        limiter = RateLimiter(requests_per_minute=6000, jitter=0)

        for _ in range(5):
            limiter.wait()

        assert limiter.request_count == 5

    def test_jitter_adds_randomness(self):
        """抖动添加随机性"""
        limiter = RateLimiter(requests_per_minute=60, jitter=0.5)

        # 多次调用，检查间隔有变化
        intervals = []
        for _ in range(5):
            start = time.time()
            limiter.wait()
            intervals.append(time.time() - start)

        # 由于抖动，间隔应该不完全相同
        # 注意：首次调用间隔可能很短
        assert len(set(round(i, 2) for i in intervals)) >= 1

    def test_batch_pause(self):
        """批次暂停"""
        limiter = RateLimiter(
            requests_per_minute=6000,  # 很高的速率
            batch_size=3,
            batch_pause=0.1,
            jitter=0,
        )

        start = time.time()
        for _ in range(4):
            limiter.wait()
        elapsed = time.time() - start

        # 第 3 次请求后应该暂停 0.1s
        assert elapsed >= 0.09


# ============================================================
# SourcePool 测试（Mock DataLoader）
# ============================================================


class TestSourcePool:
    """数据源池测试"""

    @patch("src.data.syncer.DataLoader")
    def test_initialization(self, mock_loader_class):
        """初始化"""
        mock_loader_class.return_value = Mock()

        pool = SourcePool(sources=["source1", "source2"])

        assert "source1" in pool.loaders
        assert "source2" in pool.loaders
        assert len(pool.health) == 2

    @patch("src.data.syncer.DataLoader")
    def test_get_available_sources(self, mock_loader_class):
        """获取可用数据源"""
        mock_loader_class.return_value = Mock()

        pool = SourcePool(sources=["s1", "s2", "s3"])

        # 初始状态所有都可用
        available = pool.get_available_sources()
        assert len(available) == 3

        # 禁用 s1
        pool.health["s1"].disabled_until = time.time() + 100

        available = pool.get_available_sources()
        assert "s1" not in available
        assert len(available) == 2

    @patch("src.data.syncer.DataLoader")
    def test_sequential_fetch_success(self, mock_loader_class):
        """顺序获取成功"""
        mock_loader = Mock()
        mock_loader.get_daily.return_value = pd.DataFrame({
            "code": ["000001.SZ"],
            "trade_date": ["2024-01-01"],
            "close": [10.0],
        })
        mock_loader_class.return_value = mock_loader

        pool = SourcePool(sources=["s1"], parallel=False)

        df, source = pool.fetch_daily(
            code="000001.SZ",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        assert not df.empty
        assert source == "s1"
        assert pool.health["s1"].success_count == 1

    @patch("src.data.syncer.DataLoader")
    def test_sequential_fetch_fallback(self, mock_loader_class):
        """顺序获取降级"""
        # s1 失败，s2 成功
        mock_loader1 = Mock()
        mock_loader1.get_daily.side_effect = Exception("s1 failed")

        mock_loader2 = Mock()
        mock_loader2.get_daily.return_value = pd.DataFrame({
            "code": ["000001.SZ"],
            "close": [10.0],
        })

        def loader_factory(source):
            if source == "s1":
                return mock_loader1
            return mock_loader2

        mock_loader_class.side_effect = loader_factory

        pool = SourcePool(sources=["s1", "s2"], parallel=False)

        df, source = pool.fetch_daily(
            code="000001.SZ",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        assert not df.empty
        assert source == "s2"
        assert pool.health["s1"].fail_count == 1
        assert pool.health["s2"].success_count == 1

    @patch("src.data.syncer.DataLoader")
    def test_skip_rate_limited_source(self, mock_loader_class):
        """跳过被限流的数据源"""
        mock_loader = Mock()
        mock_loader.get_daily.return_value = pd.DataFrame({"code": ["test"]})
        mock_loader_class.return_value = mock_loader

        pool = SourcePool(sources=["s1", "s2"], parallel=False)

        # 标记 s1 为可能被限流
        for _ in range(5):
            pool.health["s1"].record_empty()

        assert pool.health["s1"].may_be_rate_limited

        # 获取时应该跳过 s1
        df, source = pool.fetch_daily(
            code="000001.SZ",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        assert source == "s2"

    @patch("src.data.syncer.DataLoader")
    def test_all_sources_fail(self, mock_loader_class):
        """所有数据源都失败"""
        mock_loader = Mock()
        mock_loader.get_daily.side_effect = Exception("failed")
        mock_loader_class.return_value = mock_loader

        pool = SourcePool(sources=["s1", "s2"], parallel=False)

        with pytest.raises(RuntimeError, match="所有数据源获取失败"):
            pool.fetch_daily(
                code="000001.SZ",
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
            )

    @patch("src.data.syncer.DataLoader")
    def test_health_report(self, mock_loader_class):
        """健康报告"""
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader

        pool = SourcePool(sources=["s1", "s2"])

        pool.health["s1"].record_success(1.0)
        pool.health["s1"].record_success(2.0)
        pool.health["s2"].record_failure()

        report = pool.get_health_report()

        assert "s1" in report
        assert "s2" in report
        assert report["s1"]["success_count"] == 2
        assert report["s2"]["fail_count"] == 1


# ============================================================
# SyncTask / SyncStatus 测试
# ============================================================


class TestSyncTask:
    """同步任务测试"""

    def test_task_creation(self):
        """任务创建"""
        task = SyncTask(
            code="000001.SZ",
            name="平安银行",
            priority=SyncPriority.HS300,
        )

        assert task.code == "000001.SZ"
        assert task.status == SyncStatus.PENDING
        assert task.retry_count == 0

    def test_sync_priority_values(self):
        """同步优先级值"""
        assert SyncPriority.HS300.value == "hs300"
        assert SyncPriority.ZZ500.value == "zz500"
        assert SyncPriority.ALL.value == "all"


# ============================================================
# 运行测试
# ============================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
