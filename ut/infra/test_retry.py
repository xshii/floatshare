"""重试机制测试"""

import time
from unittest.mock import Mock

import pytest

from src.utils.retry import (
    CircuitBreaker,
    CircuitBreakerOpen,
    CircuitState,
    RetryManager,
    get_retry_manager,
    retry_network,
    retry_with_fallback,
)


class TestRetryDecorators:
    """重试装饰器测试"""

    def test_retry_network_success(self):
        """网络请求成功"""
        call_count = 0

        @retry_network(max_attempts=3)
        def fetch():
            nonlocal call_count
            call_count += 1
            return "success"

        result = fetch()
        assert result == "success"
        assert call_count == 1

    def test_retry_network_retry_on_failure(self):
        """网络请求失败重试"""
        call_count = 0

        @retry_network(max_attempts=3, min_wait=0.01, max_wait=0.1)
        def fetch():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("连接失败")
            return "success"

        result = fetch()
        assert result == "success"
        assert call_count == 3

    def test_retry_network_max_attempts_exceeded(self):
        """超过最大重试次数"""
        @retry_network(max_attempts=2, min_wait=0.01, max_wait=0.1)
        def fetch():
            raise ConnectionError("连接失败")

        with pytest.raises(ConnectionError):
            fetch()

    def test_retry_with_fallback_success(self):
        """带降级的重试成功"""
        @retry_with_fallback(max_attempts=3, fallback_value="default")
        def fetch():
            return "success"

        result = fetch()
        assert result == "success"

    def test_retry_with_fallback_returns_default(self):
        """带降级的重试返回默认值"""
        @retry_with_fallback(max_attempts=2, fallback_value="default")
        def fetch():
            raise Exception("失败")

        result = fetch()
        assert result == "default"


class TestCircuitBreaker:
    """熔断器测试"""

    def test_initial_state(self):
        """初始状态"""
        breaker = CircuitBreaker()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_success_keeps_closed(self):
        """成功保持关闭状态"""
        breaker = CircuitBreaker()
        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_failures_open_breaker(self):
        """失败次数达到阈值开启熔断器"""
        breaker = CircuitBreaker(failure_threshold=3)

        for _ in range(3):
            breaker.record_failure()

        assert breaker.state == CircuitState.OPEN
        assert not breaker.can_execute()

    def test_success_resets_failure_count(self):
        """成功重置失败计数"""
        breaker = CircuitBreaker(failure_threshold=5)

        breaker.record_failure()
        breaker.record_failure()
        assert breaker.failure_count == 2

        breaker.record_success()
        assert breaker.failure_count == 0

    def test_open_to_half_open(self):
        """开启状态到半开状态"""
        breaker = CircuitBreaker(failure_threshold=2, reset_timeout=0.1)

        # 触发熔断
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        # 等待超时
        time.sleep(0.15)

        # 应该可以执行（转为半开）
        assert breaker.can_execute()
        assert breaker.state == CircuitState.HALF_OPEN

    def test_half_open_to_closed(self):
        """半开状态到关闭状态"""
        breaker = CircuitBreaker(
            failure_threshold=2,
            reset_timeout=0,
            half_open_max_calls=2,
        )

        # 触发熔断
        breaker.record_failure()
        breaker.record_failure()

        # 转为半开
        breaker.can_execute()
        assert breaker.state == CircuitState.HALF_OPEN

        # 成功调用
        breaker.record_success()
        breaker.record_success()

        assert breaker.state == CircuitState.CLOSED

    def test_half_open_to_open(self):
        """半开状态失败回到开启状态"""
        breaker = CircuitBreaker(
            failure_threshold=2,
            reset_timeout=0,
        )

        # 触发熔断
        breaker.record_failure()
        breaker.record_failure()

        # 转为半开
        breaker.can_execute()
        assert breaker.state == CircuitState.HALF_OPEN

        # 失败
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

    def test_decorator_usage(self):
        """装饰器用法"""
        breaker = CircuitBreaker(failure_threshold=2)

        @breaker
        def risky_function():
            return "success"

        result = risky_function()
        assert result == "success"

    def test_decorator_raises_when_open(self):
        """开启时装饰器抛出异常"""
        breaker = CircuitBreaker(failure_threshold=2)

        @breaker
        def risky_function():
            raise ValueError("error")

        # 触发熔断
        with pytest.raises(ValueError):
            risky_function()
        with pytest.raises(ValueError):
            risky_function()

        # 熔断器开启
        with pytest.raises(CircuitBreakerOpen):
            risky_function()

    def test_reset(self):
        """手动重置"""
        breaker = CircuitBreaker(failure_threshold=2)

        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0


class TestRetryManager:
    """重试管理器测试"""

    def test_get_breaker(self):
        """获取熔断器"""
        manager = RetryManager()
        breaker = manager.get_breaker("test")

        assert breaker is not None
        assert breaker.state == CircuitState.CLOSED

    def test_get_same_breaker(self):
        """获取相同名称的熔断器"""
        manager = RetryManager()
        breaker1 = manager.get_breaker("test")
        breaker2 = manager.get_breaker("test")

        assert breaker1 is breaker2

    def test_reset_all(self):
        """重置所有熔断器"""
        manager = RetryManager()
        breaker1 = manager.get_breaker("test1", failure_threshold=1)
        breaker2 = manager.get_breaker("test2", failure_threshold=1)

        breaker1.record_failure()
        breaker2.record_failure()

        assert breaker1.state == CircuitState.OPEN
        assert breaker2.state == CircuitState.OPEN

        manager.reset_all()

        assert breaker1.state == CircuitState.CLOSED
        assert breaker2.state == CircuitState.CLOSED

    def test_get_status(self):
        """获取状态"""
        manager = RetryManager()
        manager.get_breaker("test1")
        breaker2 = manager.get_breaker("test2", failure_threshold=1)
        breaker2.record_failure()

        status = manager.get_status()

        assert status["test1"] == "closed"
        assert status["test2"] == "open"

    def test_global_manager(self):
        """全局管理器"""
        manager = get_retry_manager()
        assert manager is not None
        assert isinstance(manager, RetryManager)
