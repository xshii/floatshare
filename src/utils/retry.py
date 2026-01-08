"""重试机制

基于 tenacity 提供统一的重试能力：
- 指数退避
- 熔断器
- 自定义重试策略
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional, Set, Tuple, Type, TypeVar, Union

from tenacity import (
    RetryError,
    Retrying,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    wait_fixed,
    wait_random,
    before_sleep_log,
    after_log,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================
# 预定义重试策略
# ============================================================


def retry_network(
    max_attempts: int = 3,
    min_wait: float = 1,
    max_wait: float = 30,
):
    """
    网络请求重试装饰器

    Args:
        max_attempts: 最大重试次数
        min_wait: 最小等待时间(秒)
        max_wait: 最大等待时间(秒)
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


def retry_database(
    max_attempts: int = 3,
    wait_seconds: float = 0.5,
):
    """
    数据库操作重试装饰器

    Args:
        max_attempts: 最大重试次数
        wait_seconds: 固定等待时间(秒)
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_fixed(wait_seconds),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


def retry_with_fallback(
    max_attempts: int = 3,
    fallback_value: Any = None,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    带降级的重试装饰器

    Args:
        max_attempts: 最大重试次数
        fallback_value: 失败时返回的默认值
        exceptions: 需要重试的异常类型
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                for attempt in Retrying(
                    stop=stop_after_attempt(max_attempts),
                    wait=wait_exponential(multiplier=1, min=1, max=10),
                    retry=retry_if_exception_type(exceptions),
                ):
                    with attempt:
                        return func(*args, **kwargs)
            except RetryError:
                logger.error(f"{func.__name__} 重试 {max_attempts} 次后失败，返回降级值")
                return fallback_value
        return wrapper
    return decorator


# ============================================================
# 熔断器
# ============================================================


class CircuitState(Enum):
    """熔断器状态"""
    CLOSED = "closed"      # 正常状态
    OPEN = "open"          # 熔断状态
    HALF_OPEN = "half_open"  # 半开状态


@dataclass
class CircuitBreaker:
    """
    熔断器实现

    状态转换：
    - CLOSED -> OPEN: 失败次数达到阈值
    - OPEN -> HALF_OPEN: 超过重置时间
    - HALF_OPEN -> CLOSED: 成功调用
    - HALF_OPEN -> OPEN: 失败调用
    """

    failure_threshold: int = 5
    reset_timeout: float = 60.0
    half_open_max_calls: int = 3

    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = field(default=0)
    success_count: int = field(default=0)
    last_failure_time: Optional[float] = field(default=None)
    half_open_calls: int = field(default=0)

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """作为装饰器使用"""
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if not self.can_execute():
                raise CircuitBreakerOpen(f"熔断器处于开启状态: {func.__name__}")

            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure()
                raise

        return wrapper

    def can_execute(self) -> bool:
        """检查是否可以执行"""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if self._should_try_reset():
                self._transition_to_half_open()
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls < self.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False

        return False

    def record_success(self) -> None:
        """记录成功"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_max_calls:
                self._transition_to_closed()
        else:
            self.failure_count = 0

    def record_failure(self) -> None:
        """记录失败"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            self._transition_to_open()
        elif self.failure_count >= self.failure_threshold:
            self._transition_to_open()

    def _should_try_reset(self) -> bool:
        """检查是否应该尝试重置"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.reset_timeout

    def _transition_to_open(self) -> None:
        """转换到开启状态"""
        self.state = CircuitState.OPEN
        logger.warning(f"熔断器开启，失败次数: {self.failure_count}")

    def _transition_to_half_open(self) -> None:
        """转换到半开状态"""
        self.state = CircuitState.HALF_OPEN
        self.half_open_calls = 0
        self.success_count = 0
        logger.info("熔断器半开，尝试恢复")

    def _transition_to_closed(self) -> None:
        """转换到关闭状态"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        logger.info("熔断器关闭，恢复正常")

    def reset(self) -> None:
        """手动重置熔断器"""
        self._transition_to_closed()


class CircuitBreakerOpen(Exception):
    """熔断器开启异常"""
    pass


# ============================================================
# 重试管理器
# ============================================================


class RetryManager:
    """
    重试管理器

    管理多个重试策略和熔断器
    """

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}

    def get_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
    ) -> CircuitBreaker:
        """获取或创建熔断器"""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                failure_threshold=failure_threshold,
                reset_timeout=reset_timeout,
            )
        return self._breakers[name]

    def reset_all(self) -> None:
        """重置所有熔断器"""
        for breaker in self._breakers.values():
            breaker.reset()

    def get_status(self) -> dict[str, str]:
        """获取所有熔断器状态"""
        return {
            name: breaker.state.value
            for name, breaker in self._breakers.items()
        }


# 全局重试管理器
_retry_manager: Optional[RetryManager] = None


def get_retry_manager() -> RetryManager:
    """获取全局重试管理器"""
    global _retry_manager
    if _retry_manager is None:
        _retry_manager = RetryManager()
    return _retry_manager
