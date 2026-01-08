"""缓存模块

提供多级缓存支持：
- LRU 内存缓存（默认）
- Redis 缓存（可选）
- 缓存装饰器
"""

import hashlib
import json
import logging
import pickle
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, timedelta
from functools import wraps
from threading import Lock
from typing import Any, Callable, Dict, Optional, TypeVar, Union

import pandas as pd

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry:
    """缓存条目"""
    value: Any
    created_at: float
    ttl: Optional[float] = None  # 秒，None 表示永不过期

    @property
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl


class BaseCache(ABC):
    """缓存基类"""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """设置缓存"""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """删除缓存"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """清空缓存"""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """检查 key 是否存在"""
        pass


class LRUCache(BaseCache):
    """LRU 内存缓存"""

    def __init__(self, max_size: int = 1000, default_ttl: Optional[float] = None):
        """
        Args:
            max_size: 最大缓存条目数
            default_ttl: 默认过期时间（秒），None 表示不过期
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # 检查过期
            if entry.is_expired:
                del self._cache[key]
                self._misses += 1
                return None

            # 移到末尾（最近使用）
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        with self._lock:
            # 使用默认 TTL
            if ttl is None:
                ttl = self.default_ttl

            # 如果已存在，先删除
            if key in self._cache:
                del self._cache[key]

            # 检查容量
            while len(self._cache) >= self.max_size:
                # 删除最老的（最少使用的）
                self._cache.popitem(last=False)

            self._cache[key] = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl=ttl,
            )

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def exists(self, key: str) -> bool:
        with self._lock:
            if key not in self._cache:
                return False
            entry = self._cache[key]
            if entry.is_expired:
                del self._cache[key]
                return False
            return True

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> Dict[str, Any]:
        return {
            "size": self.size,
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self.hit_rate:.1%}",
        }


class RedisCache(BaseCache):
    """Redis 缓存（可选依赖）"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "floatshare:",
        default_ttl: Optional[float] = 3600,
    ):
        self.prefix = prefix
        self.default_ttl = default_ttl

        try:
            import redis
            self._client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False,
            )
            # 测试连接
            self._client.ping()
            logger.info(f"Redis 缓存已连接: {host}:{port}")
        except ImportError:
            raise ImportError("请安装 redis: pip install redis")
        except Exception as e:
            raise ConnectionError(f"Redis 连接失败: {e}")

    def _make_key(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        try:
            data = self._client.get(self._make_key(key))
            if data is None:
                return None
            return pickle.loads(data)
        except Exception as e:
            logger.warning(f"Redis get 失败: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        try:
            if ttl is None:
                ttl = self.default_ttl

            data = pickle.dumps(value)
            if ttl:
                self._client.setex(self._make_key(key), int(ttl), data)
            else:
                self._client.set(self._make_key(key), data)
        except Exception as e:
            logger.warning(f"Redis set 失败: {e}")

    def delete(self, key: str) -> bool:
        try:
            return self._client.delete(self._make_key(key)) > 0
        except Exception as e:
            logger.warning(f"Redis delete 失败: {e}")
            return False

    def clear(self) -> None:
        try:
            keys = self._client.keys(f"{self.prefix}*")
            if keys:
                self._client.delete(*keys)
        except Exception as e:
            logger.warning(f"Redis clear 失败: {e}")

    def exists(self, key: str) -> bool:
        try:
            return self._client.exists(self._make_key(key)) > 0
        except Exception as e:
            logger.warning(f"Redis exists 失败: {e}")
            return False


class TieredCache(BaseCache):
    """多级缓存（L1: 内存, L2: Redis）"""

    def __init__(
        self,
        l1_cache: Optional[LRUCache] = None,
        l2_cache: Optional[RedisCache] = None,
    ):
        self.l1 = l1_cache or LRUCache(max_size=500)
        self.l2 = l2_cache  # 可选

    def get(self, key: str) -> Optional[Any]:
        # 先查 L1
        value = self.l1.get(key)
        if value is not None:
            return value

        # 再查 L2
        if self.l2:
            value = self.l2.get(key)
            if value is not None:
                # 回填 L1
                self.l1.set(key, value)
                return value

        return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        self.l1.set(key, value, ttl)
        if self.l2:
            self.l2.set(key, value, ttl)

    def delete(self, key: str) -> bool:
        r1 = self.l1.delete(key)
        r2 = self.l2.delete(key) if self.l2 else False
        return r1 or r2

    def clear(self) -> None:
        self.l1.clear()
        if self.l2:
            self.l2.clear()

    def exists(self, key: str) -> bool:
        return self.l1.exists(key) or (self.l2.exists(key) if self.l2 else False)


# ============================================================
# 全局缓存实例
# ============================================================

_global_cache: Optional[BaseCache] = None


def get_cache() -> BaseCache:
    """获取全局缓存实例"""
    global _global_cache
    if _global_cache is None:
        _global_cache = LRUCache(max_size=1000, default_ttl=3600)
    return _global_cache


def set_cache(cache: BaseCache) -> None:
    """设置全局缓存实例"""
    global _global_cache
    _global_cache = cache


# ============================================================
# 缓存装饰器
# ============================================================


def make_cache_key(*args, **kwargs) -> str:
    """生成缓存 key"""
    key_parts = []

    for arg in args:
        if isinstance(arg, pd.DataFrame):
            key_parts.append(f"df:{len(arg)}")
        elif isinstance(arg, date):
            key_parts.append(arg.isoformat())
        elif hasattr(arg, "__dict__"):
            key_parts.append(str(type(arg).__name__))
        else:
            key_parts.append(str(arg))

    for k, v in sorted(kwargs.items()):
        if isinstance(v, date):
            key_parts.append(f"{k}={v.isoformat()}")
        else:
            key_parts.append(f"{k}={v}")

    key_str = ":".join(key_parts)

    # 如果太长，使用 hash
    if len(key_str) > 200:
        return hashlib.md5(key_str.encode()).hexdigest()

    return key_str


def cached(
    ttl: Optional[float] = 3600,
    key_prefix: str = "",
    cache: Optional[BaseCache] = None,
) -> Callable:
    """
    缓存装饰器

    Args:
        ttl: 缓存过期时间（秒），None 表示不过期
        key_prefix: 缓存 key 前缀
        cache: 缓存实例，默认使用全局缓存

    Example:
        @cached(ttl=3600, key_prefix="daily")
        def get_daily(code: str, start_date: date) -> pd.DataFrame:
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            _cache = cache or get_cache()

            # 生成 key
            func_name = func.__name__
            arg_key = make_cache_key(*args, **kwargs)
            full_key = f"{key_prefix}:{func_name}:{arg_key}" if key_prefix else f"{func_name}:{arg_key}"

            # 尝试获取缓存
            cached_value = _cache.get(full_key)
            if cached_value is not None:
                logger.debug(f"缓存命中: {full_key[:50]}...")
                return cached_value

            # 执行函数
            result = func(*args, **kwargs)

            # 存入缓存（空 DataFrame 不缓存）
            if isinstance(result, pd.DataFrame):
                if not result.empty:
                    _cache.set(full_key, result, ttl)
            elif result is not None:
                _cache.set(full_key, result, ttl)

            return result

        return wrapper
    return decorator


def cache_daily_data(
    code: str,
    start_date: date,
    end_date: date,
    ttl: float = 86400,  # 1天
) -> Callable:
    """
    日线数据专用缓存装饰器

    自动按日期范围分片缓存
    """
    def decorator(func: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> pd.DataFrame:
            _cache = get_cache()

            # 生成 key
            key = f"daily:{code}:{start_date.isoformat()}:{end_date.isoformat()}"

            # 尝试获取缓存
            cached_value = _cache.get(key)
            if cached_value is not None:
                return cached_value

            # 执行函数
            result = func(*args, **kwargs)

            # 存入缓存
            if not result.empty:
                _cache.set(key, result, ttl)

            return result

        return wrapper
    return decorator
