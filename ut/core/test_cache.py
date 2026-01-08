"""缓存模块测试"""

import time
from datetime import date

import pandas as pd
import pytest

from src.utils.cache import (
    BaseCache,
    CacheEntry,
    LRUCache,
    TieredCache,
    cached,
    get_cache,
    make_cache_key,
    set_cache,
)


class TestCacheEntry:
    """CacheEntry 测试"""

    def test_not_expired_without_ttl(self):
        entry = CacheEntry(value="test", created_at=time.time(), ttl=None)
        assert not entry.is_expired

    def test_not_expired_within_ttl(self):
        entry = CacheEntry(value="test", created_at=time.time(), ttl=10)
        assert not entry.is_expired

    def test_expired_after_ttl(self):
        entry = CacheEntry(value="test", created_at=time.time() - 5, ttl=1)
        assert entry.is_expired


class TestLRUCache:
    """LRUCache 测试"""

    def test_basic_set_get(self):
        cache = LRUCache(max_size=10)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_nonexistent_key(self):
        cache = LRUCache(max_size=10)
        assert cache.get("nonexistent") is None

    def test_delete(self):
        cache = LRUCache(max_size=10)
        cache.set("key1", "value1")
        assert cache.delete("key1")
        assert cache.get("key1") is None

    def test_delete_nonexistent(self):
        cache = LRUCache(max_size=10)
        assert not cache.delete("nonexistent")

    def test_exists(self):
        cache = LRUCache(max_size=10)
        cache.set("key1", "value1")
        assert cache.exists("key1")
        assert not cache.exists("key2")

    def test_clear(self):
        cache = LRUCache(max_size=10)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        assert cache.size == 0
        assert cache.get("key1") is None

    def test_max_size_eviction(self):
        cache = LRUCache(max_size=3)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")  # 应该驱逐 key1

        assert cache.size == 3
        assert cache.get("key1") is None  # 被驱逐
        assert cache.get("key4") == "value4"

    def test_lru_ordering(self):
        cache = LRUCache(max_size=3)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # 访问 key1，使其成为最近使用
        cache.get("key1")

        # 添加新元素，应该驱逐 key2（最少使用）
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1"  # 仍然存在
        assert cache.get("key2") is None  # 被驱逐
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_ttl_expiration(self):
        cache = LRUCache(max_size=10, default_ttl=0.1)  # 100ms TTL
        cache.set("key1", "value1")

        assert cache.get("key1") == "value1"
        time.sleep(0.15)
        assert cache.get("key1") is None  # 过期

    def test_custom_ttl(self):
        cache = LRUCache(max_size=10)
        cache.set("key1", "value1", ttl=0.1)
        cache.set("key2", "value2", ttl=1.0)

        time.sleep(0.15)
        assert cache.get("key1") is None  # 过期
        assert cache.get("key2") == "value2"  # 未过期

    def test_hit_rate(self):
        cache = LRUCache(max_size=10)
        cache.set("key1", "value1")

        cache.get("key1")  # hit
        cache.get("key1")  # hit
        cache.get("key2")  # miss

        assert cache._hits == 2
        assert cache._misses == 1
        assert cache.hit_rate == 2 / 3

    def test_stats(self):
        cache = LRUCache(max_size=10)
        cache.set("key1", "value1")
        cache.get("key1")

        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["max_size"] == 10
        assert stats["hits"] == 1
        assert stats["misses"] == 0

    def test_update_existing_key(self):
        cache = LRUCache(max_size=10)
        cache.set("key1", "value1")
        cache.set("key1", "value2")

        assert cache.get("key1") == "value2"
        assert cache.size == 1

    def test_complex_values(self):
        cache = LRUCache(max_size=10)

        # 字典
        cache.set("dict", {"a": 1, "b": 2})
        assert cache.get("dict") == {"a": 1, "b": 2}

        # 列表
        cache.set("list", [1, 2, 3])
        assert cache.get("list") == [1, 2, 3]

        # DataFrame
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        cache.set("df", df)
        pd.testing.assert_frame_equal(cache.get("df"), df)


class TestTieredCache:
    """TieredCache 测试"""

    def test_l1_only(self):
        l1 = LRUCache(max_size=10)
        cache = TieredCache(l1_cache=l1)

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_l1_hit(self):
        l1 = LRUCache(max_size=10)
        cache = TieredCache(l1_cache=l1)

        cache.set("key1", "value1")
        assert l1.get("key1") == "value1"

    def test_delete(self):
        cache = TieredCache()
        cache.set("key1", "value1")
        cache.delete("key1")
        assert cache.get("key1") is None

    def test_clear(self):
        cache = TieredCache()
        cache.set("key1", "value1")
        cache.clear()
        assert cache.get("key1") is None


class TestCacheDecorator:
    """@cached 装饰器测试"""

    def test_basic_caching(self):
        call_count = 0
        cache = LRUCache(max_size=10)
        set_cache(cache)

        @cached(ttl=60)
        def expensive_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = expensive_function(5)
        result2 = expensive_function(5)

        assert result1 == 10
        assert result2 == 10
        assert call_count == 1  # 只调用一次

    def test_different_args(self):
        call_count = 0
        cache = LRUCache(max_size=10)
        set_cache(cache)

        @cached(ttl=60)
        def func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        func(5)
        func(10)

        assert call_count == 2  # 不同参数，调用两次

    def test_with_key_prefix(self):
        cache = LRUCache(max_size=10)
        set_cache(cache)

        @cached(ttl=60, key_prefix="myprefix")
        def func(x: int) -> int:
            return x

        func(5)

        # 检查 key 包含前缀
        keys = list(cache._cache.keys())
        assert any("myprefix" in k for k in keys)

    def test_empty_dataframe_not_cached(self):
        cache = LRUCache(max_size=10)
        set_cache(cache)

        @cached(ttl=60)
        def func() -> pd.DataFrame:
            return pd.DataFrame()

        func()
        assert cache.size == 0  # 空 DataFrame 不缓存


class TestMakeCacheKey:
    """make_cache_key 测试"""

    def test_simple_args(self):
        key = make_cache_key("a", 1, 2.0)
        assert "a" in key
        assert "1" in key

    def test_date_args(self):
        d = date(2025, 1, 1)
        key = make_cache_key(d)
        assert "2025-01-01" in key

    def test_kwargs(self):
        key = make_cache_key(start=date(2025, 1, 1), code="000001")
        assert "2025-01-01" in key
        assert "000001" in key

    def test_long_key_hashed(self):
        # 创建超长参数
        long_args = ["x" * 50 for _ in range(10)]
        key = make_cache_key(*long_args)
        assert len(key) <= 32  # MD5 hash 长度


class TestGlobalCache:
    """全局缓存测试"""

    def test_get_cache_returns_instance(self):
        cache = get_cache()
        assert isinstance(cache, BaseCache)

    def test_set_cache(self):
        new_cache = LRUCache(max_size=100)
        set_cache(new_cache)
        assert get_cache() is new_cache
