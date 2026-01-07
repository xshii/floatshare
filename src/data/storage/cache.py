"""缓存管理"""

from typing import Optional, Any
from datetime import datetime, timedelta
import json
import hashlib
from pathlib import Path
import pickle


class CacheManager:
    """缓存管理器"""

    def __init__(self, cache_dir: Optional[str] = None, default_ttl: int = 3600):
        """
        初始化缓存管理器

        Args:
            cache_dir: 缓存目录
            default_ttl: 默认过期时间（秒）
        """
        if cache_dir is None:
            cache_dir = str(Path(__file__).parent.parent.parent.parent / "data" / "cache")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        self._memory_cache: dict = {}

    def _get_cache_key(self, key: str) -> str:
        """生成缓存键"""
        return hashlib.md5(key.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.pkl"

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存

        Args:
            key: 缓存键

        Returns:
            缓存值，不存在或过期返回None
        """
        cache_key = self._get_cache_key(key)

        # 先检查内存缓存
        if cache_key in self._memory_cache:
            data, expire_time = self._memory_cache[cache_key]
            if datetime.now() < expire_time:
                return data
            else:
                del self._memory_cache[cache_key]

        # 检查文件缓存
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    data, expire_time = pickle.load(f)

                if datetime.now() < expire_time:
                    # 加载到内存缓存
                    self._memory_cache[cache_key] = (data, expire_time)
                    return data
                else:
                    cache_path.unlink()  # 删除过期文件
            except Exception:
                pass

        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        设置缓存

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒），None使用默认值
        """
        cache_key = self._get_cache_key(key)
        ttl = ttl if ttl is not None else self.default_ttl
        expire_time = datetime.now() + timedelta(seconds=ttl)

        # 保存到内存
        self._memory_cache[cache_key] = (value, expire_time)

        # 保存到文件
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump((value, expire_time), f)
        except Exception:
            pass

    def delete(self, key: str) -> None:
        """删除缓存"""
        cache_key = self._get_cache_key(key)

        if cache_key in self._memory_cache:
            del self._memory_cache[cache_key]

        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            cache_path.unlink()

    def clear(self) -> None:
        """清空所有缓存"""
        self._memory_cache.clear()

        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()

    def clear_expired(self) -> int:
        """清理过期缓存"""
        count = 0

        # 清理内存缓存
        expired_keys = [
            k for k, (_, expire_time) in self._memory_cache.items()
            if datetime.now() >= expire_time
        ]
        for k in expired_keys:
            del self._memory_cache[k]
            count += 1

        # 清理文件缓存
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, "rb") as f:
                    _, expire_time = pickle.load(f)

                if datetime.now() >= expire_time:
                    cache_file.unlink()
                    count += 1
            except Exception:
                cache_file.unlink()
                count += 1

        return count
