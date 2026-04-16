"""本地两级缓存 — 内存 + pickle 文件。"""

from __future__ import annotations

import hashlib
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


class CacheManager:
    """简单两级缓存：内存命中优先，pickle 文件兜底，均带 TTL。"""

    def __init__(
        self,
        cache_dir: str | Path = "data/cache",
        default_ttl: int = 3600,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        self._memory: dict[str, tuple[Any, datetime]] = {}

    @staticmethod
    def _key(raw: str) -> str:
        return hashlib.md5(raw.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl"

    def get(self, key: str) -> Any | None:
        cache_key = self._key(key)
        now = datetime.now()

        if cache_key in self._memory:
            data, expire = self._memory[cache_key]
            if now < expire:
                return data
            del self._memory[cache_key]

        path = self._path(cache_key)
        if not path.exists():
            return None
        try:
            with path.open("rb") as f:
                data, expire = pickle.load(f)
        except Exception:
            return None
        if now >= expire:
            path.unlink(missing_ok=True)
            return None
        self._memory[cache_key] = (data, expire)
        return data

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        cache_key = self._key(key)
        ttl = ttl if ttl is not None else self.default_ttl
        expire = datetime.now() + timedelta(seconds=ttl)
        self._memory[cache_key] = (value, expire)
        try:
            with self._path(cache_key).open("wb") as f:
                pickle.dump((value, expire), f)
        except Exception:
            pass

    def delete(self, key: str) -> None:
        cache_key = self._key(key)
        self._memory.pop(cache_key, None)
        self._path(cache_key).unlink(missing_ok=True)

    def clear(self) -> None:
        self._memory.clear()
        for f in self.cache_dir.glob("*.pkl"):
            f.unlink(missing_ok=True)

    def clear_expired(self) -> int:
        count = 0
        now = datetime.now()
        expired = [k for k, (_, exp) in self._memory.items() if now >= exp]
        for k in expired:
            del self._memory[k]
            count += 1
        for f in self.cache_dir.glob("*.pkl"):
            try:
                with f.open("rb") as fh:
                    _, exp = pickle.load(fh)
                if now >= exp:
                    f.unlink()
                    count += 1
            except Exception:
                f.unlink(missing_ok=True)
                count += 1
        return count
