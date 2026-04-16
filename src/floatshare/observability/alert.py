"""告警 — apprise 一行封装，环境变量驱动。"""

from __future__ import annotations

import os
from functools import lru_cache

import apprise
from loguru import logger


@lru_cache(maxsize=1)
def _client() -> apprise.Apprise:
    """惰性构造 apprise 客户端；测试时用 `_client.cache_clear()` 重置。"""
    ap = apprise.Apprise()
    raw = os.getenv("FLOATSHARE_NOTIFY_URLS", "").strip()
    for url in (s.strip() for s in raw.split(",") if s.strip()):
        ap.add(url)
    return ap


def notify(title: str, body: str = "") -> bool:
    """发送告警；未配置渠道时仅打日志、返回 False。"""
    ap = _client()
    if len(ap) == 0:
        logger.info(f"[NOTIFY] {title} — {body}")
        return False
    return bool(ap.notify(title=title, body=body))
