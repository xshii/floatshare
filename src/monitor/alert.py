"""告警通道 — apprise 一行封装

环境变量 FLOATSHARE_NOTIFY_URLS 配置目标，逗号分隔。
示例: tgram://bottoken/chatid,mailto://user:pass@gmail.com
没配置时 notify() 静默 no-op。
"""

from __future__ import annotations

import os
from typing import Optional

import apprise
from loguru import logger

_apobj: Optional[apprise.Apprise] = None


def _get() -> apprise.Apprise:
    global _apobj
    if _apobj is None:
        _apobj = apprise.Apprise()
        urls = os.getenv("FLOATSHARE_NOTIFY_URLS", "").strip()
        for u in [s.strip() for s in urls.split(",") if s.strip()]:
            _apobj.add(u)
    return _apobj


def notify(title: str, body: str = "") -> bool:
    """发送告警；未配置渠道时仅打日志。"""
    ap = _get()
    if len(ap) == 0:
        logger.info(f"[NOTIFY] {title} — {body}")
        return False
    return ap.notify(title=title, body=body)
