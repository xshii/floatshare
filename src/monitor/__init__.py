"""日志与告警 — 薄封装 loguru + apprise"""

from loguru import logger as logger
from .alert import notify as notify

__all__ = ["logger", "notify"]
