"""可观测性 — 日志 + 告警，零内部依赖。"""

from loguru import logger as logger

from floatshare.observability.alert import notify

__all__ = ["logger", "notify"]
