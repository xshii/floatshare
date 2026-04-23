"""可观测性 — 日志 + 告警，零内部依赖。

日志写入两个 sink:
  1. stderr (彩色)
  2. logs/floatshare_YYYY-MM-DD.log (按天滚动 + 30 天保留 + zip 压缩)

环境变量:
  LOG_LEVEL  — 日志级别 (DEBUG/INFO/WARNING/ERROR), 默认 INFO
  LOG_DIR    — 日志目录, 默认 ./logs
  LOG_FILE_DISABLE — 设为 "1" 可禁用文件输出（测试/CI 用）
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger as logger

from floatshare.observability import features
from floatshare.observability.alert import notify

__all__ = ["features", "logger", "notify", "setup_logger"]

_DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

_FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"


def setup_logger() -> None:
    """初始化 logger（幂等）— 移除默认 sink，添加 stderr + 文件 sink。"""
    logger.remove()

    level = os.getenv("LOG_LEVEL", "INFO").upper()

    logger.add(sys.stderr, level=level, format=_DEFAULT_FORMAT, colorize=True)

    if os.getenv("LOG_FILE_DISABLE") == "1":
        return

    log_dir = Path(os.getenv("LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "floatshare_{time:YYYY-MM-DD}.log",
        level=level,
        format=_FILE_FORMAT,
        rotation="00:00",  # 每天 0 点滚动
        retention="30 days",  # 保留 30 天
        compression="zip",  # 旧日志压缩
        enqueue=True,  # 多进程安全
        encoding="utf-8",
    )


# 模块加载时自动初始化
setup_logger()
