"""日志配置模块

提供统一的日志配置，支持：
- 控制台输出
- 文件持久化
- 日志轮转（按大小或时间）
- 不同模块不同日志级别
"""

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional


DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int = logging.INFO,
    log_dir: Optional[str] = None,
    log_file: str = "floatshare.log",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    rotation: str = "size",  # "size" or "time"
    console: bool = True,
    format_string: str = DEFAULT_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> logging.Logger:
    """
    配置全局日志

    Args:
        level: 日志级别
        log_dir: 日志目录，None 则不写文件
        log_file: 日志文件名
        max_bytes: 单文件最大大小（按大小轮转时）
        backup_count: 保留的备份文件数
        rotation: 轮转方式 "size" 或 "time"
        console: 是否输出到控制台
        format_string: 日志格式
        date_format: 日期格式

    Returns:
        root logger
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 清除已有的 handler
    root_logger.handlers.clear()

    formatter = logging.Formatter(format_string, date_format)

    # 控制台输出
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # 文件输出
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        log_file_path = log_path / log_file

        if rotation == "size":
            file_handler = RotatingFileHandler(
                log_file_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
        elif rotation == "time":
            file_handler = TimedRotatingFileHandler(
                log_file_path,
                when="midnight",
                interval=1,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.suffix = "%Y-%m-%d"
        else:
            file_handler = logging.FileHandler(log_file_path, encoding="utf-8")

        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    获取指定名称的 logger

    Args:
        name: logger 名称，通常使用 __name__
        level: 可选的日志级别，覆盖全局设置

    Returns:
        Logger 实例
    """
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger


def setup_module_levels(levels: dict) -> None:
    """
    为不同模块设置不同的日志级别

    Args:
        levels: 模块名到日志级别的映射
                如 {"src.data": logging.DEBUG, "src.strategy": logging.WARNING}
    """
    for module, level in levels.items():
        logging.getLogger(module).setLevel(level)


class LogContext:
    """日志上下文管理器，用于临时调整日志级别"""

    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.new_level = level
        self.old_level = logger.level

    def __enter__(self):
        self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)


# 便捷函数
def init_production_logging(log_dir: str = "logs") -> logging.Logger:
    """生产环境日志配置"""
    return setup_logging(
        level=logging.INFO,
        log_dir=log_dir,
        log_file="floatshare.log",
        max_bytes=50 * 1024 * 1024,  # 50MB
        backup_count=10,
        rotation="size",
        console=True,
    )


def init_debug_logging(log_dir: Optional[str] = None) -> logging.Logger:
    """调试环境日志配置"""
    return setup_logging(
        level=logging.DEBUG,
        log_dir=log_dir,
        console=True,
        format_string="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
    )


def init_sync_logging(log_dir: str = "logs") -> logging.Logger:
    """数据同步日志配置（按天轮转）"""
    return setup_logging(
        level=logging.INFO,
        log_dir=log_dir,
        log_file="sync.log",
        rotation="time",
        backup_count=30,  # 保留 30 天
        console=True,
    )
