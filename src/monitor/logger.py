"""日志管理"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class Logger:
    """日志管理器"""

    _loggers: dict = {}

    def __init__(
        self,
        name: str = "floatshare",
        level: str = "INFO",
        log_dir: Optional[str] = None,
        console: bool = True,
        file: bool = True,
    ):
        """
        初始化日志器

        Args:
            name: 日志器名称
            level: 日志级别
            log_dir: 日志目录
            console: 是否输出到控制台
            file: 是否输出到文件
        """
        self.name = name
        self.level = getattr(logging, level.upper(), logging.INFO)

        if log_dir:
            self.log_dir = Path(log_dir)
        else:
            self.log_dir = Path(__file__).parent.parent.parent / "logs"

        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._logger = self._setup_logger(console, file)

    def _setup_logger(self, console: bool, file: bool) -> logging.Logger:
        """设置日志器"""
        if self.name in Logger._loggers:
            return Logger._loggers[self.name]

        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)
        logger.handlers.clear()

        # 格式
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # 控制台处理器
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # 文件处理器
        if file:
            today = datetime.now().strftime("%Y%m%d")
            log_file = self.log_dir / f"{self.name}_{today}.log"

            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(self.level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        Logger._loggers[self.name] = logger
        return logger

    def debug(self, message: str, *args, **kwargs) -> None:
        """DEBUG日志"""
        self._logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        """INFO日志"""
        self._logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        """WARNING日志"""
        self._logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        """ERROR日志"""
        self._logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs) -> None:
        """CRITICAL日志"""
        self._logger.critical(message, *args, **kwargs)

    def exception(self, message: str, *args, **kwargs) -> None:
        """异常日志（包含堆栈）"""
        self._logger.exception(message, *args, **kwargs)

    @classmethod
    def get_logger(cls, name: str = "floatshare") -> "Logger":
        """获取日志器"""
        if name not in cls._loggers:
            return cls(name)
        return cls._loggers[name]


# 全局日志器
_default_logger: Optional[Logger] = None


def get_logger(name: str = "floatshare") -> Logger:
    """获取全局日志器"""
    global _default_logger
    if _default_logger is None:
        _default_logger = Logger(name)
    return _default_logger


def debug(message: str, *args, **kwargs) -> None:
    """全局DEBUG日志"""
    get_logger().debug(message, *args, **kwargs)


def info(message: str, *args, **kwargs) -> None:
    """全局INFO日志"""
    get_logger().info(message, *args, **kwargs)


def warning(message: str, *args, **kwargs) -> None:
    """全局WARNING日志"""
    get_logger().warning(message, *args, **kwargs)


def error(message: str, *args, **kwargs) -> None:
    """全局ERROR日志"""
    get_logger().error(message, *args, **kwargs)
