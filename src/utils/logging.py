"""日志配置模块

提供统一的日志配置，支持：
- 控制台输出
- 文件持久化
- 日志轮转（按大小或时间）
- 不同模块不同日志级别
- 结构化日志 (structlog)
"""

import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar

try:
    import structlog
    from structlog.types import Processor
    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False

T = TypeVar("T")

DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ============================================================
# 基础日志配置 (标准库)
# ============================================================


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


# ============================================================
# 便捷函数
# ============================================================


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


# ============================================================
# 结构化日志 (structlog)
# ============================================================


def configure_structlog(
    level: str = "INFO",
    json_format: bool = False,
    add_timestamp: bool = True,
    log_file: Optional[str] = None,
) -> None:
    """
    配置结构化日志

    Args:
        level: 日志级别
        json_format: 是否使用 JSON 格式
        add_timestamp: 是否添加时间戳
        log_file: 日志文件路径
    """
    if not HAS_STRUCTLOG:
        raise ImportError("structlog 未安装，请运行: pip install structlog")

    # 共享处理器
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if add_timestamp:
        shared_processors.insert(0, structlog.processors.TimeStamper(fmt="iso"))

    # 根据格式选择渲染器
    if json_format:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    # 配置 structlog
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # 配置标准库 logging
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    handlers = [console_handler]

    # 文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.handlers = handlers
    root_logger.setLevel(getattr(logging, level.upper()))


def get_structlog(name: Optional[str] = None):
    """
    获取结构化日志器

    Args:
        name: 日志器名称

    Returns:
        绑定的结构化日志器
    """
    if not HAS_STRUCTLOG:
        return get_logger(name or __name__)
    return structlog.get_logger(name)


# ============================================================
# 上下文管理
# ============================================================


def bind_context(**kwargs) -> None:
    """绑定上下文变量"""
    if HAS_STRUCTLOG:
        structlog.contextvars.bind_contextvars(**kwargs)


def unbind_context(*keys: str) -> None:
    """解绑上下文变量"""
    if HAS_STRUCTLOG:
        structlog.contextvars.unbind_contextvars(*keys)


def clear_context() -> None:
    """清除所有上下文变量"""
    if HAS_STRUCTLOG:
        structlog.contextvars.clear_contextvars()


@contextmanager
def log_context(**kwargs):
    """
    临时绑定上下文变量

    Example:
        with log_context(request_id="abc123"):
            logger.info("processing request")
    """
    bind_context(**kwargs)
    try:
        yield
    finally:
        unbind_context(*kwargs.keys())


# ============================================================
# 性能追踪
# ============================================================


@contextmanager
def log_duration(
    operation: str,
    logger: Optional[Any] = None,
    level: str = "info",
    **extra,
):
    """
    记录操作耗时

    Example:
        with log_duration("data_sync", source="akshare"):
            sync_data()
    """
    if logger is None:
        logger = get_structlog()

    start_time = time.perf_counter()
    try:
        yield
        duration_ms = (time.perf_counter() - start_time) * 1000
        getattr(logger, level)(
            f"{operation}_completed",
            duration_ms=round(duration_ms, 2),
            **extra,
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            f"{operation}_failed",
            duration_ms=round(duration_ms, 2),
            error=str(e),
            error_type=type(e).__name__,
            **extra,
        )
        raise


def log_function_call(
    logger: Optional[Any] = None,
    level: str = "debug",
    log_args: bool = True,
    log_result: bool = False,
):
    """
    函数调用日志装饰器

    Example:
        @log_function_call()
        def process_data(code: str) -> pd.DataFrame:
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            nonlocal logger
            if logger is None:
                logger = get_structlog(func.__module__)

            call_info: Dict[str, Any] = {
                "function": func.__name__,
            }

            if log_args:
                call_info["args"] = str(args)[:200]
                call_info["kwargs"] = str(kwargs)[:200]

            getattr(logger, level)(f"calling_{func.__name__}", **call_info)

            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000

                result_info: Dict[str, Any] = {
                    "function": func.__name__,
                    "duration_ms": round(duration_ms, 2),
                }

                if log_result:
                    result_info["result"] = str(result)[:200]

                getattr(logger, level)(f"{func.__name__}_completed", **result_info)
                return result

            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.error(
                    f"{func.__name__}_failed",
                    function=func.__name__,
                    duration_ms=round(duration_ms, 2),
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise

        return wrapper
    return decorator


# ============================================================
# 业务日志辅助
# ============================================================


class DataSyncLogger:
    """数据同步专用日志器"""

    def __init__(self):
        self.logger = get_structlog("data_sync")

    def sync_started(self, source: str, codes_count: int) -> None:
        self.logger.info(
            "sync_started",
            source=source,
            codes_count=codes_count,
        )

    def sync_progress(
        self,
        source: str,
        completed: int,
        total: int,
        current_code: str,
    ) -> None:
        self.logger.info(
            "sync_progress",
            source=source,
            completed=completed,
            total=total,
            current_code=current_code,
            progress_pct=round(completed / total * 100, 1),
        )

    def sync_completed(
        self,
        source: str,
        total: int,
        success: int,
        failed: int,
        duration_seconds: float,
    ) -> None:
        self.logger.info(
            "sync_completed",
            source=source,
            total=total,
            success=success,
            failed=failed,
            duration_seconds=round(duration_seconds, 2),
            success_rate=round(success / total * 100, 1) if total > 0 else 0,
        )

    def sync_error(self, source: str, code: str, error: str) -> None:
        self.logger.error(
            "sync_error",
            source=source,
            code=code,
            error=error,
        )


class BacktestLogger:
    """回测专用日志器"""

    def __init__(self):
        self.logger = get_structlog("backtest")

    def backtest_started(
        self,
        strategy: str,
        start_date: str,
        end_date: str,
        initial_capital: float,
    ) -> None:
        self.logger.info(
            "backtest_started",
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
        )

    def backtest_completed(
        self,
        strategy: str,
        total_return: float,
        max_drawdown: float,
        sharpe_ratio: float,
        duration_seconds: float,
    ) -> None:
        self.logger.info(
            "backtest_completed",
            strategy=strategy,
            total_return=round(total_return, 4),
            max_drawdown=round(max_drawdown, 4),
            sharpe_ratio=round(sharpe_ratio, 2),
            duration_seconds=round(duration_seconds, 2),
        )

    def trade_executed(
        self,
        code: str,
        direction: str,
        price: float,
        volume: int,
        amount: float,
    ) -> None:
        self.logger.info(
            "trade_executed",
            code=code,
            direction=direction,
            price=price,
            volume=volume,
            amount=amount,
        )
