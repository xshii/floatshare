"""FloatShare — 个人量化交易框架。

公开 API 走 facade，使用方写：
    from floatshare import run_backtest, register, discover
"""

from floatshare.application import (
    AllSourcesFailed,
    BacktestResult,
    DataLoader,
    create_default_loader,
    run_backtest,
)
from floatshare.observability import logger, notify
from floatshare.strategy import discover, get, list_strategies, register

__version__ = "0.2.0"

__all__ = [
    "AllSourcesFailed",
    "BacktestResult",
    "DataLoader",
    "__version__",
    "create_default_loader",
    "discover",
    "get",
    "list_strategies",
    "logger",
    "notify",
    "register",
    "run_backtest",
]
