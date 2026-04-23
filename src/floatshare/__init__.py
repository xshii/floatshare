"""FloatShare — 个人量化交易框架。

新人 5 分钟入门:
    from floatshare import register, run_backtest, create_default_loader, logger

    @register("my_strategy")
    class MyStrategy(bt.Strategy):
        ...

    loader = create_default_loader()
    data = loader.get_daily("600000.SH")
    result = run_backtest(MyStrategy, data)
    result.print_summary()

更深入的用法按需 import:
    from floatshare.factors.technical import MAFactor       # 因子库
    from floatshare.interfaces.data_source import ...        # Protocol 契约
    from floatshare.application import DataSyncer            # 用例
    from floatshare.infrastructure.data_sources.tushare import TushareSource  # 适配器实现
"""

# 包入口加载 .env — 让 TUSHARE_TOKEN / FLOATSHARE_NOTIFY_URLS 等无需 shell export
from dotenv import load_dotenv as _load_dotenv

_load_dotenv()

from floatshare.application import (
    AllSourcesFailed as AllSourcesFailed,
)
from floatshare.application import (
    BacktestResult as BacktestResult,
)
from floatshare.application import (
    DataLoader as DataLoader,
)
from floatshare.application import (
    DataSyncer as DataSyncer,
)
from floatshare.application import (
    create_default_loader as create_default_loader,
)
from floatshare.application import (
    run_backtest as run_backtest,
)
from floatshare.observability import logger as logger
from floatshare.observability import notify as notify
from floatshare.registry import (
    clear as clear,
)
from floatshare.registry import (
    discover as discover,
)
from floatshare.registry import (
    get as get,
)
from floatshare.registry import (
    list_strategies as list_strategies,
)
from floatshare.registry import (
    register as register,
)
from floatshare.registry import (
    unregister as unregister,
)

__version__ = "0.2.0"

__all__ = [
    "AllSourcesFailed",
    "BacktestResult",
    "DataLoader",
    "DataSyncer",
    "__version__",
    "clear",
    "create_default_loader",
    "discover",
    "get",
    "list_strategies",
    "logger",
    "notify",
    "register",
    "run_backtest",
    "unregister",
]
