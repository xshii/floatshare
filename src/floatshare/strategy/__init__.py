"""策略框架 — backtrader.Strategy 直接暴露，外加注册表与因子库。"""

from floatshare.strategy.registry import (
    clear,
    discover,
    get,
    list_strategies,
    register,
    unregister,
)

__all__ = ["clear", "discover", "get", "list_strategies", "register", "unregister"]
