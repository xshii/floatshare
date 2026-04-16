"""策略注册中心 — 仅做 name -> class 的映射。"""

from __future__ import annotations

from typing import Dict, List, Optional, Type

import backtrader as bt


class StrategyRegistry:
    """策略注册表（轻量装饰器版本）。"""

    _strategies: Dict[str, Type[bt.Strategy]] = {}

    @classmethod
    def register(cls, name: Optional[str] = None):
        def decorator(strategy_cls: Type[bt.Strategy]) -> Type[bt.Strategy]:
            key = name or getattr(strategy_cls, "name", None) or strategy_cls.__name__
            cls._strategies[key] = strategy_cls
            return strategy_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Type[bt.Strategy]]:
        return cls._strategies.get(name)

    @classmethod
    def list_strategies(cls) -> List[str]:
        return list(cls._strategies.keys())

    @classmethod
    def unregister(cls, name: str) -> bool:
        return cls._strategies.pop(name, None) is not None

    @classmethod
    def clear(cls) -> None:
        cls._strategies.clear()
