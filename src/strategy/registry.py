"""策略注册中心"""

from typing import Dict, Type, Optional, List
from src.strategy.base import Strategy


class StrategyRegistry:
    """策略注册中心"""

    _strategies: Dict[str, Type[Strategy]] = {}

    @classmethod
    def register(cls, name: Optional[str] = None):
        """
        策略注册装饰器

        Usage:
            @StrategyRegistry.register("my_strategy")
            class MyStrategy(Strategy):
                pass
        """

        def decorator(strategy_cls: Type[Strategy]) -> Type[Strategy]:
            strategy_name = name or strategy_cls.name or strategy_cls.__name__
            cls._strategies[strategy_name] = strategy_cls
            return strategy_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Type[Strategy]]:
        """获取策略类"""
        return cls._strategies.get(name)

    @classmethod
    def create(cls, name: str, params: Optional[dict] = None) -> Strategy:
        """创建策略实例"""
        strategy_cls = cls.get(name)
        if strategy_cls is None:
            raise ValueError(f"策略 '{name}' 未注册")
        return strategy_cls(params)

    @classmethod
    def list_strategies(cls) -> List[str]:
        """列出所有已注册策略"""
        return list(cls._strategies.keys())

    @classmethod
    def unregister(cls, name: str) -> bool:
        """注销策略"""
        if name in cls._strategies:
            del cls._strategies[name]
            return True
        return False

    @classmethod
    def clear(cls) -> None:
        """清空所有注册"""
        cls._strategies.clear()
