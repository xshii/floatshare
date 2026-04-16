"""策略注册表 — 模块级命名空间，配合自动发现。

Python Cookbook 10.14 的标准模式：用 `pkgutil.iter_modules` 遍历包内
所有模块来触发它们的装饰器副作用，避免手写 `from .x import X`。
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable

_REGISTRY: dict[str, type] = {}


def register(name: str | None = None) -> Callable[[type], type]:
    """将策略类登记到全局表。"""

    def decorator(cls: type) -> type:
        key = name or getattr(cls, "name", None) or cls.__name__
        _REGISTRY[key] = cls
        return cls

    return decorator


def get(name: str) -> type | None:
    return _REGISTRY.get(name)


def list_strategies() -> list[str]:
    return sorted(_REGISTRY)


def unregister(name: str) -> bool:
    return _REGISTRY.pop(name, None) is not None


def clear() -> None:
    _REGISTRY.clear()


def discover(package: str = "strategies") -> list[str]:
    """递归 import 指定包下所有模块，触发 `@register` 副作用。

    Args:
        package: 要扫描的包名（默认 `strategies`，项目根的应用层策略目录）

    Returns:
        新加载的模块名列表。
    """
    pkg = importlib.import_module(package)
    loaded: list[str] = []
    for mod_info in pkgutil.iter_modules(pkg.__path__, prefix=f"{package}."):
        importlib.import_module(mod_info.name)
        loaded.append(mod_info.name)
    return loaded
