"""依赖注入容器

提供简单的依赖注入支持：
- 服务注册和解析
- 单例和工厂模式
- 作用域管理
- 接口绑定
"""

import inspect
import logging
from abc import ABC, abstractmethod
from enum import Enum
from threading import Lock
from typing import Any, Callable, Dict, Generic, Optional, Type, TypeVar, Union, get_type_hints

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Lifetime(Enum):
    """服务生命周期"""
    SINGLETON = "singleton"  # 单例，全局唯一
    TRANSIENT = "transient"  # 瞬态，每次请求创建新实例
    SCOPED = "scoped"  # 作用域，同一作用域内共享


class ServiceDescriptor:
    """服务描述符"""

    def __init__(
        self,
        service_type: Type,
        implementation: Union[Type, Callable, Any],
        lifetime: Lifetime = Lifetime.SINGLETON,
    ):
        self.service_type = service_type
        self.implementation = implementation
        self.lifetime = lifetime
        self.instance: Optional[Any] = None

        # 如果是实例，直接设为单例
        if not callable(implementation) and not isinstance(implementation, type):
            self.instance = implementation
            self.lifetime = Lifetime.SINGLETON


class Container:
    """依赖注入容器"""

    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._lock = Lock()
        self._scoped_instances: Dict[str, Dict[Type, Any]] = {}

    def register(
        self,
        service_type: Type[T],
        implementation: Union[Type[T], Callable[..., T], T],
        lifetime: Lifetime = Lifetime.SINGLETON,
    ) -> "Container":
        """
        注册服务

        Args:
            service_type: 服务接口/类型
            implementation: 实现类、工厂函数或实例
            lifetime: 生命周期

        Example:
            container.register(IDataLoader, AKShareLoader, Lifetime.SINGLETON)
            container.register(ICache, lambda: LRUCache(1000))
            container.register(Settings, settings_instance)
        """
        with self._lock:
            self._services[service_type] = ServiceDescriptor(
                service_type=service_type,
                implementation=implementation,
                lifetime=lifetime,
            )
        return self

    def register_singleton(
        self,
        service_type: Type[T],
        implementation: Union[Type[T], Callable[..., T], T],
    ) -> "Container":
        """注册单例服务"""
        return self.register(service_type, implementation, Lifetime.SINGLETON)

    def register_transient(
        self,
        service_type: Type[T],
        implementation: Union[Type[T], Callable[..., T]],
    ) -> "Container":
        """注册瞬态服务"""
        return self.register(service_type, implementation, Lifetime.TRANSIENT)

    def register_scoped(
        self,
        service_type: Type[T],
        implementation: Union[Type[T], Callable[..., T]],
    ) -> "Container":
        """注册作用域服务"""
        return self.register(service_type, implementation, Lifetime.SCOPED)

    def resolve(self, service_type: Type[T], scope_id: Optional[str] = None) -> T:
        """
        解析服务

        Args:
            service_type: 服务类型
            scope_id: 作用域 ID（用于 SCOPED 生命周期）

        Returns:
            服务实例

        Raises:
            KeyError: 服务未注册
        """
        if service_type not in self._services:
            raise KeyError(f"服务未注册: {service_type.__name__}")

        descriptor = self._services[service_type]

        # 单例：返回已有实例或创建新实例
        if descriptor.lifetime == Lifetime.SINGLETON:
            if descriptor.instance is None:
                with self._lock:
                    if descriptor.instance is None:
                        descriptor.instance = self._create_instance(descriptor)
            return descriptor.instance

        # 瞬态：每次创建新实例
        if descriptor.lifetime == Lifetime.TRANSIENT:
            return self._create_instance(descriptor)

        # 作用域：同一作用域内共享
        if descriptor.lifetime == Lifetime.SCOPED:
            if scope_id is None:
                raise ValueError("SCOPED 服务需要提供 scope_id")

            if scope_id not in self._scoped_instances:
                self._scoped_instances[scope_id] = {}

            scoped = self._scoped_instances[scope_id]
            if service_type not in scoped:
                scoped[service_type] = self._create_instance(descriptor)

            return scoped[service_type]

        raise ValueError(f"未知的生命周期: {descriptor.lifetime}")

    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """创建服务实例"""
        impl = descriptor.implementation

        # 如果是实例，直接返回
        if not callable(impl) and not isinstance(impl, type):
            return impl

        # 如果是工厂函数
        if callable(impl) and not isinstance(impl, type):
            return impl()

        # 如果是类，尝试自动注入依赖
        return self._construct(impl)

    def _construct(self, cls: Type[T]) -> T:
        """构造实例，自动注入依赖"""
        # 获取构造函数签名
        sig = inspect.signature(cls.__init__)
        hints = get_type_hints(cls.__init__) if hasattr(cls.__init__, "__annotations__") else {}

        kwargs = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue

            # 获取参数类型
            param_type = hints.get(name)
            if param_type is None:
                # 无类型注解，检查是否有默认值
                if param.default is inspect.Parameter.empty:
                    raise ValueError(
                        f"无法注入参数 {name}: 缺少类型注解且无默认值"
                    )
                continue

            # 尝试解析依赖
            if param_type in self._services:
                kwargs[name] = self.resolve(param_type)
            elif param.default is not inspect.Parameter.empty:
                # 有默认值，使用默认值
                continue
            else:
                raise ValueError(
                    f"无法注入参数 {name}: 类型 {param_type.__name__} 未注册"
                )

        return cls(**kwargs)

    def clear_scope(self, scope_id: str) -> None:
        """清除指定作用域的实例"""
        if scope_id in self._scoped_instances:
            del self._scoped_instances[scope_id]

    def is_registered(self, service_type: Type) -> bool:
        """检查服务是否已注册"""
        return service_type in self._services

    def get_all_services(self) -> Dict[Type, ServiceDescriptor]:
        """获取所有已注册的服务"""
        return dict(self._services)


# ============================================================
# 全局容器
# ============================================================

_container: Optional[Container] = None


def get_container() -> Container:
    """获取全局容器"""
    global _container
    if _container is None:
        _container = Container()
    return _container


def set_container(container: Container) -> None:
    """设置全局容器"""
    global _container
    _container = container


# ============================================================
# 装饰器
# ============================================================


def injectable(
    lifetime: Lifetime = Lifetime.SINGLETON,
    as_type: Optional[Type] = None,
) -> Callable[[Type[T]], Type[T]]:
    """
    标记类为可注入服务

    Args:
        lifetime: 生命周期
        as_type: 注册为指定类型（接口）

    Example:
        @injectable()
        class MyService:
            pass

        @injectable(as_type=IDataLoader)
        class AKShareLoader(IDataLoader):
            pass
    """
    def decorator(cls: Type[T]) -> Type[T]:
        container = get_container()
        service_type = as_type or cls
        container.register(service_type, cls, lifetime)
        return cls

    return decorator


def inject(service_type: Type[T]) -> T:
    """
    注入服务

    Example:
        loader = inject(IDataLoader)
    """
    return get_container().resolve(service_type)


# ============================================================
# 服务提供者
# ============================================================


class ServiceProvider:
    """服务提供者基类"""

    @abstractmethod
    def register(self, container: Container) -> None:
        """注册服务到容器"""
        pass


class DataServiceProvider(ServiceProvider):
    """数据服务提供者示例"""

    def register(self, container: Container) -> None:
        from src.data.loader import DataLoader
        from src.data.storage.database import DatabaseStorage
        from src.data.validator import DataValidator

        container.register_singleton(DataLoader, lambda: DataLoader(source="akshare"))
        container.register_singleton(DatabaseStorage, DatabaseStorage)
        container.register_transient(DataValidator, DataValidator)


class CacheServiceProvider(ServiceProvider):
    """缓存服务提供者示例"""

    def register(self, container: Container) -> None:
        from src.utils.cache import BaseCache, LRUCache, get_cache

        container.register_singleton(BaseCache, get_cache)


def configure_services(providers: list[ServiceProvider]) -> Container:
    """配置服务"""
    container = get_container()
    for provider in providers:
        provider.register(container)
    return container
