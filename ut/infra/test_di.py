"""依赖注入测试"""

from abc import ABC, abstractmethod

import pytest

from src.utils.di import (
    Container,
    Lifetime,
    ServiceDescriptor,
    get_container,
    inject,
    injectable,
    set_container,
)


# ============================================================
# 测试用接口和实现
# ============================================================


class IService(ABC):
    @abstractmethod
    def do_something(self) -> str:
        pass


class ServiceA(IService):
    def __init__(self):
        self.name = "ServiceA"

    def do_something(self) -> str:
        return f"Hello from {self.name}"


class ServiceB(IService):
    def __init__(self):
        self.name = "ServiceB"

    def do_something(self) -> str:
        return f"Hello from {self.name}"


class DependentService:
    def __init__(self, service: IService):
        self.service = service

    def call_service(self) -> str:
        return self.service.do_something()


class ServiceWithDefault:
    def __init__(self, value: int = 42):
        self.value = value


# ============================================================
# 测试类
# ============================================================


class TestServiceDescriptor:
    """ServiceDescriptor 测试"""

    def test_descriptor_with_class(self):
        desc = ServiceDescriptor(IService, ServiceA, Lifetime.SINGLETON)
        assert desc.service_type == IService
        assert desc.implementation == ServiceA
        assert desc.lifetime == Lifetime.SINGLETON
        assert desc.instance is None

    def test_descriptor_with_instance(self):
        instance = ServiceA()
        desc = ServiceDescriptor(IService, instance, Lifetime.TRANSIENT)
        # 实例应该自动变成单例
        assert desc.instance is instance
        assert desc.lifetime == Lifetime.SINGLETON


class TestContainer:
    """Container 测试"""

    def setup_method(self):
        self.container = Container()

    def test_register_and_resolve_class(self):
        self.container.register(IService, ServiceA)
        service = self.container.resolve(IService)

        assert isinstance(service, ServiceA)
        assert service.do_something() == "Hello from ServiceA"

    def test_register_and_resolve_instance(self):
        instance = ServiceA()
        instance.name = "CustomName"
        self.container.register(IService, instance)

        service = self.container.resolve(IService)
        assert service.do_something() == "Hello from CustomName"

    def test_register_and_resolve_factory(self):
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return ServiceA()

        self.container.register(IService, factory, Lifetime.TRANSIENT)

        self.container.resolve(IService)
        self.container.resolve(IService)

        assert call_count == 2  # 瞬态模式，每次调用工厂

    def test_singleton_lifetime(self):
        self.container.register(IService, ServiceA, Lifetime.SINGLETON)

        service1 = self.container.resolve(IService)
        service2 = self.container.resolve(IService)

        assert service1 is service2

    def test_transient_lifetime(self):
        self.container.register(IService, ServiceA, Lifetime.TRANSIENT)

        service1 = self.container.resolve(IService)
        service2 = self.container.resolve(IService)

        assert service1 is not service2

    def test_scoped_lifetime(self):
        self.container.register(IService, ServiceA, Lifetime.SCOPED)

        # 同一作用域内共享
        service1 = self.container.resolve(IService, scope_id="scope1")
        service2 = self.container.resolve(IService, scope_id="scope1")
        assert service1 is service2

        # 不同作用域不同实例
        service3 = self.container.resolve(IService, scope_id="scope2")
        assert service1 is not service3

    def test_scoped_requires_scope_id(self):
        self.container.register(IService, ServiceA, Lifetime.SCOPED)

        with pytest.raises(ValueError, match="scope_id"):
            self.container.resolve(IService)  # 没有提供 scope_id

    def test_clear_scope(self):
        self.container.register(IService, ServiceA, Lifetime.SCOPED)

        service1 = self.container.resolve(IService, scope_id="scope1")
        self.container.clear_scope("scope1")
        service2 = self.container.resolve(IService, scope_id="scope1")

        assert service1 is not service2  # 清除后是新实例

    def test_resolve_unregistered_raises(self):
        with pytest.raises(KeyError, match="未注册"):
            self.container.resolve(IService)

    def test_is_registered(self):
        assert not self.container.is_registered(IService)
        self.container.register(IService, ServiceA)
        assert self.container.is_registered(IService)

    def test_auto_inject_dependency(self):
        self.container.register(IService, ServiceA)
        self.container.register(DependentService, DependentService)

        dependent = self.container.resolve(DependentService)

        assert isinstance(dependent.service, ServiceA)
        assert dependent.call_service() == "Hello from ServiceA"

    def test_auto_inject_with_default(self):
        self.container.register(ServiceWithDefault, ServiceWithDefault)

        service = self.container.resolve(ServiceWithDefault)
        assert service.value == 42

    def test_fluent_api(self):
        result = (
            self.container
            .register_singleton(IService, ServiceA)
            .register_transient(ServiceWithDefault, ServiceWithDefault)
        )

        assert result is self.container
        assert self.container.is_registered(IService)
        assert self.container.is_registered(ServiceWithDefault)

    def test_get_all_services(self):
        self.container.register(IService, ServiceA)
        self.container.register(ServiceWithDefault, ServiceWithDefault)

        services = self.container.get_all_services()
        assert len(services) == 2
        assert IService in services
        assert ServiceWithDefault in services


class TestInjectableDecorator:
    """@injectable 装饰器测试"""

    def setup_method(self):
        # 重置全局容器
        set_container(Container())

    def test_injectable_registers_class(self):
        @injectable()
        class MyService:
            pass

        container = get_container()
        assert container.is_registered(MyService)

    def test_injectable_with_interface(self):
        @injectable(as_type=IService)
        class MyServiceImpl(IService):
            def do_something(self) -> str:
                return "Hello"

        container = get_container()
        assert container.is_registered(IService)

        service = container.resolve(IService)
        assert isinstance(service, MyServiceImpl)

    def test_injectable_with_lifetime(self):
        @injectable(lifetime=Lifetime.TRANSIENT)
        class TransientService:
            pass

        container = get_container()
        service1 = container.resolve(TransientService)
        service2 = container.resolve(TransientService)

        assert service1 is not service2


class TestInjectFunction:
    """inject() 函数测试"""

    def setup_method(self):
        container = Container()
        container.register(IService, ServiceA)
        set_container(container)

    def test_inject(self):
        service = inject(IService)
        assert isinstance(service, ServiceA)


class TestGlobalContainer:
    """全局容器测试"""

    def test_get_container_returns_container(self):
        container = get_container()
        assert isinstance(container, Container)

    def test_set_container(self):
        new_container = Container()
        new_container.register(IService, ServiceB)

        set_container(new_container)

        assert get_container() is new_container
        service = get_container().resolve(IService)
        assert isinstance(service, ServiceB)
