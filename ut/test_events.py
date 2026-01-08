"""事件总线测试"""

import time
from datetime import datetime

import pytest

from src.utils.events import (
    DataEvent,
    ErrorEvent,
    Event,
    EventBus,
    EventHandler,
    OrderEvent,
    OrderFilledEvent,
    OrderRejectedEvent,
    Priority,
    SignalEvent,
    SyncProgressEvent,
    get_event_bus,
    on_event,
    publish,
    set_event_bus,
    subscribe,
)


class TestEvent:
    """Event 基类测试"""

    def test_event_timestamp(self):
        event = Event()
        assert isinstance(event.timestamp, datetime)

    def test_event_type(self):
        event = Event()
        assert event.event_type == "Event"

        order_event = OrderEvent()
        assert order_event.event_type == "OrderEvent"

    def test_event_source(self):
        event = Event(source="test_source")
        assert event.source == "test_source"


class TestOrderEvents:
    """订单事件测试"""

    def test_order_event(self):
        event = OrderEvent(
            order_id="123",
            code="000001.SZ",
            direction="buy",
            quantity=100,
            price=10.5,
        )
        assert event.order_id == "123"
        assert event.code == "000001.SZ"

    def test_order_filled_event(self):
        event = OrderFilledEvent(
            order_id="123",
            fill_price=10.5,
            fill_quantity=100,
            commission=5.0,
        )
        assert event.fill_price == 10.5
        assert event.commission == 5.0

    def test_order_rejected_event(self):
        event = OrderRejectedEvent(
            order_id="123",
            reason="Insufficient balance",
        )
        assert event.reason == "Insufficient balance"


class TestSyncProgressEvent:
    """同步进度事件测试"""

    def test_progress_pct(self):
        event = SyncProgressEvent(total=100, completed=50)
        assert event.progress_pct == 50.0

    def test_progress_pct_zero_total(self):
        event = SyncProgressEvent(total=0, completed=0)
        assert event.progress_pct == 0


class TestEventHandler:
    """EventHandler 测试"""

    def test_should_handle_without_filter(self):
        handler = EventHandler(callback=lambda e: None)
        assert handler.should_handle(Event())

    def test_should_handle_with_filter_true(self):
        handler = EventHandler(
            callback=lambda e: None,
            filter_fn=lambda e: True,
        )
        assert handler.should_handle(Event())

    def test_should_handle_with_filter_false(self):
        handler = EventHandler(
            callback=lambda e: None,
            filter_fn=lambda e: False,
        )
        assert not handler.should_handle(Event())


class TestEventBus:
    """EventBus 测试"""

    def setup_method(self):
        self.bus = EventBus()
        self.received_events = []

    def test_subscribe_and_publish(self):
        self.bus.subscribe(Event, lambda e: self.received_events.append(e))

        event = Event()
        self.bus.publish(event)

        assert len(self.received_events) == 1
        assert self.received_events[0] is event

    def test_subscribe_specific_type(self):
        self.bus.subscribe(OrderEvent, lambda e: self.received_events.append(e))

        self.bus.publish(Event())  # 不应该收到
        self.bus.publish(OrderEvent())  # 应该收到

        assert len(self.received_events) == 1
        assert isinstance(self.received_events[0], OrderEvent)

    def test_subscribe_parent_type(self):
        """订阅父类型应该收到子类型事件"""
        self.bus.subscribe(OrderEvent, lambda e: self.received_events.append(e))

        self.bus.publish(OrderFilledEvent())  # 子类事件

        assert len(self.received_events) == 1
        assert isinstance(self.received_events[0], OrderFilledEvent)

    def test_priority(self):
        results = []

        self.bus.subscribe(
            Event,
            lambda e: results.append("low"),
            priority=Priority.LOW,
        )
        self.bus.subscribe(
            Event,
            lambda e: results.append("high"),
            priority=Priority.HIGH,
        )
        self.bus.subscribe(
            Event,
            lambda e: results.append("normal"),
            priority=Priority.NORMAL,
        )

        self.bus.publish(Event())

        assert results == ["high", "normal", "low"]

    def test_filter(self):
        self.bus.subscribe(
            OrderEvent,
            lambda e: self.received_events.append(e),
            filter_fn=lambda e: e.code == "000001.SZ",
        )

        self.bus.publish(OrderEvent(code="000001.SZ"))  # 应该收到
        self.bus.publish(OrderEvent(code="600000.SH"))  # 不应该收到

        assert len(self.received_events) == 1
        assert self.received_events[0].code == "000001.SZ"

    def test_once(self):
        self.bus.subscribe(
            Event,
            lambda e: self.received_events.append(e),
            once=True,
        )

        self.bus.publish(Event())
        self.bus.publish(Event())

        assert len(self.received_events) == 1  # 只收到一次

    def test_subscribe_all(self):
        self.bus.subscribe_all(lambda e: self.received_events.append(e))

        self.bus.publish(Event())
        self.bus.publish(OrderEvent())
        self.bus.publish(SignalEvent())

        assert len(self.received_events) == 3

    def test_unsubscribe(self):
        handler = lambda e: self.received_events.append(e)
        self.bus.subscribe(Event, handler)

        self.bus.publish(Event())
        assert len(self.received_events) == 1

        result = self.bus.unsubscribe(Event, handler)
        assert result is True

        self.bus.publish(Event())
        assert len(self.received_events) == 1  # 不再收到

    def test_unsubscribe_not_found(self):
        result = self.bus.unsubscribe(Event, lambda e: None)
        assert result is False

    def test_fluent_api(self):
        result = (
            self.bus
            .subscribe(Event, lambda e: None)
            .subscribe(OrderEvent, lambda e: None)
        )
        assert result is self.bus

    def test_exception_handling(self):
        """处理器异常不应影响其他处理器"""
        def bad_handler(e):
            raise ValueError("Test error")

        self.bus.subscribe(Event, bad_handler, priority=Priority.HIGH)
        self.bus.subscribe(
            Event,
            lambda e: self.received_events.append(e),
            priority=Priority.LOW,
        )

        self.bus.publish(Event())

        # 尽管第一个处理器异常，第二个仍应执行
        assert len(self.received_events) == 1

    def test_history(self):
        self.bus.publish(Event())
        self.bus.publish(OrderEvent())
        self.bus.publish(SignalEvent())

        history = self.bus.get_history()
        assert len(history) == 3

    def test_history_filter_by_type(self):
        self.bus.publish(Event())
        self.bus.publish(OrderEvent())
        self.bus.publish(SignalEvent())

        history = self.bus.get_history(event_type=OrderEvent)
        assert len(history) == 1
        assert isinstance(history[0], OrderEvent)

    def test_history_limit(self):
        for _ in range(10):
            self.bus.publish(Event())

        history = self.bus.get_history(limit=5)
        assert len(history) == 5

    def test_clear_history(self):
        self.bus.publish(Event())
        self.bus.clear_history()

        assert len(self.bus.get_history()) == 0


class TestAsyncEventBus:
    """异步 EventBus 测试"""

    def test_async_publish(self):
        bus = EventBus(async_mode=True, max_workers=2)
        received = []

        bus.subscribe(Event, lambda e: received.append(e))
        bus.publish(Event())

        # 等待异步处理完成
        time.sleep(0.1)

        assert len(received) == 1

        bus.shutdown()


class TestGlobalEventBus:
    """全局事件总线测试"""

    def setup_method(self):
        set_event_bus(EventBus())

    def test_get_event_bus(self):
        bus = get_event_bus()
        assert isinstance(bus, EventBus)

    def test_publish_function(self):
        received = []
        subscribe(Event, lambda e: received.append(e))
        publish(Event())

        assert len(received) == 1

    def test_subscribe_function(self):
        received = []
        subscribe(Event, lambda e: received.append(e))
        get_event_bus().publish(Event())

        assert len(received) == 1


class TestOnEventDecorator:
    """@on_event 装饰器测试"""

    def setup_method(self):
        set_event_bus(EventBus())

    def test_decorator(self):
        received = []

        @on_event(Event)
        def handler(event: Event):
            received.append(event)

        publish(Event())

        assert len(received) == 1

    def test_decorator_with_priority(self):
        results = []

        @on_event(Event, priority=Priority.LOW)
        def low_handler(event: Event):
            results.append("low")

        @on_event(Event, priority=Priority.HIGH)
        def high_handler(event: Event):
            results.append("high")

        publish(Event())

        assert results == ["high", "low"]
