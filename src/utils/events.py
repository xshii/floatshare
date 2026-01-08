"""事件总线模块

提供发布-订阅模式的事件系统：
- 解耦组件间通信
- 支持同步和异步处理
- 事件优先级
- 事件过滤
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import Lock
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="Event")


# ============================================================
# 事件基类
# ============================================================


@dataclass
class Event:
    """事件基类"""
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None

    @property
    def event_type(self) -> str:
        return self.__class__.__name__


@dataclass
class OrderEvent(Event):
    """订单事件"""
    order_id: str = ""
    code: str = ""
    direction: str = ""
    quantity: int = 0
    price: float = 0.0


@dataclass
class OrderFilledEvent(OrderEvent):
    """订单成交事件"""
    fill_price: float = 0.0
    fill_quantity: int = 0
    commission: float = 0.0


@dataclass
class OrderRejectedEvent(OrderEvent):
    """订单拒绝事件"""
    reason: str = ""


@dataclass
class PositionEvent(Event):
    """持仓事件"""
    code: str = ""
    quantity: int = 0
    avg_cost: float = 0.0
    current_price: float = 0.0


@dataclass
class SignalEvent(Event):
    """信号事件"""
    code: str = ""
    direction: str = ""
    strength: float = 0.0
    reason: str = ""


@dataclass
class DataEvent(Event):
    """数据事件"""
    code: str = ""
    data_type: str = ""  # daily, minute, etc.
    rows: int = 0


@dataclass
class SyncProgressEvent(Event):
    """同步进度事件"""
    total: int = 0
    completed: int = 0
    failed: int = 0
    current_code: str = ""

    @property
    def progress_pct(self) -> float:
        return self.completed / self.total * 100 if self.total > 0 else 0


@dataclass
class ErrorEvent(Event):
    """错误事件"""
    error_type: str = ""
    message: str = ""
    details: Optional[Dict[str, Any]] = None


# ============================================================
# 事件处理器
# ============================================================


class Priority(Enum):
    """处理器优先级"""
    HIGHEST = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    LOWEST = 4


@dataclass
class EventHandler:
    """事件处理器包装"""
    callback: Callable[[Event], None]
    priority: Priority = Priority.NORMAL
    filter_fn: Optional[Callable[[Event], bool]] = None
    once: bool = False  # 是否只执行一次

    def should_handle(self, event: Event) -> bool:
        """是否应该处理该事件"""
        if self.filter_fn is None:
            return True
        return self.filter_fn(event)


# ============================================================
# 事件总线
# ============================================================


class EventBus:
    """事件总线"""

    def __init__(self, async_mode: bool = False, max_workers: int = 4):
        """
        Args:
            async_mode: 是否异步处理事件
            max_workers: 异步模式下的最大工作线程数
        """
        self._handlers: Dict[Type[Event], List[EventHandler]] = {}
        self._global_handlers: List[EventHandler] = []
        self._lock = Lock()
        self._async_mode = async_mode
        self._executor = ThreadPoolExecutor(max_workers=max_workers) if async_mode else None
        self._event_history: List[Event] = []
        self._max_history = 1000

    def subscribe(
        self,
        event_type: Type[T],
        callback: Callable[[T], None],
        priority: Priority = Priority.NORMAL,
        filter_fn: Optional[Callable[[T], bool]] = None,
        once: bool = False,
    ) -> "EventBus":
        """
        订阅事件

        Args:
            event_type: 事件类型
            callback: 回调函数
            priority: 优先级
            filter_fn: 过滤函数，返回 True 表示处理
            once: 是否只执行一次

        Example:
            bus.subscribe(OrderFilledEvent, on_order_filled)
            bus.subscribe(OrderEvent, on_any_order, priority=Priority.HIGH)
        """
        handler = EventHandler(
            callback=callback,
            priority=priority,
            filter_fn=filter_fn,
            once=once,
        )

        with self._lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []

            self._handlers[event_type].append(handler)
            # 按优先级排序
            self._handlers[event_type].sort(key=lambda h: h.priority.value)

        return self

    def subscribe_all(
        self,
        callback: Callable[[Event], None],
        priority: Priority = Priority.NORMAL,
    ) -> "EventBus":
        """订阅所有事件"""
        handler = EventHandler(callback=callback, priority=priority)

        with self._lock:
            self._global_handlers.append(handler)
            self._global_handlers.sort(key=lambda h: h.priority.value)

        return self

    def unsubscribe(
        self,
        event_type: Type[Event],
        callback: Callable[[Event], None],
    ) -> bool:
        """取消订阅"""
        with self._lock:
            if event_type not in self._handlers:
                return False

            handlers = self._handlers[event_type]
            for i, h in enumerate(handlers):
                if h.callback == callback:
                    handlers.pop(i)
                    return True

        return False

    def publish(self, event: Event) -> None:
        """
        发布事件

        Args:
            event: 事件对象
        """
        # 记录历史
        self._record_history(event)

        # 获取处理器
        handlers = self._get_handlers(event)

        if not handlers:
            logger.debug(f"事件 {event.event_type} 无处理器")
            return

        # 处理事件
        if self._async_mode and self._executor:
            for handler in handlers:
                self._executor.submit(self._execute_handler, handler, event)
        else:
            self._execute_sync(handlers, event)

    def _get_handlers(self, event: Event) -> List[EventHandler]:
        """获取事件的所有处理器"""
        handlers = []

        with self._lock:
            # 全局处理器
            handlers.extend(self._global_handlers)

            # 特定类型处理器（包括父类）
            for event_type in type(event).__mro__:
                if event_type in self._handlers:
                    handlers.extend(self._handlers[event_type])

                if event_type == Event:
                    break

        # 按优先级排序
        handlers.sort(key=lambda h: h.priority.value)

        return handlers

    def _execute_sync(self, handlers: List[EventHandler], event: Event) -> None:
        """同步执行处理器"""
        handlers_to_remove = []

        for handler in handlers:
            if not handler.should_handle(event):
                continue

            try:
                handler.callback(event)
            except Exception as e:
                logger.error(f"事件处理器异常: {e}", exc_info=True)

            if handler.once:
                handlers_to_remove.append(handler)

        # 移除一次性处理器
        self._remove_handlers(handlers_to_remove)

    def _execute_handler(self, handler: EventHandler, event: Event) -> None:
        """执行单个处理器（异步模式）"""
        if not handler.should_handle(event):
            return

        try:
            handler.callback(event)
        except Exception as e:
            logger.error(f"事件处理器异常: {e}", exc_info=True)

        if handler.once:
            self._remove_handlers([handler])

    def _remove_handlers(self, handlers: List[EventHandler]) -> None:
        """移除处理器"""
        with self._lock:
            for handler in handlers:
                # 从全局处理器移除
                if handler in self._global_handlers:
                    self._global_handlers.remove(handler)
                    continue

                # 从特定类型处理器移除
                for event_handlers in self._handlers.values():
                    if handler in event_handlers:
                        event_handlers.remove(handler)
                        break

    def _record_history(self, event: Event) -> None:
        """记录事件历史"""
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)

    def get_history(
        self,
        event_type: Optional[Type[Event]] = None,
        limit: int = 100,
    ) -> List[Event]:
        """获取事件历史"""
        history = self._event_history[-limit:]

        if event_type:
            history = [e for e in history if isinstance(e, event_type)]

        return history

    def clear_history(self) -> None:
        """清除历史"""
        self._event_history.clear()

    def shutdown(self) -> None:
        """关闭事件总线"""
        if self._executor:
            self._executor.shutdown(wait=True)


# ============================================================
# 装饰器
# ============================================================


def on_event(
    event_type: Type[T],
    priority: Priority = Priority.NORMAL,
    bus: Optional[EventBus] = None,
) -> Callable:
    """
    事件处理器装饰器

    Example:
        @on_event(OrderFilledEvent)
        def handle_order_filled(event: OrderFilledEvent):
            print(f"订单成交: {event.order_id}")
    """
    def decorator(func: Callable[[T], None]) -> Callable[[T], None]:
        _bus = bus or get_event_bus()
        _bus.subscribe(event_type, func, priority)
        return func

    return decorator


# ============================================================
# 全局事件总线
# ============================================================


_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """获取全局事件总线"""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def set_event_bus(bus: EventBus) -> None:
    """设置全局事件总线"""
    global _event_bus
    _event_bus = bus


def publish(event: Event) -> None:
    """发布事件到全局总线"""
    get_event_bus().publish(event)


def subscribe(
    event_type: Type[T],
    callback: Callable[[T], None],
    priority: Priority = Priority.NORMAL,
) -> None:
    """订阅全局总线事件"""
    get_event_bus().subscribe(event_type, callback, priority)
