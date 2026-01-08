"""分布式追踪模块

基于 OpenTelemetry 提供追踪能力：
- 请求追踪
- 性能分析
- 错误追踪
"""

import logging
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.trace import Status, StatusCode, Span
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================
# 配置
# ============================================================


def configure_tracing(
    service_name: str = "floatshare",
    service_version: str = "1.0.0",
    console_export: bool = True,
    otlp_endpoint: Optional[str] = None,
) -> None:
    """
    配置分布式追踪

    Args:
        service_name: 服务名称
        service_version: 服务版本
        console_export: 是否输出到控制台
        otlp_endpoint: OTLP 端点 (如 "http://localhost:4317")
    """
    if not HAS_OTEL:
        logger.warning("OpenTelemetry 未安装，追踪功能不可用")
        return

    resource = Resource.create({
        "service.name": service_name,
        "service.version": service_version,
    })

    provider = TracerProvider(resource=resource)

    # 控制台导出器
    if console_export:
        console_processor = BatchSpanProcessor(ConsoleSpanExporter())
        provider.add_span_processor(console_processor)

    # OTLP 导出器
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            otlp_processor = BatchSpanProcessor(otlp_exporter)
            provider.add_span_processor(otlp_processor)
        except ImportError:
            logger.warning("OTLP exporter 未安装")

    trace.set_tracer_provider(provider)
    logger.info(f"追踪已配置: {service_name} v{service_version}")


def get_tracer(name: str = "floatshare") -> Any:
    """获取 Tracer"""
    if not HAS_OTEL:
        return _NoOpTracer()
    return trace.get_tracer(name)


# ============================================================
# 追踪装饰器
# ============================================================


def traced(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
):
    """
    函数追踪装饰器

    Example:
        @traced("process_data", {"data_type": "daily"})
        def process_data(code: str) -> pd.DataFrame:
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        span_name = name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            tracer = get_tracer()

            with tracer.start_as_current_span(span_name) as span:
                # 添加属性
                if attributes:
                    for k, v in attributes.items():
                        span.set_attribute(k, v)

                # 添加函数参数
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)

                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper
    return decorator


@contextmanager
def trace_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
):
    """
    追踪上下文管理器

    Example:
        with trace_span("sync_data", {"source": "akshare"}):
            sync_data()
    """
    tracer = get_tracer()

    with tracer.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)

        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


def add_span_attribute(key: str, value: Any) -> None:
    """添加当前 Span 属性"""
    if not HAS_OTEL:
        return

    span = trace.get_current_span()
    if span:
        span.set_attribute(key, value)


def add_span_event(name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
    """添加当前 Span 事件"""
    if not HAS_OTEL:
        return

    span = trace.get_current_span()
    if span:
        span.add_event(name, attributes=attributes)


# ============================================================
# 业务追踪
# ============================================================


class DataSyncTracer:
    """数据同步追踪器"""

    def __init__(self):
        self.tracer = get_tracer("data_sync")

    @contextmanager
    def trace_sync(
        self,
        source: str,
        codes_count: int,
    ):
        """追踪同步任务"""
        with self.tracer.start_as_current_span("data_sync") as span:
            span.set_attribute("sync.source", source)
            span.set_attribute("sync.codes_count", codes_count)
            yield span

    @contextmanager
    def trace_code_sync(
        self,
        code: str,
        source: str,
    ):
        """追踪单只股票同步"""
        with self.tracer.start_as_current_span("sync_code") as span:
            span.set_attribute("stock.code", code)
            span.set_attribute("sync.source", source)
            yield span


class BacktestTracer:
    """回测追踪器"""

    def __init__(self):
        self.tracer = get_tracer("backtest")

    @contextmanager
    def trace_backtest(
        self,
        strategy: str,
        start_date: str,
        end_date: str,
    ):
        """追踪回测任务"""
        with self.tracer.start_as_current_span("backtest") as span:
            span.set_attribute("backtest.strategy", strategy)
            span.set_attribute("backtest.start_date", start_date)
            span.set_attribute("backtest.end_date", end_date)
            yield span

    @contextmanager
    def trace_trade(
        self,
        code: str,
        direction: str,
    ):
        """追踪交易"""
        with self.tracer.start_as_current_span("trade") as span:
            span.set_attribute("trade.code", code)
            span.set_attribute("trade.direction", direction)
            yield span


# ============================================================
# NoOp 实现 (OpenTelemetry 未安装时使用)
# ============================================================


class _NoOpSpan:
    """空操作 Span"""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _NoOpTracer:
    """空操作 Tracer"""

    def start_as_current_span(self, name: str, **kwargs):
        return _NoOpSpan()

    def start_span(self, name: str, **kwargs):
        return _NoOpSpan()


# 兼容性别名
if HAS_OTEL:
    Status = Status
    StatusCode = StatusCode
else:
    class Status:
        def __init__(self, code, description=None):
            pass

    class StatusCode:
        OK = "OK"
        ERROR = "ERROR"
