"""可观测性模块

提供 Prometheus 风格的指标收集：
- Counter: 计数器
- Gauge: 仪表盘
- Histogram: 直方图
- 指标导出 (HTTP/文件)
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# 指标类型
# ============================================================


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """指标值"""
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)


class Metric(ABC):
    """指标基类"""

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._lock = threading.Lock()

    @abstractmethod
    def get_type(self) -> MetricType:
        pass

    @abstractmethod
    def get_samples(self) -> List[MetricValue]:
        pass

    def _validate_labels(self, labels: Dict[str, str]) -> None:
        """验证标签"""
        for name in labels:
            if name not in self.label_names:
                raise ValueError(f"未知标签: {name}")


class Counter(Metric):
    """计数器 - 只增不减"""

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ):
        super().__init__(name, description, labels)
        self._values: Dict[Tuple, float] = defaultdict(float)

    def get_type(self) -> MetricType:
        return MetricType.COUNTER

    def inc(self, value: float = 1, **labels) -> None:
        """增加计数"""
        if value < 0:
            raise ValueError("Counter 不能减少")

        self._validate_labels(labels)
        key = tuple(sorted(labels.items()))

        with self._lock:
            self._values[key] += value

    def get(self, **labels) -> float:
        """获取当前值"""
        key = tuple(sorted(labels.items()))
        return self._values.get(key, 0)

    def get_samples(self) -> List[MetricValue]:
        samples = []
        with self._lock:
            for key, value in self._values.items():
                labels = dict(key)
                samples.append(MetricValue(value=value, labels=labels))
        return samples


class Gauge(Metric):
    """仪表盘 - 可增可减"""

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ):
        super().__init__(name, description, labels)
        self._values: Dict[Tuple, float] = {}

    def get_type(self) -> MetricType:
        return MetricType.GAUGE

    def set(self, value: float, **labels) -> None:
        """设置值"""
        self._validate_labels(labels)
        key = tuple(sorted(labels.items()))

        with self._lock:
            self._values[key] = value

    def inc(self, value: float = 1, **labels) -> None:
        """增加"""
        self._validate_labels(labels)
        key = tuple(sorted(labels.items()))

        with self._lock:
            self._values[key] = self._values.get(key, 0) + value

    def dec(self, value: float = 1, **labels) -> None:
        """减少"""
        self.inc(-value, **labels)

    def get(self, **labels) -> float:
        """获取当前值"""
        key = tuple(sorted(labels.items()))
        return self._values.get(key, 0)

    def get_samples(self) -> List[MetricValue]:
        samples = []
        with self._lock:
            for key, value in self._values.items():
                labels = dict(key)
                samples.append(MetricValue(value=value, labels=labels))
        return samples


class Histogram(Metric):
    """直方图 - 分布统计"""

    DEFAULT_BUCKETS = (
        0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5,
        0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float("inf"),
    )

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        buckets: Optional[Tuple[float, ...]] = None,
    ):
        super().__init__(name, description, labels)
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self._counts: Dict[Tuple, Dict[float, int]] = defaultdict(
            lambda: {b: 0 for b in self.buckets}
        )
        self._sums: Dict[Tuple, float] = defaultdict(float)
        self._totals: Dict[Tuple, int] = defaultdict(int)

    def get_type(self) -> MetricType:
        return MetricType.HISTOGRAM

    def observe(self, value: float, **labels) -> None:
        """记录观察值"""
        self._validate_labels(labels)
        key = tuple(sorted(labels.items()))

        with self._lock:
            self._sums[key] += value
            self._totals[key] += 1

            for bucket in self.buckets:
                if value <= bucket:
                    self._counts[key][bucket] += 1

    def get_samples(self) -> List[MetricValue]:
        samples = []

        with self._lock:
            for key, counts in self._counts.items():
                labels = dict(key)

                # 桶计数
                cumulative = 0
                for bucket in self.buckets:
                    cumulative += counts[bucket]
                    bucket_labels = {**labels, "le": str(bucket)}
                    samples.append(MetricValue(
                        value=cumulative,
                        labels=bucket_labels,
                    ))

                # 总和
                samples.append(MetricValue(
                    value=self._sums[key],
                    labels={**labels, "_type": "sum"},
                ))

                # 计数
                samples.append(MetricValue(
                    value=self._totals[key],
                    labels={**labels, "_type": "count"},
                ))

        return samples


# ============================================================
# 指标注册表
# ============================================================


class MetricsRegistry:
    """指标注册表"""

    def __init__(self, prefix: str = "floatshare"):
        self.prefix = prefix
        self._metrics: Dict[str, Metric] = {}
        self._lock = threading.Lock()

    def _full_name(self, name: str) -> str:
        """生成完整指标名"""
        return f"{self.prefix}_{name}"

    def counter(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ) -> Counter:
        """创建或获取 Counter"""
        full_name = self._full_name(name)

        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Counter(full_name, description, labels)
            return self._metrics[full_name]

    def gauge(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ) -> Gauge:
        """创建或获取 Gauge"""
        full_name = self._full_name(name)

        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Gauge(full_name, description, labels)
            return self._metrics[full_name]

    def histogram(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        buckets: Optional[Tuple[float, ...]] = None,
    ) -> Histogram:
        """创建或获取 Histogram"""
        full_name = self._full_name(name)

        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Histogram(
                    full_name, description, labels, buckets
                )
            return self._metrics[full_name]

    def get_all_metrics(self) -> Dict[str, Metric]:
        """获取所有指标"""
        return dict(self._metrics)

    def export_prometheus(self) -> str:
        """导出为 Prometheus 格式"""
        lines = []

        for name, metric in self._metrics.items():
            # HELP
            if metric.description:
                lines.append(f"# HELP {name} {metric.description}")

            # TYPE
            lines.append(f"# TYPE {name} {metric.get_type().value}")

            # 值
            for sample in metric.get_samples():
                if sample.labels:
                    label_str = ",".join(
                        f'{k}="{v}"' for k, v in sample.labels.items()
                    )
                    lines.append(f"{name}{{{label_str}}} {sample.value}")
                else:
                    lines.append(f"{name} {sample.value}")

        return "\n".join(lines)

    def export_json(self) -> Dict[str, Any]:
        """导出为 JSON 格式"""
        result = {}

        for name, metric in self._metrics.items():
            samples = []
            for sample in metric.get_samples():
                samples.append({
                    "value": sample.value,
                    "labels": sample.labels,
                    "timestamp": sample.timestamp,
                })
            result[name] = {
                "type": metric.get_type().value,
                "description": metric.description,
                "samples": samples,
            }

        return result


# ============================================================
# 装饰器
# ============================================================


def timed(
    histogram: Histogram,
    **labels,
) -> Callable:
    """
    计时装饰器

    Example:
        request_latency = metrics.histogram("http_request_latency_seconds")

        @timed(request_latency, method="GET")
        def handle_request():
            ...
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.time() - start
                histogram.observe(elapsed, **labels)

        return wrapper
    return decorator


def counted(
    counter: Counter,
    **labels,
) -> Callable:
    """
    计数装饰器

    Example:
        request_count = metrics.counter("http_requests_total")

        @counted(request_count, method="GET")
        def handle_request():
            ...
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            counter.inc(**labels)
            return func(*args, **kwargs)

        return wrapper
    return decorator


# ============================================================
# 预定义指标
# ============================================================


class FloatShareMetrics:
    """FloatShare 预定义指标"""

    def __init__(self, registry: Optional[MetricsRegistry] = None):
        self.registry = registry or MetricsRegistry()

        # 数据同步指标
        self.sync_total = self.registry.counter(
            "data_sync_total",
            "数据同步总次数",
            labels=["source", "status"],
        )

        self.sync_rows = self.registry.counter(
            "data_sync_rows_total",
            "同步数据行数",
            labels=["source"],
        )

        self.sync_duration = self.registry.histogram(
            "data_sync_duration_seconds",
            "同步耗时",
            labels=["source"],
        )

        # 缓存指标
        self.cache_hits = self.registry.counter(
            "cache_hits_total",
            "缓存命中次数",
        )

        self.cache_misses = self.registry.counter(
            "cache_misses_total",
            "缓存未命中次数",
        )

        self.cache_size = self.registry.gauge(
            "cache_size",
            "缓存大小",
        )

        # 数据库指标
        self.db_queries = self.registry.counter(
            "db_queries_total",
            "数据库查询次数",
            labels=["operation"],
        )

        self.db_query_duration = self.registry.histogram(
            "db_query_duration_seconds",
            "数据库查询耗时",
            labels=["operation"],
        )

        # 回测指标
        self.backtest_runs = self.registry.counter(
            "backtest_runs_total",
            "回测运行次数",
            labels=["strategy"],
        )

        self.backtest_duration = self.registry.histogram(
            "backtest_duration_seconds",
            "回测耗时",
            labels=["strategy"],
        )


# ============================================================
# 全局注册表
# ============================================================

_registry: Optional[MetricsRegistry] = None
_metrics: Optional[FloatShareMetrics] = None


def get_registry() -> MetricsRegistry:
    """获取全局指标注册表"""
    global _registry
    if _registry is None:
        _registry = MetricsRegistry()
    return _registry


def get_metrics() -> FloatShareMetrics:
    """获取预定义指标"""
    global _metrics
    if _metrics is None:
        _metrics = FloatShareMetrics(get_registry())
    return _metrics
