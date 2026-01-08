"""数据管道模块

提供链式数据处理能力：
- 可组合的处理步骤
- 数据转换和清洗
- 错误处理和重试
- 进度追踪
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

import pandas as pd

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


# ============================================================
# 管道上下文
# ============================================================


@dataclass
class PipelineContext:
    """管道执行上下文"""
    pipeline_name: str = ""
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    current_step: str = ""
    step_index: int = 0
    total_steps: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0
        end = self.finished_at or datetime.now()
        return (end - self.started_at).total_seconds()

    @property
    def is_finished(self) -> bool:
        return self.finished_at is not None

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0


@dataclass
class StepResult(Generic[T]):
    """步骤执行结果"""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    elapsed_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# 管道步骤
# ============================================================


class Step(ABC, Generic[T, R]):
    """管道步骤基类"""

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def process(self, data: T, context: PipelineContext) -> R:
        """处理数据"""
        pass

    def __repr__(self) -> str:
        return f"<{self.name}>"


class FunctionStep(Step[T, R]):
    """函数包装步骤"""

    def __init__(
        self,
        func: Callable[[T], R],
        name: Optional[str] = None,
    ):
        super().__init__(name or func.__name__)
        self.func = func

    def process(self, data: T, context: PipelineContext) -> R:
        return self.func(data)


class FilterStep(Step[pd.DataFrame, pd.DataFrame]):
    """过滤步骤"""

    def __init__(
        self,
        condition: Callable[[pd.DataFrame], pd.Series],
        name: str = "Filter",
    ):
        super().__init__(name)
        self.condition = condition

    def process(self, data: pd.DataFrame, context: PipelineContext) -> pd.DataFrame:
        mask = self.condition(data)
        result = data[mask].copy()
        context.metadata[f"{self.name}_filtered"] = len(data) - len(result)
        return result


class TransformStep(Step[pd.DataFrame, pd.DataFrame]):
    """转换步骤"""

    def __init__(
        self,
        transform: Callable[[pd.DataFrame], pd.DataFrame],
        name: str = "Transform",
    ):
        super().__init__(name)
        self.transform = transform

    def process(self, data: pd.DataFrame, context: PipelineContext) -> pd.DataFrame:
        return self.transform(data)


class ValidateStep(Step[pd.DataFrame, pd.DataFrame]):
    """校验步骤"""

    def __init__(
        self,
        validator: Callable[[pd.DataFrame], bool],
        error_message: str = "Validation failed",
        name: str = "Validate",
    ):
        super().__init__(name)
        self.validator = validator
        self.error_message = error_message

    def process(self, data: pd.DataFrame, context: PipelineContext) -> pd.DataFrame:
        if not self.validator(data):
            raise ValueError(self.error_message)
        return data


class MapStep(Step[pd.DataFrame, pd.DataFrame]):
    """列映射步骤"""

    def __init__(
        self,
        column: str,
        func: Callable[[Any], Any],
        name: Optional[str] = None,
    ):
        super().__init__(name or f"Map_{column}")
        self.column = column
        self.func = func

    def process(self, data: pd.DataFrame, context: PipelineContext) -> pd.DataFrame:
        result = data.copy()
        result[self.column] = result[self.column].apply(self.func)
        return result


class AddColumnStep(Step[pd.DataFrame, pd.DataFrame]):
    """添加列步骤"""

    def __init__(
        self,
        column: str,
        func: Callable[[pd.DataFrame], pd.Series],
        name: Optional[str] = None,
    ):
        super().__init__(name or f"AddColumn_{column}")
        self.column = column
        self.func = func

    def process(self, data: pd.DataFrame, context: PipelineContext) -> pd.DataFrame:
        result = data.copy()
        result[self.column] = self.func(data)
        return result


class DropColumnsStep(Step[pd.DataFrame, pd.DataFrame]):
    """删除列步骤"""

    def __init__(self, columns: List[str], name: str = "DropColumns"):
        super().__init__(name)
        self.columns = columns

    def process(self, data: pd.DataFrame, context: PipelineContext) -> pd.DataFrame:
        cols_to_drop = [c for c in self.columns if c in data.columns]
        return data.drop(columns=cols_to_drop)


class RenameColumnsStep(Step[pd.DataFrame, pd.DataFrame]):
    """重命名列步骤"""

    def __init__(self, mapping: Dict[str, str], name: str = "RenameColumns"):
        super().__init__(name)
        self.mapping = mapping

    def process(self, data: pd.DataFrame, context: PipelineContext) -> pd.DataFrame:
        return data.rename(columns=self.mapping)


class SortStep(Step[pd.DataFrame, pd.DataFrame]):
    """排序步骤"""

    def __init__(
        self,
        by: Union[str, List[str]],
        ascending: bool = True,
        name: str = "Sort",
    ):
        super().__init__(name)
        self.by = by
        self.ascending = ascending

    def process(self, data: pd.DataFrame, context: PipelineContext) -> pd.DataFrame:
        return data.sort_values(by=self.by, ascending=self.ascending)


class DeduplicateStep(Step[pd.DataFrame, pd.DataFrame]):
    """去重步骤"""

    def __init__(
        self,
        subset: Optional[List[str]] = None,
        keep: str = "last",
        name: str = "Deduplicate",
    ):
        super().__init__(name)
        self.subset = subset
        self.keep = keep

    def process(self, data: pd.DataFrame, context: PipelineContext) -> pd.DataFrame:
        before = len(data)
        result = data.drop_duplicates(subset=self.subset, keep=self.keep)
        context.metadata[f"{self.name}_removed"] = before - len(result)
        return result


class FillNaStep(Step[pd.DataFrame, pd.DataFrame]):
    """填充空值步骤"""

    def __init__(
        self,
        value: Any = None,
        method: Optional[str] = None,
        columns: Optional[List[str]] = None,
        name: str = "FillNa",
    ):
        super().__init__(name)
        self.value = value
        self.method = method
        self.columns = columns

    def process(self, data: pd.DataFrame, context: PipelineContext) -> pd.DataFrame:
        result = data.copy()

        if self.columns:
            for col in self.columns:
                if col in result.columns:
                    if self.method:
                        result[col] = result[col].fillna(method=self.method)
                    else:
                        result[col] = result[col].fillna(self.value)
        else:
            if self.method:
                result = result.fillna(method=self.method)
            else:
                result = result.fillna(self.value)

        return result


# ============================================================
# 管道类
# ============================================================


class Pipeline(Generic[T, R]):
    """数据管道"""

    def __init__(self, name: str = "Pipeline"):
        self.name = name
        self.steps: List[Step] = []
        self._on_step_complete: Optional[Callable[[str, int, int], None]] = None
        self._on_error: Optional[Callable[[str, Exception], None]] = None

    def add(self, step: Step) -> "Pipeline":
        """添加步骤"""
        self.steps.append(step)
        return self

    def add_function(
        self,
        func: Callable,
        name: Optional[str] = None,
    ) -> "Pipeline":
        """添加函数步骤"""
        return self.add(FunctionStep(func, name))

    def filter(
        self,
        condition: Callable[[pd.DataFrame], pd.Series],
        name: str = "Filter",
    ) -> "Pipeline":
        """添加过滤步骤"""
        return self.add(FilterStep(condition, name))

    def transform(
        self,
        transform: Callable[[pd.DataFrame], pd.DataFrame],
        name: str = "Transform",
    ) -> "Pipeline":
        """添加转换步骤"""
        return self.add(TransformStep(transform, name))

    def validate(
        self,
        validator: Callable[[pd.DataFrame], bool],
        error_message: str = "Validation failed",
        name: str = "Validate",
    ) -> "Pipeline":
        """添加校验步骤"""
        return self.add(ValidateStep(validator, error_message, name))

    def map_column(
        self,
        column: str,
        func: Callable[[Any], Any],
    ) -> "Pipeline":
        """添加列映射"""
        return self.add(MapStep(column, func))

    def add_column(
        self,
        column: str,
        func: Callable[[pd.DataFrame], pd.Series],
    ) -> "Pipeline":
        """添加新列"""
        return self.add(AddColumnStep(column, func))

    def drop_columns(self, columns: List[str]) -> "Pipeline":
        """删除列"""
        return self.add(DropColumnsStep(columns))

    def rename_columns(self, mapping: Dict[str, str]) -> "Pipeline":
        """重命名列"""
        return self.add(RenameColumnsStep(mapping))

    def sort(
        self,
        by: Union[str, List[str]],
        ascending: bool = True,
    ) -> "Pipeline":
        """排序"""
        return self.add(SortStep(by, ascending))

    def deduplicate(
        self,
        subset: Optional[List[str]] = None,
        keep: str = "last",
    ) -> "Pipeline":
        """去重"""
        return self.add(DeduplicateStep(subset, keep))

    def fill_na(
        self,
        value: Any = None,
        method: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> "Pipeline":
        """填充空值"""
        return self.add(FillNaStep(value, method, columns))

    def on_step_complete(
        self,
        callback: Callable[[str, int, int], None],
    ) -> "Pipeline":
        """设置步骤完成回调"""
        self._on_step_complete = callback
        return self

    def on_error(
        self,
        callback: Callable[[str, Exception], None],
    ) -> "Pipeline":
        """设置错误回调"""
        self._on_error = callback
        return self

    def execute(self, data: T) -> StepResult[R]:
        """
        执行管道

        Args:
            data: 输入数据

        Returns:
            StepResult 包含执行结果
        """
        context = PipelineContext(
            pipeline_name=self.name,
            started_at=datetime.now(),
            total_steps=len(self.steps),
        )

        current_data = data
        start_time = time.time()

        try:
            for i, step in enumerate(self.steps):
                context.step_index = i
                context.current_step = step.name

                step_start = time.time()

                try:
                    current_data = step.process(current_data, context)
                except Exception as e:
                    logger.error(f"管道 {self.name} 步骤 {step.name} 失败: {e}")
                    context.errors.append(f"{step.name}: {str(e)}")

                    if self._on_error:
                        self._on_error(step.name, e)

                    context.finished_at = datetime.now()
                    return StepResult(
                        success=False,
                        error=str(e),
                        elapsed_ms=(time.time() - start_time) * 1000,
                        metadata=context.metadata,
                    )

                step_elapsed = (time.time() - step_start) * 1000
                logger.debug(f"步骤 {step.name} 完成，耗时 {step_elapsed:.2f}ms")

                if self._on_step_complete:
                    self._on_step_complete(step.name, i + 1, len(self.steps))

            context.finished_at = datetime.now()

            return StepResult(
                success=True,
                data=current_data,
                elapsed_ms=(time.time() - start_time) * 1000,
                metadata=context.metadata,
            )

        except Exception as e:
            logger.error(f"管道 {self.name} 异常: {e}")
            context.finished_at = datetime.now()
            return StepResult(
                success=False,
                error=str(e),
                elapsed_ms=(time.time() - start_time) * 1000,
                metadata=context.metadata,
            )

    def __repr__(self) -> str:
        steps_str = " -> ".join(s.name for s in self.steps)
        return f"<Pipeline '{self.name}': {steps_str}>"


# ============================================================
# 预定义管道
# ============================================================


def create_daily_data_pipeline(code: str = "") -> Pipeline:
    """创建日线数据处理管道"""
    return (
        Pipeline(f"DailyData_{code}")
        .validate(
            lambda df: not df.empty,
            error_message="数据为空",
            name="NotEmpty",
        )
        .validate(
            lambda df: all(c in df.columns for c in ["open", "high", "low", "close"]),
            error_message="缺少必要列",
            name="RequiredColumns",
        )
        .deduplicate(subset=["trade_date"] if "trade_date" in [] else None)
        .sort("trade_date")
        .filter(
            lambda df: df["high"] >= df["low"],
            name="ValidHighLow",
        )
        .filter(
            lambda df: (df["open"] >= df["low"]) & (df["open"] <= df["high"]),
            name="ValidOpen",
        )
        .filter(
            lambda df: (df["close"] >= df["low"]) & (df["close"] <= df["high"]),
            name="ValidClose",
        )
    )


def create_clean_pipeline() -> Pipeline:
    """创建数据清洗管道"""
    return (
        Pipeline("CleanData")
        .deduplicate()
        .fill_na(method="ffill")
        .fill_na(value=0)  # 剩余空值填0
    )
