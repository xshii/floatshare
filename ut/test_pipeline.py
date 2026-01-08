"""数据管道测试"""

import pandas as pd
import pytest

from src.data.pipeline import (
    AddColumnStep,
    DeduplicateStep,
    DropColumnsStep,
    FillNaStep,
    FilterStep,
    FunctionStep,
    MapStep,
    Pipeline,
    PipelineContext,
    RenameColumnsStep,
    SortStep,
    StepResult,
    TransformStep,
    ValidateStep,
    create_clean_pipeline,
    create_daily_data_pipeline,
)


class TestPipelineContext:
    """PipelineContext 测试"""

    def test_initial_state(self):
        ctx = PipelineContext(pipeline_name="test")
        assert ctx.pipeline_name == "test"
        assert ctx.started_at is None
        assert not ctx.is_finished
        assert not ctx.has_errors

    def test_elapsed_seconds(self):
        from datetime import datetime, timedelta

        ctx = PipelineContext()
        ctx.started_at = datetime.now() - timedelta(seconds=5)
        assert ctx.elapsed_seconds >= 5

    def test_has_errors(self):
        ctx = PipelineContext()
        assert not ctx.has_errors

        ctx.errors.append("error")
        assert ctx.has_errors


class TestStepResult:
    """StepResult 测试"""

    def test_success_result(self):
        result = StepResult(success=True, data="test")
        assert result.success
        assert result.data == "test"
        assert result.error is None

    def test_error_result(self):
        result = StepResult(success=False, error="failed")
        assert not result.success
        assert result.error == "failed"


class TestFunctionStep:
    """FunctionStep 测试"""

    def test_basic_function(self):
        step = FunctionStep(lambda x: x * 2)
        ctx = PipelineContext()

        result = step.process(5, ctx)
        assert result == 10

    def test_with_name(self):
        step = FunctionStep(lambda x: x, name="MyStep")
        assert step.name == "MyStep"


class TestFilterStep:
    """FilterStep 测试"""

    def test_filter(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        step = FilterStep(lambda df: df["a"] > 2)
        ctx = PipelineContext()

        result = step.process(df, ctx)
        assert len(result) == 3
        assert list(result["a"]) == [3, 4, 5]

    def test_filter_records_metadata(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        step = FilterStep(lambda df: df["a"] > 2, name="MyFilter")
        ctx = PipelineContext()

        step.process(df, ctx)
        assert ctx.metadata["MyFilter_filtered"] == 2


class TestTransformStep:
    """TransformStep 测试"""

    def test_transform(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        step = TransformStep(lambda df: df.assign(b=df["a"] * 2))
        ctx = PipelineContext()

        result = step.process(df, ctx)
        assert "b" in result.columns
        assert list(result["b"]) == [2, 4, 6]


class TestValidateStep:
    """ValidateStep 测试"""

    def test_validate_pass(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        step = ValidateStep(lambda df: len(df) > 0)
        ctx = PipelineContext()

        result = step.process(df, ctx)
        pd.testing.assert_frame_equal(result, df)

    def test_validate_fail(self):
        df = pd.DataFrame()
        step = ValidateStep(lambda df: len(df) > 0, error_message="Empty!")
        ctx = PipelineContext()

        with pytest.raises(ValueError, match="Empty!"):
            step.process(df, ctx)


class TestMapStep:
    """MapStep 测试"""

    def test_map(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        step = MapStep("a", lambda x: x * 10)
        ctx = PipelineContext()

        result = step.process(df, ctx)
        assert list(result["a"]) == [10, 20, 30]


class TestAddColumnStep:
    """AddColumnStep 测试"""

    def test_add_column(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        step = AddColumnStep("b", lambda df: df["a"] * 2)
        ctx = PipelineContext()

        result = step.process(df, ctx)
        assert "b" in result.columns
        assert list(result["b"]) == [2, 4, 6]


class TestDropColumnsStep:
    """DropColumnsStep 测试"""

    def test_drop_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        step = DropColumnsStep(["b", "c"])
        ctx = PipelineContext()

        result = step.process(df, ctx)
        assert list(result.columns) == ["a"]

    def test_drop_nonexistent_column(self):
        df = pd.DataFrame({"a": [1]})
        step = DropColumnsStep(["b"])  # 不存在的列
        ctx = PipelineContext()

        result = step.process(df, ctx)  # 不应报错
        assert list(result.columns) == ["a"]


class TestRenameColumnsStep:
    """RenameColumnsStep 测试"""

    def test_rename(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        step = RenameColumnsStep({"a": "x", "b": "y"})
        ctx = PipelineContext()

        result = step.process(df, ctx)
        assert list(result.columns) == ["x", "y"]


class TestSortStep:
    """SortStep 测试"""

    def test_sort_ascending(self):
        df = pd.DataFrame({"a": [3, 1, 2]})
        step = SortStep("a", ascending=True)
        ctx = PipelineContext()

        result = step.process(df, ctx)
        assert list(result["a"]) == [1, 2, 3]

    def test_sort_descending(self):
        df = pd.DataFrame({"a": [3, 1, 2]})
        step = SortStep("a", ascending=False)
        ctx = PipelineContext()

        result = step.process(df, ctx)
        assert list(result["a"]) == [3, 2, 1]


class TestDeduplicateStep:
    """DeduplicateStep 测试"""

    def test_deduplicate(self):
        df = pd.DataFrame({"a": [1, 1, 2, 2, 3]})
        step = DeduplicateStep()
        ctx = PipelineContext()

        result = step.process(df, ctx)
        assert len(result) == 3

    def test_deduplicate_subset(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 2, 3]})
        step = DeduplicateStep(subset=["a"], keep="first")
        ctx = PipelineContext()

        result = step.process(df, ctx)
        assert len(result) == 2

    def test_deduplicate_records_metadata(self):
        df = pd.DataFrame({"a": [1, 1, 1]})
        step = DeduplicateStep(name="MyDedup")
        ctx = PipelineContext()

        step.process(df, ctx)
        assert ctx.metadata["MyDedup_removed"] == 2


class TestFillNaStep:
    """FillNaStep 测试"""

    def test_fill_value(self):
        df = pd.DataFrame({"a": [1, None, 3]})
        step = FillNaStep(value=0)
        ctx = PipelineContext()

        result = step.process(df, ctx)
        assert list(result["a"]) == [1, 0, 3]

    def test_fill_specific_columns(self):
        df = pd.DataFrame({"a": [1, None], "b": [None, 2]})
        step = FillNaStep(value=0, columns=["a"])
        ctx = PipelineContext()

        result = step.process(df, ctx)
        assert result["a"].isna().sum() == 0
        assert result["b"].isna().sum() == 1


class TestPipeline:
    """Pipeline 测试"""

    def test_basic_pipeline(self):
        pipeline = (
            Pipeline("test")
            .add_function(lambda x: x * 2)
            .add_function(lambda x: x + 1)
        )

        result = pipeline.execute(5)

        assert result.success
        assert result.data == 11  # (5 * 2) + 1

    def test_dataframe_pipeline(self):
        df = pd.DataFrame({"a": [3, 1, 2]})

        pipeline = (
            Pipeline("test")
            .sort("a")
            .add_column("b", lambda df: df["a"] * 2)
        )

        result = pipeline.execute(df)

        assert result.success
        assert list(result.data["a"]) == [1, 2, 3]
        assert list(result.data["b"]) == [2, 4, 6]

    def test_pipeline_error_handling(self):
        pipeline = (
            Pipeline("test")
            .validate(lambda df: False, error_message="Always fails")
        )

        result = pipeline.execute(pd.DataFrame())

        assert not result.success
        assert "Always fails" in result.error

    def test_pipeline_error_callback(self):
        errors = []

        def on_error(step, exception):
            errors.append((step, str(exception)))

        pipeline = (
            Pipeline("test")
            .on_error(on_error)
            .validate(lambda df: False, error_message="Test error")
        )

        pipeline.execute(pd.DataFrame())

        assert len(errors) == 1
        assert errors[0][0] == "Validate"

    def test_pipeline_step_complete_callback(self):
        completed = []

        def on_complete(step, current, total):
            completed.append((step, current, total))

        pipeline = (
            Pipeline("test")
            .on_step_complete(on_complete)
            .add_function(lambda x: x, name="Step1")
            .add_function(lambda x: x, name="Step2")
        )

        pipeline.execute(1)

        assert len(completed) == 2
        assert completed[0] == ("Step1", 1, 2)
        assert completed[1] == ("Step2", 2, 2)

    def test_pipeline_fluent_api(self):
        pipeline = (
            Pipeline("test")
            .filter(lambda df: df["a"] > 0)
            .transform(lambda df: df)
            .map_column("a", lambda x: x)
            .drop_columns(["b"])
            .rename_columns({"a": "x"})
            .sort("x")
            .deduplicate()
            .fill_na(0)
        )

        assert len(pipeline.steps) == 8

    def test_pipeline_repr(self):
        pipeline = (
            Pipeline("test")
            .add_function(lambda x: x, name="A")
            .add_function(lambda x: x, name="B")
        )

        repr_str = repr(pipeline)
        assert "test" in repr_str
        assert "A" in repr_str
        assert "B" in repr_str

    def test_pipeline_elapsed_time(self):
        import time

        pipeline = Pipeline("test").add_function(lambda x: (time.sleep(0.01), x)[1])

        result = pipeline.execute(1)

        assert result.elapsed_ms >= 10


class TestPredefinedPipelines:
    """预定义管道测试"""

    def test_create_clean_pipeline(self):
        pipeline = create_clean_pipeline()
        assert pipeline.name == "CleanData"
        assert len(pipeline.steps) > 0

    def test_create_daily_data_pipeline(self):
        pipeline = create_daily_data_pipeline("000001")
        assert "000001" in pipeline.name
        assert len(pipeline.steps) > 0
