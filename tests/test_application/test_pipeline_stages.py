"""stage fn 单测 — 不触网络, 只验证 retry / success / 全失败 分支."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from floatshare.application.news_ingest import IngestResult
from floatshare.application.pipeline.stages import stage_s1c_news_ingest
from floatshare.domain.pipeline import StageContext


class _FakeSource:
    """占位 TushareSource — 实际 ingest 被 monkeypatch, 不会调任何方法."""


@pytest.fixture
def fake_stage_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """替掉 TushareSource + time.sleep + ingest 主函数, 返回 calls list."""
    calls: list[dict[str, Any]] = []
    sleeps: list[float] = []

    monkeypatch.setattr(
        "floatshare.application.pipeline.stages.time.sleep",
        sleeps.append,
    )
    # stage fn 里 `from floatshare.infrastructure.data_sources.tushare import TushareSource`
    # 是函数内 import, patch 原模块即可
    monkeypatch.setattr(
        "floatshare.infrastructure.data_sources.tushare.TushareSource",
        _FakeSource,
    )
    return calls, sleeps


def _ctx(tmp_path: Path) -> StageContext:
    return StageContext(trade_date="2026-04-21", db_path=tmp_path / "db.sqlite")


def test_s1c_first_attempt_success(monkeypatch: pytest.MonkeyPatch, fake_stage_env, tmp_path: Path):
    calls, sleeps = fake_stage_env

    def fake_ingest(td, src, db):
        calls.append({"td": td, "db": db})
        return IngestResult("2026-04-21", raw_rows=12, mentions=5, text_length=3200, success=True)

    monkeypatch.setattr(
        "floatshare.application.pipeline.stages.ingest_cctv_news_for_date",
        fake_ingest,
        raising=False,
    )
    # 注入到 stages 模块的命名空间 (stage fn 是 local import, 需 patch 源模块)
    monkeypatch.setattr(
        "floatshare.application.news_ingest.ingest_cctv_news_for_date",
        fake_ingest,
    )

    extras = stage_s1c_news_ingest(_ctx(tmp_path), max_attempts=4, retry_interval_sec=0)
    assert extras == {"attempts": 1, "raw_rows": 12, "mentions": 5, "text_length": 3200}
    assert len(calls) == 1
    assert sleeps == []  # 首次成功, 不 sleep


def test_s1c_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch, fake_stage_env, tmp_path: Path):
    calls, sleeps = fake_stage_env
    results = iter(
        [
            IngestResult("2026-04-21", 0, 0, 0, success=False, error="empty_response"),
            IngestResult("2026-04-21", 0, 0, 0, success=False, error="empty_response"),
            IngestResult("2026-04-21", raw_rows=8, mentions=3, text_length=2100, success=True),
        ]
    )

    def fake_ingest(td, src, db):
        calls.append(1)
        return next(results)

    monkeypatch.setattr("floatshare.application.news_ingest.ingest_cctv_news_for_date", fake_ingest)

    extras = stage_s1c_news_ingest(_ctx(tmp_path), max_attempts=4, retry_interval_sec=13)
    assert extras == {"attempts": 3, "raw_rows": 8, "mentions": 3, "text_length": 2100}
    assert len(calls) == 3
    assert sleeps == [13, 13]  # 前两次失败各 sleep 一次


def test_s1c_all_attempts_fail_raises(
    monkeypatch: pytest.MonkeyPatch, fake_stage_env, tmp_path: Path
):
    calls, sleeps = fake_stage_env

    def fake_ingest(td, src, db):
        calls.append(1)
        return IngestResult("2026-04-21", 0, 0, 0, success=False, error="empty_response")

    monkeypatch.setattr("floatshare.application.news_ingest.ingest_cctv_news_for_date", fake_ingest)

    with pytest.raises(RuntimeError, match="empty_response"):
        stage_s1c_news_ingest(_ctx(tmp_path), max_attempts=3, retry_interval_sec=0)
    assert len(calls) == 3
    assert len(sleeps) == 2  # 3 次尝试之间 sleep 2 次 (最后一次失败后不 sleep)


def test_s1c_in_stage_gates_is_callable_tuple():
    """STAGE_GATES[S1C_NEWS_INGEST] 必须注册, 即便是空 tuple."""
    from floatshare.application.pipeline.gates import STAGE_GATES
    from floatshare.domain.enums import PipelineStage

    assert PipelineStage.S1C_NEWS_INGEST in STAGE_GATES
    assert isinstance(STAGE_GATES[PipelineStage.S1C_NEWS_INGEST], tuple)
