"""回测盲点: T 日选中 → T+1 停牌 的概率测量 (eval.count_recommend_then_suspended)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from floatshare.ml.eval import count_recommend_then_suspended


def _make_cube_fixture() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """3 只股 × 5 天. traded[d,c] 显式编排, 方便断言."""
    dates = np.array(
        [pd.Timestamp(f"2026-04-{d:02d}") for d in range(17, 22)],
        dtype="datetime64[ns]",
    )
    traded = np.ones((5, 3), dtype=bool)
    # 第 2 天 (index=1) 股 B 停牌
    traded[1, 1] = False
    # 第 4 天 (index=3) 股 C 停牌
    traded[3, 2] = False
    tokens = ["A.SZ", "B.SZ", "C.SZ"]
    return dates, traded, tokens


def test_no_suspension_case() -> None:
    """选股在次日均有成交 → rate=0, 无 warning."""
    dates, traded, tokens = _make_cube_fixture()
    # T=day 0 选 A (次日 A 有成交), T=day 2 选 A (次日 A 也有)
    signals = pd.DataFrame(
        [
            {"trade_date": dates[0], "token_id": "A.SZ", "weight": 1.0},
            {"trade_date": dates[2], "token_id": "A.SZ", "weight": 1.0},
        ]
    )
    r = count_recommend_then_suspended(signals, dates, traded, tokens)
    assert r.total == 2
    assert r.suspended_next_day == 0
    assert r.rate == 0.0


def test_catch_next_day_suspension() -> None:
    """T=day 0 选 B, T+1=day 1 B 停牌 → 命中."""
    dates, traded, tokens = _make_cube_fixture()
    signals = pd.DataFrame(
        [
            {"trade_date": dates[0], "token_id": "B.SZ", "weight": 0.5},
            {"trade_date": dates[0], "token_id": "A.SZ", "weight": 0.5},  # A 次日正常
        ]
    )
    r = count_recommend_then_suspended(signals, dates, traded, tokens)
    assert r.total == 2
    assert r.suspended_next_day == 1
    assert r.rate == 0.5
    assert abs(r.rate_weighted - 0.5) < 1e-9  # 权重各 0.5


def test_weighted_rate_differs_from_unweighted() -> None:
    """权重差异 → weighted 和 unweighted 不同."""
    dates, traded, tokens = _make_cube_fixture()
    # 2 个 case: A 权重 0.9 (次日正常), B 权重 0.1 (次日停牌)
    signals = pd.DataFrame(
        [
            {"trade_date": dates[0], "token_id": "A.SZ", "weight": 0.9},
            {"trade_date": dates[0], "token_id": "B.SZ", "weight": 0.1},
        ]
    )
    r = count_recommend_then_suspended(signals, dates, traded, tokens)
    assert r.total == 2
    assert r.suspended_next_day == 1
    assert r.rate == 0.5  # 命中率 1/2
    assert abs(r.rate_weighted - 0.1) < 1e-9  # 权重命中率 0.1/1.0


def test_skip_last_day() -> None:
    """最后一天的选股没有 T+1 可查, 应跳过不计入 total."""
    dates, traded, tokens = _make_cube_fixture()
    signals = pd.DataFrame(
        [
            {"trade_date": dates[-1], "token_id": "A.SZ", "weight": 1.0},  # 最后一天, 跳过
            {"trade_date": dates[0], "token_id": "A.SZ", "weight": 1.0},
        ]
    )
    r = count_recommend_then_suspended(signals, dates, traded, tokens)
    assert r.total == 1


def test_empty_signals() -> None:
    dates, traded, tokens = _make_cube_fixture()
    r = count_recommend_then_suspended(pd.DataFrame(), dates, traded, tokens)
    assert r.total == 0
    assert r.rate == 0.0


def test_csv_log_written(tmp_path) -> None:
    dates, traded, tokens = _make_cube_fixture()
    signals = pd.DataFrame([{"trade_date": dates[0], "token_id": "B.SZ", "weight": 1.0}])
    log = tmp_path / "suspended.csv"
    r = count_recommend_then_suspended(signals, dates, traded, tokens, log_path=log)
    assert r.suspended_next_day == 1
    assert log.exists()
    df = pd.read_csv(log)
    assert list(df.columns) == ["trade_date", "next_date", "code", "weight"]
    assert df.iloc[0]["code"] == "B.SZ"
