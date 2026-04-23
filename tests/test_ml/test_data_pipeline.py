"""数据管线单测 — features → normalize → dataset 端到端。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from floatshare.ml.config import DataConfig
from floatshare.ml.data.dataset import N_INDUSTRIES, build_cube
from floatshare.ml.features import FEATURE_COLS, N_FEATURES, compute_features
from floatshare.ml.normalize import cross_sectional_zscore

# === features.py ===


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    """3 只股 × 100 天的合成 OHLCV panel。"""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    rows = []
    rng = np.random.default_rng(42)
    for code in ("AAA.SZ", "BBB.SZ", "CCC.SZ"):
        prices = 10 * np.exp(np.cumsum(rng.normal(0, 0.02, len(dates))))
        for i, d in enumerate(dates):
            rows.append(
                {
                    "code": code,
                    "trade_date": d,
                    "open": prices[i] * 0.99,
                    "high": prices[i] * 1.01,
                    "low": prices[i] * 0.98,
                    "close": prices[i],
                    "volume": rng.integers(100_000, 1_000_000),
                    "amount": rng.integers(1_000_000, 10_000_000),
                    "pe_ttm": 20.0,
                    "pb": 2.0,
                    "turnover_rate": 1.0,
                    "total_mv": 1e10,
                    "net_mf_amount": rng.normal(0, 1e6),
                    "buy_lg_amount": rng.integers(100_000, 1_000_000),
                    "buy_elg_amount": rng.integers(100_000, 1_000_000),
                    "buy_sm_amount": rng.integers(100_000, 1_000_000),
                    "buy_md_amount": rng.integers(100_000, 1_000_000),
                    "sell_sm_amount": rng.integers(100_000, 1_000_000),
                    "sell_md_amount": rng.integers(100_000, 1_000_000),
                    "sell_lg_amount": rng.integers(100_000, 1_000_000),
                    "sell_elg_amount": rng.integers(100_000, 1_000_000),
                }
            )
    return pd.DataFrame(rows)


def test_features_shape_and_cols(synthetic_panel) -> None:
    feats = compute_features(synthetic_panel)
    assert len(feats) == 300
    for col in FEATURE_COLS:
        assert col in feats.columns
    assert "code" in feats.columns
    assert "trade_date" in feats.columns


def test_features_no_lookahead(synthetic_panel) -> None:
    """新加一天数据后, 旧天的特征值应不变 (因果性)。"""
    feats_full = compute_features(synthetic_panel)
    feats_partial = compute_features(synthetic_panel[synthetic_panel["trade_date"] <= "2024-03-15"])
    a = (
        feats_full.query("code == 'AAA.SZ' and trade_date <= '2024-03-15'")["ret_5d"]
        .dropna()
        .to_numpy()
    )
    b = feats_partial.query("code == 'AAA.SZ'")["ret_5d"].dropna().to_numpy()
    assert len(a) == len(b)
    np.testing.assert_allclose(a, b, atol=1e-10)


# === normalize.py ===


def test_normalize_zscore_mean_zero(synthetic_panel) -> None:
    """标准化后, 每天每个特征 mean ≈ 0 (除非全为 NaN)。"""
    feats = compute_features(synthetic_panel)
    z = cross_sectional_zscore(feats)
    by_day = z.groupby("trade_date")[list(FEATURE_COLS)].mean()
    # 跳过 warm-up 期 (前 60 天有大量 NaN→0 不是真 z-score)
    after_warmup = by_day.iloc[60:].mean().abs()
    assert (after_warmup < 0.1).all(), f"非零均值列: {after_warmup[after_warmup >= 0.1]}"


def test_normalize_clip_bound(synthetic_panel) -> None:
    feats = compute_features(synthetic_panel)
    z = cross_sectional_zscore(feats, clip_thresh=5.0)
    feature_data = z[list(FEATURE_COLS)].to_numpy()
    assert feature_data.max() <= 5.0
    assert feature_data.min() >= -5.0
    assert not np.isnan(feature_data).any()


# === dataset.py (用真实 SQLite, 仅在 DB 可用时跑) ===


def test_build_cube_phase1(tmp_path) -> None:
    """Phase 1: 31 行业, 端到端能构建出非空 cube。"""
    pytest.importorskip("sqlalchemy")
    db = "data/floatshare.db"
    import os

    if not os.path.exists(db):
        pytest.skip("数据库不可用")
    cfg = DataConfig(
        db_path=db, cache_dir=str(tmp_path), use_cache=False, universe_mode="top_mv", top_mv_n=10
    )
    cube = build_cube(cfg, "2024-06-01", "2024-12-31", phase=1)
    assert cube.n_tokens == N_INDUSTRIES
    assert cube.features.shape == (cube.n_days, N_INDUSTRIES, N_FEATURES)
    assert cube.features.dtype == np.float32
    assert cube.traded.mean() > 0.95  # 行业指数几乎天天交易


def test_build_cube_phase2_has_stocks(tmp_path) -> None:
    """Phase 2: 31 行业 + N 股票, stock token meta 正确。"""
    db = "data/floatshare.db"
    import os

    if not os.path.exists(db):
        pytest.skip("数据库不可用")
    cfg = DataConfig(
        db_path=db, cache_dir=str(tmp_path), use_cache=False, universe_mode="top_mv", top_mv_n=10
    )
    cube = build_cube(cfg, "2024-06-01", "2024-12-31", phase=2)
    assert cube.n_tokens == N_INDUSTRIES + 10
    assert cube.n_industries == N_INDUSTRIES
    stock_tokens = [t for t in cube.tokens if t.token_type == 1]
    assert len(stock_tokens) == 10
    for t in stock_tokens:
        assert 0 <= t.industry_id < N_INDUSTRIES  # 父行业 id 合法


def test_build_cube_cache_roundtrip(tmp_path) -> None:
    """cache 写盘后读回, cube 数据一致。"""
    db = "data/floatshare.db"
    import os

    if not os.path.exists(db):
        pytest.skip("数据库不可用")
    cfg = DataConfig(
        db_path=db, cache_dir=str(tmp_path), use_cache=True, universe_mode="top_mv", top_mv_n=5
    )
    cold = build_cube(cfg, "2024-10-01", "2024-12-31", phase=1)
    warm = build_cube(cfg, "2024-10-01", "2024-12-31", phase=1)

    np.testing.assert_array_equal(cold.dates, warm.dates)
    np.testing.assert_array_equal(cold.features, warm.features)
    np.testing.assert_array_equal(cold.traded, warm.traded)
    np.testing.assert_array_equal(cold.prices, warm.prices, err_msg="价格 cache roundtrip 不一致")
    assert [t.token_id for t in cold.tokens] == [t.token_id for t in warm.tokens]
