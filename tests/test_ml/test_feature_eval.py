"""特征评价单测 — IC / RankIC / Entropy 计算 + history CSV roundtrip。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from floatshare.ml.data.dataset import MarketCube, TokenMeta
from floatshare.ml.feature_eval import FeatureEvaluator


def _toy_cube(n_days: int = 50, n_tokens: int = 5, n_feat: int = 3, seed: int = 0) -> MarketCube:
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0, 0.02, (n_days, n_tokens)).astype(np.float32)
    prices = (10 * np.exp(np.cumsum(log_ret, axis=0))).astype(np.float32)
    # feature 0: 真 alpha (跟未来 5 天 return 有相关性), feat 1/2: 噪声
    features = rng.normal(0, 1, (n_days, n_tokens, n_feat)).astype(np.float32)
    # 让 feat 0 与下 5 天 return 正相关 (注入 IC)
    cum = np.concatenate([np.zeros((1, n_tokens)), np.cumsum(log_ret, axis=0)])
    K = 5
    fwd = (cum[K : K + n_days - K] - cum[: n_days - K])[: n_days - K - 1]
    features[: len(fwd), :, 0] = fwd + rng.normal(0, 0.5, fwd.shape).astype(np.float32) * 0.01
    traded = np.ones((n_days, n_tokens), dtype=bool)
    tokens = [TokenMeta(f"T{i}", 0, i) for i in range(n_tokens)]
    return MarketCube(
        dates=np.arange(n_days), tokens=tokens, features=features, prices=prices, traded=traded
    )


def test_evaluator_detects_alpha_feature(tmp_path) -> None:
    """注入 IC 的 feature 应该排第一 (rank_ic 绝对值最大)。"""
    cube = _toy_cube(n_days=80, n_tokens=10, n_feat=3, seed=42)
    ev = FeatureEvaluator(["alpha", "noise1", "noise2"], tmp_path / "fe.csv", reward_horizon=5)
    rows = ev.evaluate(cube, epoch=0)
    assert len(rows) == 3
    by_rank = sorted(rows, key=lambda r: abs(r.rank_ic), reverse=True)
    assert by_rank[0].feature == "alpha", (
        f"alpha 应排第一, 实际: {[(r.feature, r.rank_ic) for r in by_rank]}"
    )


def test_evaluator_history_save_load(tmp_path) -> None:
    cube = _toy_cube(n_days=40, n_tokens=5, n_feat=3, seed=1)
    out = tmp_path / "history.csv"
    ev = FeatureEvaluator(["a", "b", "c"], out, reward_horizon=3)
    ev.evaluate(cube, epoch=0)
    ev.evaluate(cube, epoch=10)
    ev.save()
    df = pd.read_csv(out)
    assert len(df) == 6  # 3 features × 2 epochs
    assert set(df.columns) >= {"epoch", "feature", "ic", "rank_ic", "entropy", "cv", "abs_mean"}
    assert sorted(df["epoch"].unique().tolist()) == [0, 10]


def test_evaluator_top_n() -> None:
    cube = _toy_cube(n_days=60, n_tokens=8, n_feat=3, seed=7)
    ev = FeatureEvaluator(["alpha", "n1", "n2"], "/tmp/_unused.csv", reward_horizon=5)
    ev.evaluate(cube, epoch=0)
    top = ev.top_n(n=2, by="rank_ic")
    assert len(top) == 2
    assert top[0].feature == "alpha"


def test_entropy_nonzero_for_diverse() -> None:
    """正态分布 entropy 应明显大于常数序列。"""
    cube_diverse = _toy_cube(seed=3)
    ev = FeatureEvaluator(["a", "b", "c"], "/tmp/_e.csv")
    rows = ev.evaluate(cube_diverse, epoch=0)
    for r in rows:
        assert r.entropy > 0.5, f"{r.feature} entropy {r.entropy} too low"
