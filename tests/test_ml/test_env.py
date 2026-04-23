"""MarketEnv 单测 — 重点验证 reward 公式正确性 + state 滚动逻辑。"""

from __future__ import annotations

import numpy as np
import pytest

from floatshare.ml.config import ModelConfig, PPOConfig
from floatshare.ml.data.dataset import MarketCube, TokenMeta
from floatshare.ml.rl.env import MarketEnv


def _synthetic_cube(
    n_days: int = 100,
    n_industries: int = 3,
    n_stocks: int = 0,
    seed: int = 42,
) -> MarketCube:
    """合成 cube 用于 env 测试 — 价格 / 特征 / mask 全合成可控。"""
    rng = np.random.default_rng(seed)
    n_tokens = n_industries + n_stocks
    F = 22
    # 价格: random walk
    log_ret = rng.normal(0.001, 0.02, (n_days, n_tokens)).astype(np.float32)
    log_ret[0] = 0
    prices = (10 * np.exp(np.cumsum(log_ret, axis=0))).astype(np.float32)
    features = rng.normal(0, 1, (n_days, n_tokens, F)).astype(np.float32)
    traded = np.ones((n_days, n_tokens), dtype=bool)

    tokens: list[TokenMeta] = [
        TokenMeta(token_id=f"IND{i}", token_type=0, industry_id=i) for i in range(n_industries)
    ]
    tokens.extend(
        TokenMeta(token_id=f"STK{s}", token_type=1, industry_id=s % n_industries)
        for s in range(n_stocks)
    )

    return MarketCube(
        dates=np.arange("2024-01-01", n_days, dtype="datetime64[D]"),
        tokens=tokens,
        features=features,
        prices=prices,
        traded=traded,
    )


def test_env_reset_and_state_shape() -> None:
    cube = _synthetic_cube(n_days=100, n_industries=3, n_stocks=0)
    mcfg = ModelConfig(phase=1, n_industries=3, seq_len=10, n_features=22)
    env = MarketEnv(cube, PPOConfig(reward_horizon=5), mcfg)

    state = env.reset()
    assert state.x.shape == (3, 10, 22)  # (N, T, F)
    assert state.token_types.shape == (3,)
    assert state.mask.shape == (3,)


def test_env_reward_phase1_alpha_correct() -> None:
    """Phase 1 reward = Σ w_ind * (r_ind - r_mkt). 手算验证。"""
    cube = _synthetic_cube(n_days=50, n_industries=3, n_stocks=0)
    mcfg = ModelConfig(phase=1, n_industries=3, seq_len=10, n_features=22)
    # 关掉换手罚, 单测纯 alpha 项
    env = MarketEnv(cube, PPOConfig(reward_horizon=5, turnover_penalty=0.0), mcfg)

    env.reset(start_idx=10)
    K = 5
    K_ret = env.log_returns[10 : 10 + K].sum(axis=0)  # (3,)
    mkt = float(env.market_returns[10 : 10 + K].sum())
    expected_alpha = K_ret - mkt

    weights = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    expected_reward = float((weights * expected_alpha).sum())

    reward, _, _ = env.step(weights)
    assert reward == pytest.approx(expected_reward, abs=1e-5)


def test_env_reward_phase2_selection_plus_timing() -> None:
    """Phase 2: r_select (stock - industry) + γ * r_timing (industry - mkt)."""
    cube = _synthetic_cube(n_days=50, n_industries=3, n_stocks=6)
    mcfg = ModelConfig(phase=2, n_industries=3, seq_len=10, n_features=22)
    env = MarketEnv(
        cube,
        PPOConfig(reward_horizon=5, turnover_penalty=0.0, industry_timing_weight=0.3),
        mcfg,
    )

    env.reset(start_idx=10)
    K = 5
    K_ret = env.log_returns[10 : 10 + K].sum(axis=0)  # (9,)
    mkt = float(env.market_returns[10 : 10 + K].sum())

    # 构造 weights: 行业部分 0 (Phase 2 只看 stock_w), stock_w sum=1
    # 6 stocks 分布: industry 0 有 stock 0,1; industry 1 有 2,3; industry 2 有 4,5
    weights = np.zeros(9, dtype=np.float32)
    weights[3:9] = [0.2, 0.1, 0.3, 0.1, 0.2, 0.1]
    stock_w = weights[3:]
    stock_ids = np.array([0, 1, 2, 0, 1, 2])  # default cycling per cube setup

    # selection: stock - industry
    sel = (stock_w * (K_ret[3:] - K_ret[stock_ids])).sum()
    # timing: aggregate stock_w to industry, × (ind - mkt)
    ind_w = np.zeros(3)
    np.add.at(ind_w, stock_ids, stock_w)
    timing = (ind_w * (K_ret[:3] - mkt)).sum()
    expected = float(sel + 0.3 * timing)

    reward, _, _ = env.step(weights)
    assert reward == pytest.approx(expected, abs=1e-5)


def test_env_turnover_penalty() -> None:
    """连续 step, 第二次 step 加换手罚 |w2 - w1|_1 * λ_turn。"""
    cube = _synthetic_cube(n_days=50, n_industries=3)
    mcfg = ModelConfig(phase=1, n_industries=3, seq_len=10, n_features=22)
    env = MarketEnv(cube, PPOConfig(reward_horizon=5, turnover_penalty=0.01), mcfg)

    env.reset(start_idx=10)
    w1 = np.array([1, 0, 0], dtype=np.float32)
    _r1, _, _ = env.step(w1)

    w2 = np.array([0, 1, 0], dtype=np.float32)
    r2, _, _ = env.step(w2)

    # 第二次 reward 含 -0.01 * |w2-w1|_1 = -0.01 * 2 = -0.02
    K = 5
    K_ret_t2 = env.log_returns[15 : 15 + K].sum(axis=0)
    mkt_t2 = float(env.market_returns[15 : 15 + K].sum())
    expected_alpha = float((w2 * (K_ret_t2 - mkt_t2)).sum())
    expected_r2 = expected_alpha - 0.01 * 2.0
    assert r2 == pytest.approx(expected_r2, abs=1e-5)


def test_env_done_at_end() -> None:
    cube = _synthetic_cube(n_days=30, n_industries=3)
    mcfg = ModelConfig(phase=1, n_industries=3, seq_len=10, n_features=22)
    env = MarketEnv(cube, PPOConfig(reward_horizon=5), mcfg)

    env.reset(start_idx=10)
    weights = np.ones(3, dtype=np.float32) / 3
    n_steps = 0
    done = False
    while not done and n_steps < 20:
        _, _, done = env.step(weights)
        n_steps += 1
    assert done
    assert n_steps <= 5  # 30 天 / K=5 ≈ 4 步即结束
