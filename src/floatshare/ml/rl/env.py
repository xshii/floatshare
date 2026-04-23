"""MarketEnv — 接 MarketCube, 实现 reward 公式 + 状态滚动。

Reward 公式 (见 ml/__init__.py):
    Phase 1: R = Σ_ind w_ind * (r_ind - r_market) - λ_turn * |Δw|
    Phase 2: R = Σ_i w_i * (r_i - r_industry(i))
               + γ * Σ_ind w_ind_total * (r_ind - r_market)
               - λ_turn * |Δw|

熵奖励 (β_ent * H(w)) 在 PPO loss 里加, 不在 env reward 里。

API 风格类 gym 但简化 (无 gym 依赖):
    state = env.reset(start_idx=None)
    reward, next_state, done = env.step(weights)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from floatshare.ml.types import EnvState

if TYPE_CHECKING:
    from floatshare.ml.config import ModelConfig, PPOConfig
    from floatshare.ml.data.dataset import MarketCube


class MarketEnv:
    """numpy-based env, 状态/奖励仅用 numpy 算, agent 接口处再转 torch。"""

    def __init__(
        self,
        cube: MarketCube,
        ppo_cfg: PPOConfig,
        model_cfg: ModelConfig,
        market_returns: np.ndarray | None = None,
    ) -> None:
        self.cube = cube
        self.cfg = ppo_cfg
        self.mcfg = model_cfg

        # === 预算 log returns (n_days-1, n_tokens) ===
        # nan/inf 一律置 0 (停盘日没价格)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ret = np.log(cube.prices[1:] / cube.prices[:-1])
        self.log_returns = np.nan_to_num(
            log_ret,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

        # market benchmark: 默认用 31 行业日均收益代理 (兜底)
        if market_returns is not None:
            n_avail = len(self.log_returns)
            self.market_returns = market_returns[:n_avail].astype(np.float32)
        else:
            self.market_returns = self.log_returns[:, : model_cfg.n_industries].mean(axis=1)

        # token meta (固定不变)
        self.token_types = np.array([t.token_type for t in cube.tokens], dtype=np.int64)
        self.industry_ids = np.array([t.industry_id for t in cube.tokens], dtype=np.int64)

        self.t: int = 0
        self.prev_weights: np.ndarray | None = None

    @property
    def n_tokens(self) -> int:
        return self.cube.n_tokens

    @property
    def n_days(self) -> int:
        return self.cube.n_days

    def reset(self, start_idx: int | None = None) -> EnvState:
        """重置 env. start_idx=None → 从 seq_len (历史足够) 开始。"""
        T = self.mcfg.seq_len
        K = self.cfg.reward_horizon
        max_start = self.cube.n_days - K - 1
        if max_start < T:
            raise ValueError(f"cube 太短: n_days={self.cube.n_days} < seq_len({T})+K({K})+1")
        self.t = max(T, start_idx if start_idx is not None else T)
        self.t = min(self.t, max_start)
        self.prev_weights = None
        return self._state()

    def _state(self) -> EnvState:
        """当前 state — 取过去 T 天的特征切片。"""
        T = self.mcfg.seq_len
        # cube.features (n_days, n_tokens, F) → window (T, N, F) → (N, T, F)
        x = self.cube.features[self.t - T + 1 : self.t + 1]
        x = np.transpose(x, (1, 0, 2))
        return EnvState(
            x=x.astype(np.float32),
            token_types=self.token_types,
            industry_ids=self.industry_ids,
            mask=self.cube.traded[self.t].copy(),
        )

    def step(self, weights: np.ndarray) -> tuple[float, EnvState | None, bool]:
        """应用 action (weights), 跳 K 天, 返回 (reward, next_state, done)。

        weights: (n_tokens,) 行权重, sum=1, masked 位置应为 0
        """
        K = self.cfg.reward_horizon
        K_ret = self.log_returns[self.t : self.t + K].sum(axis=0)  # (n_tokens,)
        mkt_K_ret = float(self.market_returns[self.t : self.t + K].sum())

        if self.mcfg.phase == 1:
            reward = self._reward_phase1(weights, K_ret, mkt_K_ret)
        else:
            reward = self._reward_phase2(weights, K_ret, mkt_K_ret)

        # 换手罚
        if self.prev_weights is not None:
            turnover = float(np.abs(weights - self.prev_weights).sum())
            reward -= self.cfg.turnover_penalty * turnover

        self.prev_weights = weights.copy()
        self.t += K
        done = self.t + K >= self.cube.n_days
        next_s = None if done else self._state()
        return reward, next_s, done

    def _reward_phase1(
        self,
        weights: np.ndarray,
        K_ret: np.ndarray,
        mkt_K_ret: float,
    ) -> float:
        """Σ_ind w_ind * (r_ind - r_market). 仅前 n_industries 个 token 是行业。"""
        n_ind = self.mcfg.n_industries
        alpha = K_ret[:n_ind] - mkt_K_ret
        return float((weights[:n_ind] * alpha).sum())

    def _reward_phase2(
        self,
        weights: np.ndarray,
        K_ret: np.ndarray,
        mkt_K_ret: float,
    ) -> float:
        """选股 alpha (stock - industry) + γ * 行业 timing alpha。"""
        n_ind = self.mcfg.n_industries
        stock_w = weights[n_ind:]
        stock_K_ret = K_ret[n_ind:]
        stock_ind_ids = self.industry_ids[n_ind:]
        ind_K_ret = K_ret[:n_ind]

        # 选股 alpha: 同行业内超额
        selection_alpha = stock_K_ret - ind_K_ret[stock_ind_ids]
        r_select = float((stock_w * selection_alpha).sum())

        # 行业 timing: 把 stock_w 聚合回行业, × (ind - mkt)
        ind_total_w = np.zeros(n_ind, dtype=np.float32)
        np.add.at(ind_total_w, stock_ind_ids, stock_w)
        timing_alpha = ind_K_ret - mkt_K_ret
        r_timing = float((ind_total_w * timing_alpha).sum())

        return r_select + self.cfg.industry_timing_weight * r_timing
