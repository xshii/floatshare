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

        # market benchmark: 默认兜底策略分 phase
        # - phase 1/2: 前 n_industries 个 token 是行业, 用行业日均收益代理
        # - phase 3: tokens 全是股票 (无行业 token), 用全股池日均收益代理
        if market_returns is not None:
            n_avail = len(self.log_returns)
            self.market_returns = market_returns[:n_avail].astype(np.float32)
        elif model_cfg.phase == 3:
            self.market_returns = self.log_returns.mean(axis=1).astype(np.float32)
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

    def _compute_reward(
        self,
        weights: np.ndarray,
        t: int,
        prev_weights: np.ndarray | None,
    ) -> float:
        """纯函数: 给定 (weights, t, prev_weights) 算 reward, 不改 state.

        抽出来让 step() 和 peek_reward() 共用 — GRPO group rollout 要对同一 state
        采 G 个 action 拿 G 个 reward, 必须能 peek 而不 mutate self.t / prev_weights.
        """
        K = self.cfg.reward_horizon
        K_ret = self.log_returns[t : t + K].sum(axis=0)  # (n_tokens,)
        mkt_K_ret = float(self.market_returns[t : t + K].sum())

        if self.mcfg.phase == 1:
            reward = self._reward_phase1(weights, K_ret, mkt_K_ret)
        elif self.mcfg.phase == 2:
            reward = self._reward_phase2(weights, K_ret, mkt_K_ret)
        else:  # phase == 3
            reward = self._reward_phase3(weights, K_ret, mkt_K_ret)

        if prev_weights is not None:
            turnover = float(np.abs(weights - prev_weights).sum())
            reward -= self.cfg.turnover_penalty * turnover
        return reward

    def peek_reward(self, weights: np.ndarray) -> float:
        """GRPO 专用: 同 state 试多 action 拿 reward, 不改 self.t / prev_weights.

        典型用法:
            state = env.reset()
            for g in range(G):
                w_g = sample_action(policy(state))
                r_g = env.peek_reward(w_g)   # 不 advance
            # 最后再 env.step(选一个 action) 真正推进
        """
        return self._compute_reward(weights, self.t, self.prev_weights)

    def step(self, weights: np.ndarray) -> tuple[float, EnvState | None, bool]:
        """应用 action (weights), 跳 K 天, 返回 (reward, next_state, done)。

        weights: (n_tokens,) 行权重, sum=1, masked 位置应为 0
        """
        reward = self._compute_reward(weights, self.t, self.prev_weights)
        self.prev_weights = weights.copy()
        self.t += self.cfg.reward_horizon
        done = self.t + self.cfg.reward_horizon >= self.cube.n_days
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

    def _reward_phase3(
        self,
        weights: np.ndarray,
        K_ret: np.ndarray,
        mkt_K_ret: float,
    ) -> float:
        """抓涨停选股 alpha: Σ w_i · (r_i - r_market).

        Phase 3 tokens 全是股票 (无行业 token), weights shape (n_stocks,).
        Reward 鼓励把权重压在超越市场的股上. Dirichlet 采样产连续权重,
        backtest 部署时可以 argmax / top-K 离散化 (训练让 softmax 自然尖锐).
        """
        alpha = K_ret - mkt_K_ret
        return float((weights * alpha).sum())
