"""PPOTrainer — Phase 1 / Phase 2 PPO (rollout + clipped surrogate + GAE).

与 PopTrainer 共享 BaseTrainer 的 epoch / 早停 / ckpt 骨架, 自身只管:
    - 构建 MarketEnv (train + val, 含 market benchmark 选 equal_sw / hs300)
    - rollout (每 epoch collect_rollout n_steps) + PPO update
    - FeatureEvaluator (每 10 epoch 写 IC/RankIC CSV)
    - eval 用 run_deterministic_rollout 跑 val env 算 sharpe
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from floatshare.ml.data.dataset import build_cube
from floatshare.ml.data.loader import load_market_returns
from floatshare.ml.evaluation.env import run_deterministic_rollout
from floatshare.ml.evaluation.metrics import (
    compute_max_drawdown,
    compute_sharpe,
    compute_turnover_avg,
)
from floatshare.ml.feature_eval import FeatureEvaluator
from floatshare.ml.features import FEATURE_COLS
from floatshare.ml.rl.env import MarketEnv
from floatshare.ml.rl.ppo import ppo_update
from floatshare.ml.rl.rollout import collect_rollout
from floatshare.ml.training.base import BaseTrainer
from floatshare.observability import logger

if TYPE_CHECKING:
    from floatshare.ml.config import PPOConfig


@dataclass(slots=True)
class _PPOTrainCtx:
    env_train: MarketEnv
    env_val: MarketEnv
    feat_eval: FeatureEvaluator


class PPOTrainer(BaseTrainer):
    """Phase 1/2 PPO 训练器."""

    def __init__(
        self,
        model_cfg,
        data_cfg,
        ppo_cfg: PPOConfig,
        train_cfg,
        *,
        epochs: int,
    ) -> None:
        super().__init__(model_cfg, data_cfg, train_cfg, epochs=epochs, eval_every=5)
        self.ppo_cfg = ppo_cfg

    @property
    def _metric_to_max(self) -> str:
        return "sharpe"

    @property
    def _ckpt_name(self) -> str:
        return f"phase{self.model_cfg.phase}_best.pt"

    def _build_train_ctx(self) -> _PPOTrainCtx:
        cube_train, cube_val = self._build_cubes()

        mkt_train, mkt_val = self._load_market_benchmark()
        env_train = MarketEnv(cube_train, self.ppo_cfg, self.model_cfg, market_returns=mkt_train)
        env_val = MarketEnv(cube_val, self.ppo_cfg, self.model_cfg, market_returns=mkt_val)

        log_dir = Path(self.train_cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        feat_eval = FeatureEvaluator(
            list(FEATURE_COLS),
            log_dir / f"feature_eval_phase{self.model_cfg.phase}.csv",
            reward_horizon=self.ppo_cfg.reward_horizon,
        )
        return _PPOTrainCtx(env_train=env_train, env_val=env_val, feat_eval=feat_eval)

    def _build_val_ctx(self, train_ctx: _PPOTrainCtx) -> _PPOTrainCtx:
        return train_ctx

    def _total_steps(self, train_ctx: _PPOTrainCtx) -> int:
        del train_ctx
        return self.epochs

    def _build_scheduler(self, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
        # base 只调 scheduler.step(), CosineAnnealingLR 和 LambdaLR 在这点上通用.
        return torch.optim.lr_scheduler.CosineAnnealingLR(  # type: ignore[return-value]
            self.optimizer,
            T_max=total_steps,
            eta_min=self.train_cfg.lr * 0.1,
        )

    def _run_train_epoch(self, epoch: int, ctx: _PPOTrainCtx) -> dict[str, float]:
        batch = collect_rollout(
            ctx.env_train,
            self.model,
            n_steps=self.ppo_cfg.rollout_days,
            device=self.device,
            model_cfg=self.model_cfg,
        )
        m = ppo_update(self.model, self.optimizer, batch, self.ppo_cfg, self.model_cfg, self.device)
        self.scheduler.step()

        if epoch % 10 == 0:
            ctx.feat_eval.evaluate(ctx.env_train.cube, epoch)
            ctx.feat_eval.save()
            top = ctx.feat_eval.top_n(n=3, by="rank_ic")
            logger.info(
                "  特征 Top-3 RankIC: " + ", ".join(f"{r.feature}={r.rank_ic:+.3f}" for r in top),
            )

        return {
            "train_R": float(batch.rewards.mean()),
            "pol": m.policy_loss,
            "ent": m.entropy,
            "kl": m.kl,
        }

    def _run_eval(self, ctx: _PPOTrainCtx) -> dict[str, float]:
        rewards: list[float] = []
        weights_hist: list[np.ndarray] = []

        def _on_step(_t, _s, weights_full: np.ndarray, reward: float) -> None:
            rewards.append(reward)
            weights_hist.append(weights_full)

        run_deterministic_rollout(
            ctx.env_val,
            self.model,
            self.device,
            self.model_cfg,
            _on_step,
        )
        rs = np.array(rewards, dtype=np.float64)
        K = self.ppo_cfg.reward_horizon
        return {
            "n_steps": len(rs),
            "sharpe": compute_sharpe(rs, K),
            "cum_return": float(rs.sum()),
            "mean_reward": float(rs.mean()) if len(rs) else 0.0,
            "max_drawdown": compute_max_drawdown(rs),
            "turnover_avg": compute_turnover_avg(weights_hist),
        }

    # --- helpers ---------------------------------------------------------

    def _build_cubes(self):
        logger.info("构建 train cube …")
        cube_train = build_cube(
            self.data_cfg,
            self.data_cfg.train_start,
            self.data_cfg.train_end,
            phase=self.model_cfg.phase,
        )
        logger.info(f"  train: {cube_train.n_days} days × {cube_train.n_tokens} tokens")
        logger.info("构建 val cube …")
        cube_val = build_cube(
            self.data_cfg,
            self.data_cfg.val_start,
            self.data_cfg.val_end,
            phase=self.model_cfg.phase,
        )
        logger.info(f"  val: {cube_val.n_days} days × {cube_val.n_tokens} tokens")
        return cube_train, cube_val

    def _load_market_benchmark(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """equal_sw: 返回 (None, None) 让 env 用 31 行业等权兜底; hs300: 读 index_daily."""
        if getattr(self.train_cfg, "market_baseline", "equal_sw") != "hs300":
            logger.info("  baseline=equal_sw (31 行业等权日均)")
            return None, None
        mkt_train = load_market_returns(
            self.data_cfg.db_path,
            self.data_cfg.train_start,
            self.data_cfg.train_end,
        ).to_numpy()
        mkt_val = load_market_returns(
            self.data_cfg.db_path,
            self.data_cfg.val_start,
            self.data_cfg.val_end,
        ).to_numpy()
        logger.info(
            f"  baseline=HS300 train={len(mkt_train)}d val={len(mkt_val)}d",
        )
        return mkt_train, mkt_val
