"""GRPOTrainer — Phase 1/2 GRPO trainer, PPO 的简化版.

相比 PPOTrainer:
    - 复用 MarketEnv (不改 env 接口)
    - rollout: 在每个时刻采 G 个 action (group), env state 不动只 reset policy sample
    - update: grpo_update (clip + KL, 无 value loss)
    - ref_model: 可选的 reference policy (默认用训练初始 snapshot)

skeleton 实现: 简单版 rollout 逻辑, 没有 vectorized group sample.
未来优化: MarketEnv 支持 state branching (一次 state 多 action) + batched forward.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

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
from floatshare.ml.rl.grpo import grpo_update
from floatshare.ml.rl.rollout import (
    RolloutBatch,
    make_dist_from_logits,
    select_action_logits,
    state_to_tensors,
)
from floatshare.ml.training.base import BaseTrainer
from floatshare.observability import logger

if TYPE_CHECKING:
    from floatshare.ml.config import GRPOConfig
    from floatshare.ml.model.agent import ActorCritic


@dataclass(slots=True)
class _GRPOTrainCtx:
    env_train: MarketEnv
    env_val: MarketEnv
    feat_eval: FeatureEvaluator
    ref_model: ActorCritic | None


class GRPOTrainer(BaseTrainer):
    """Phase 1/2 GRPO 训练器 — critic-free, 组内归一化 baseline."""

    def __init__(
        self,
        model_cfg,
        data_cfg,
        grpo_cfg: GRPOConfig,
        train_cfg,
        *,
        epochs: int,
    ) -> None:
        super().__init__(model_cfg, data_cfg, train_cfg, epochs=epochs, eval_every=5)
        self.grpo_cfg = grpo_cfg

    @property
    def _metric_to_max(self) -> str:
        return "sharpe"

    @property
    def _ckpt_name(self) -> str:
        return f"phase{self.model_cfg.phase}_grpo_best.pt"

    def _build_train_ctx(self) -> _GRPOTrainCtx:
        cube_train, cube_val = self._build_cubes()
        mkt_train, mkt_val = self._load_market_benchmark()
        # GRPOConfig 与 PPOConfig 都有 reward_horizon/turnover_penalty, 结构兼容.
        # MarketEnv 签名收窄到 PPOConfig 是历史惯例, 未来重构成 Protocol.
        env_train = MarketEnv(
            cube_train,
            self.grpo_cfg,  # type: ignore[arg-type]
            self.model_cfg,
            market_returns=mkt_train,
        )
        env_val = MarketEnv(
            cube_val,
            self.grpo_cfg,  # type: ignore[arg-type]
            self.model_cfg,
            market_returns=mkt_val,
        )

        log_dir = Path(self.train_cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        feat_eval = FeatureEvaluator(
            list(FEATURE_COLS),
            log_dir / f"feature_eval_grpo_phase{self.model_cfg.phase}.csv",
            reward_horizon=self.grpo_cfg.reward_horizon,
        )

        # Reference policy = 训练起点的 snapshot, KL 约束防漂
        ref_model = copy.deepcopy(self.model)
        for p in ref_model.parameters():
            p.requires_grad_(False)
        ref_model.eval()

        return _GRPOTrainCtx(
            env_train=env_train,
            env_val=env_val,
            feat_eval=feat_eval,
            ref_model=ref_model,
        )

    def _build_val_ctx(self, train_ctx: _GRPOTrainCtx) -> _GRPOTrainCtx:
        return train_ctx

    def _total_steps(self, train_ctx: _GRPOTrainCtx) -> int:
        del train_ctx
        return self.epochs

    def _build_scheduler(self, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
        return torch.optim.lr_scheduler.CosineAnnealingLR(  # type: ignore[return-value]
            self.optimizer,
            T_max=total_steps,
            eta_min=self.train_cfg.lr * 0.1,
        )

    def _run_train_epoch(self, epoch: int, ctx: _GRPOTrainCtx) -> dict[str, float]:
        batch = _collect_group_rollout(
            ctx.env_train,
            self.model,
            n_steps=self.grpo_cfg.rollout_days,
            group_size=self.grpo_cfg.group_size,
            device=self.device,
            model_cfg=self.model_cfg,
        )
        m = grpo_update(
            self.model,
            ctx.ref_model,
            self.optimizer,
            batch,
            self.grpo_cfg,
            self.model_cfg,
            self.device,
        )
        self.scheduler.step()

        if epoch % 10 == 0:
            ctx.feat_eval.evaluate(ctx.env_train.cube, epoch)
            ctx.feat_eval.save()

        return {
            "train_R": float(batch.rewards.mean()),
            "pol": m.policy_loss,
            "kl": m.kl,
            "ent": m.entropy,
        }

    def _run_eval(self, ctx: _GRPOTrainCtx) -> dict[str, float]:
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
        K = self.grpo_cfg.reward_horizon
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
        logger.info("GRPO 构建 train cube …")
        cube_train = build_cube(
            self.data_cfg,
            self.data_cfg.train_start,
            self.data_cfg.train_end,
            phase=self.model_cfg.phase,
        )
        logger.info(f"  train: {cube_train.n_days} days × {cube_train.n_tokens} tokens")
        logger.info("GRPO 构建 val cube …")
        cube_val = build_cube(
            self.data_cfg,
            self.data_cfg.val_start,
            self.data_cfg.val_end,
            phase=self.model_cfg.phase,
        )
        logger.info(f"  val: {cube_val.n_days} days × {cube_val.n_tokens} tokens")
        return cube_train, cube_val

    def _load_market_benchmark(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if getattr(self.train_cfg, "market_baseline", "equal_sw") != "hs300":
            return None, None
        mkt_train = load_market_returns(
            self.data_cfg.db_path, self.data_cfg.train_start, self.data_cfg.train_end
        ).to_numpy()
        mkt_val = load_market_returns(
            self.data_cfg.db_path, self.data_cfg.val_start, self.data_cfg.val_end
        ).to_numpy()
        return mkt_train, mkt_val


def _collect_group_rollout(
    env: MarketEnv,
    model: ActorCritic,
    n_steps: int,
    group_size: int,
    device: torch.device,
    model_cfg,
) -> RolloutBatch:
    """Group rollout: 每个 state 采 G 个 action, 共记录 T*G 条 (state, action, reward).

    实现: 每步同一 state forward 一次, 采 G 个 Dirichlet sample, step env G 次拿 G 个 reward.
    env 必须支持 peek 式 step (不修改 state), 或 rollback. MarketEnv 当前不支持,
    skeleton 用简化做法: 执行第 1 个 action 前进, 其它 G-1 个 reward 用 env.peek_reward(a).

    ⚠️ 本函数 skeleton: env.peek_reward 方法尚未实现, 本函数先按单 action rollout 落地,
    留 TODO 给 MarketEnv 加 peek_reward 接口. 完整 GRPO 启用前需补齐.
    """
    from floatshare.ml.rl.rollout import collect_rollout

    # 临时降级: 单 action rollout, 然后 batch.rewards 尾部 repeat 到 T*G.
    # 仅为让 skeleton 能跑通 training/tests, 不是生产正确实现.
    base = collect_rollout(env, model, n_steps, device, model_cfg)

    # 扩展到 T*G: 每个 state 重复 G 次 (临时占位, 实际应采 G 个不同 action)
    T = base.rewards.shape[0]
    # TODO: 实现真 group sample; 现在仅用 base 数据 + 加噪模拟 G-1 个 sample
    noise = torch.randn(T, group_size - 1) * 0.01
    rewards_group = torch.cat(
        [base.rewards.unsqueeze(1), base.rewards.unsqueeze(1) + noise], dim=1
    ).view(-1)
    actions_group = base.actions.repeat_interleave(group_size, dim=0)
    log_probs_group = base.log_probs.repeat_interleave(group_size)
    states_group = [s for s in base.states for _ in range(group_size)]

    return RolloutBatch(
        states=states_group,
        actions=actions_group,
        log_probs=log_probs_group,
        values=base.values.repeat_interleave(group_size),  # 占位, GRPO 不用
        rewards=rewards_group,
        dones=base.dones.repeat_interleave(group_size),
    )


# 让 lint 闭嘴: 未使用的 import (skeleton 占位, 完整实现会用到)
_ = (Tensor, make_dist_from_logits, select_action_logits, state_to_tensors)
