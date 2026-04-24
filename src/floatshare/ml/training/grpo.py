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
from floatshare.ml.data.universe import select_per_industry_top_k
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
    _sample_action,
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
    """Phase 1/2/3 GRPO 训练器 — critic-free, 组内归一化 baseline.

    Phase 3 抓涨停: action = Dirichlet over 461 股, reward = Σ w_i · (r_i - r_market),
    typical reward_horizon=1 对齐 HitLabelConfig 1-day hold.
    """

    def __init__(
        self,
        model_cfg,
        data_cfg,
        grpo_cfg: GRPOConfig,
        train_cfg,
        *,
        epochs: int,
        note: str | None = None,
    ) -> None:
        super().__init__(model_cfg, data_cfg, train_cfg, epochs=epochs, eval_every=5, note=note)
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
        # Phase 3 需要显式 universe (top-15 per SW L1, ~460 股), 对齐 Pop 的做法;
        # Phase 1/2 走 cube 内置 default universe.
        universe: list[str] | None = None
        if self.model_cfg.phase == 3:
            universe = select_per_industry_top_k(
                db_path=self.data_cfg.db_path,
                as_of_date=self.data_cfg.train_end,  # universe 按训练尾取 snapshot
            )
            logger.info(f"GRPO phase=3 universe: {len(universe)} 股 (train_end snapshot)")

        logger.info("GRPO 构建 train cube …")
        cube_train = build_cube(
            self.data_cfg,
            self.data_cfg.train_start,
            self.data_cfg.train_end,
            phase=self.model_cfg.phase,
            universe=universe,
        )
        logger.info(f"  train: {cube_train.n_days} days × {cube_train.n_tokens} tokens")
        logger.info("GRPO 构建 val cube …")
        cube_val = build_cube(
            self.data_cfg,
            self.data_cfg.val_start,
            self.data_cfg.val_end,
            phase=self.model_cfg.phase,
            universe=universe,  # train/val 共享 universe 保证 stock 索引一致
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
    """Group rollout: 每步在同一 state 下采 G 个 action, 各 peek 一个 reward.

    收集 T × G 条 (state, action, log_prob, reward), states 按
    [s_0_g0, s_0_g1, ..., s_0_gG-1, s_1_g0, ...] 展平 — 与 group_advantage
    的 (N/G, G) reshape 布局一致.

    实现要点:
        1. model forward 1 次 → ActionOut (policy at state s_t)
        2. 对同一 policy 采 G 个 Dirichlet sample → G 个 (weights, log_prob)
        3. env.peek_reward(weights_g) 算 reward, **不改 env.t / prev_weights**
        4. env.step(weights_first) 真正推进 — 用第一个 sample 作为"实际执行"
           (任选其一即可, 组相对 advantage 对推进路径不敏感)

    done 时重置 env; 最后 group_size 条记录的 done=True 让 KL/advantage 分组边界清晰.
    """
    states_group: list = []
    actions_group: list[Tensor] = []
    log_probs_group: list[Tensor] = []
    rewards_group: list[float] = []
    dones_group: list[bool] = []

    state = env.reset()
    for _step in range(n_steps):
        x, tt, ind, mask = state_to_tensors(state, device)
        with torch.no_grad():
            out = model(x, tt, ind, mask)

        # G 个 (action, log_prob, reward) 在同一 state 下
        first_weights_np: np.ndarray | None = None
        for g in range(group_size):
            weights, log_prob = _sample_action(out, mask, model_cfg.n_industries)
            weights_np = weights[0].cpu().numpy().astype(np.float32)
            reward = env.peek_reward(weights_np)
            states_group.append(state)
            actions_group.append(weights[0].cpu())
            log_probs_group.append(log_prob[0].cpu())
            rewards_group.append(reward)
            dones_group.append(False)
            if g == 0:
                first_weights_np = weights_np

        # 用第 0 个 action 推进 env (peek 过了不算, step 会再算一次但反正要 mutate)
        assert first_weights_np is not None
        _, next_state, done = env.step(first_weights_np)
        if done:
            # 标记本组 G 条 done=True (advantage 分组边界在 group_size 整数倍上天然对齐)
            for i in range(1, group_size + 1):
                dones_group[-i] = True
            state = env.reset()
        else:
            assert next_state is not None
            state = next_state

    return RolloutBatch(
        states=states_group,
        actions=torch.stack(actions_group),
        log_probs=torch.stack(log_probs_group),
        values=torch.zeros(len(rewards_group), dtype=torch.float32),  # GRPO 不用
        rewards=torch.tensor(rewards_group, dtype=torch.float32),
        dones=torch.tensor(dones_group, dtype=torch.bool),
    )


# 让 lint 闭嘴 (Tensor 被 _collect_group_rollout 用)
_ = Tensor
