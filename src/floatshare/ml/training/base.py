"""BaseTrainer — 通用 training loop 骨架 (AdamW + scheduler + 早停 + ckpt).

子类实现:
    _build_scheduler(total_steps)     — 返回 LR scheduler (Pop: warmup+cosine, PPO: cosine)
    _build_cubes()                    — 返回 (train_cube, val_cube) 或 (env_train, env_val)
    _total_steps(train_ctx)           — 总训练步数 (for scheduler)
    _run_train_epoch(epoch, train_ctx)→ dict[str, float]  — 一个 epoch 训练, 返回 metric
    _run_eval(val_ctx)                → dict[str, float]  — val 指标 (含 _metric_to_max key)
    _metric_to_max                    — 哪个 val 指标最大化 (auc / sharpe)
    _ckpt_name                        — 保存文件名
    _save_ckpt(path, val_m)           — 哪个模型被 save (raw / EMA / …)

注: train_ctx / val_ctx 是每个子类自定义的 "prepared data" 容器 (cube / env).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from floatshare.ml.model.agent import ActorCritic
from floatshare.observability import logger

if TYPE_CHECKING:
    from floatshare.ml.config import DataConfig, ModelConfig, TrainConfig


class BaseTrainer(ABC):
    """通用训练器骨架. 子类填空 _abstract 钩子."""

    def __init__(
        self,
        model_cfg: ModelConfig,
        data_cfg: DataConfig,
        train_cfg: TrainConfig,
        *,
        epochs: int,
        eval_every: int = 2,
    ) -> None:
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg
        self.epochs = epochs
        self.eval_every = eval_every

        self.device = torch.device(train_cfg.device)
        self.model = ActorCritic(model_cfg).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )

        self.ckpt_dir = Path(train_cfg.ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self._best_metric: float = -float("inf")
        self._no_improve: int = 0

    # --- Abstract hooks (子类必实) ---------------------------------------

    @abstractmethod
    def _build_train_ctx(self) -> Any:
        """准备训练数据容器 (cube, env, …)."""

    @abstractmethod
    def _build_val_ctx(self, train_ctx: Any) -> Any:
        """准备 val 数据 (可复用 train_ctx 里的 universe / model_cfg)."""

    @abstractmethod
    def _total_steps(self, train_ctx: Any) -> int:
        """scheduler 需要的总 step 数."""

    @abstractmethod
    def _build_scheduler(self, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
        """返回 LR scheduler."""

    @abstractmethod
    def _run_train_epoch(self, epoch: int, train_ctx: Any) -> dict[str, float]:
        """跑一个 epoch 的 train, 返回 train metrics."""

    @abstractmethod
    def _run_eval(self, val_ctx: Any) -> dict[str, float]:
        """跑 val, 返回 metric dict (含 self._metric_to_max 字段)."""

    @property
    @abstractmethod
    def _metric_to_max(self) -> str: ...

    @property
    @abstractmethod
    def _ckpt_name(self) -> str: ...

    # --- Hooks with default impl -----------------------------------------

    def _save_ckpt(self, path: Path, val_m: dict[str, float]) -> None:
        """默认 save raw model. EMA 子类覆盖."""
        del val_m
        self.model.save(path)

    def _format_train_log(self, train_m: dict[str, float]) -> str:
        return " ".join(f"{k}={v:.4f}" for k, v in train_m.items())

    def _format_val_log(self, val_m: dict[str, float]) -> str:
        return " ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in val_m.items()
        )

    # --- Main loop (不需 override) ---------------------------------------

    def fit(self) -> float:
        """执行训练. 返回 best val metric."""
        logger.info(
            f"{type(self).__name__} — device={self.device} "
            f"phase={self.model_cfg.phase} epochs={self.epochs}",
        )
        train_ctx = self._build_train_ctx()
        val_ctx = self._build_val_ctx(train_ctx)
        total_steps = self._total_steps(train_ctx)
        self.scheduler = self._build_scheduler(total_steps)
        logger.info(f"模型参数: {self.model.n_params():,}")

        for epoch in range(self.epochs):
            t0 = time.time()
            train_m = self._run_train_epoch(epoch, train_ctx)
            train_t = time.time() - t0
            cur_lr = self.optimizer.param_groups[0]["lr"]

            is_eval_epoch = epoch % self.eval_every == 0 or epoch == self.epochs - 1
            if is_eval_epoch:
                t_eval = time.time()
                val_m = self._run_eval(val_ctx)
                eval_t = time.time() - t_eval
                logger.info(
                    f"E{epoch:02d} | {self._format_train_log(train_m)} "
                    f"{self._format_val_log(val_m)} lr={cur_lr:.2e} "
                    f"| train={train_t:.1f}s eval={eval_t:.1f}s",
                )
                if self._update_best_and_check_stop(val_m, epoch):
                    break
            else:
                logger.info(
                    f"E{epoch:02d} | {self._format_train_log(train_m)} | {train_t:.1f}s",
                )

        logger.info(f"完成 ✓ best val {self._metric_to_max}={self._best_metric:.3f}")
        return self._best_metric

    def _update_best_and_check_stop(
        self,
        val_m: dict[str, float],
        epoch: int,
    ) -> bool:
        """检查是否早停. True = 停, False = 继续."""
        cur = val_m[self._metric_to_max]
        if cur > self._best_metric:
            self._best_metric = cur
            self._no_improve = 0
            path = self.ckpt_dir / self._ckpt_name
            self._save_ckpt(path, val_m)
            logger.info(
                f"  ✓ 新最佳 {self._metric_to_max}={cur:.3f}, save → {path}",
            )
            return False
        self._no_improve += 1
        logger.info(
            f"  · no improve {self._no_improve}/{self.train_cfg.early_stop_patience}",
        )
        if self._no_improve >= self.train_cfg.early_stop_patience:
            logger.info(
                f"  ⚠ early stop @ E{epoch:02d} "
                f"({self._metric_to_max} 连续 {self.train_cfg.early_stop_patience} 次未升)",
            )
            return True
        return False
