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

import dataclasses
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from floatshare.ml.model.agent import ActorCritic
from floatshare.ml.tracking import MetricsTracker, RunHandle
from floatshare.observability import logger, notify

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
        note: str | None = None,
        tracker: MetricsTracker | None = None,
    ) -> None:
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg
        self.epochs = epochs
        self.eval_every = eval_every
        self.note = note

        # 先 seed 再建模型: ActorCritic.__init__ 用 torch.randn 初始化 weights,
        # np.random.shuffle 是 train loop 里的 batch 洗牌, MPS RNG 影响 dropout.
        # 不 seed → 每次 run 不同 init + 不同 shuffle + 不同 dropout, 没复现性.
        self._seed_all(train_cfg.seed)

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
        self._tracker = tracker if tracker is not None else MetricsTracker()

    @staticmethod
    def _seed_all(seed: int) -> None:
        """Seed 所有随机源 — Python random / numpy / torch CPU / torch MPS / torch CUDA.

        说明: torch.manual_seed 只管 CPU; MPS 和 CUDA 要各自的 seed API.
        numpy 是 batch shuffle 用的; random 库一般不走, 但保险起见一起 seed.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

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
        n_params = self.model.n_params()
        logger.info(f"模型参数: {n_params:,}")

        trainer_name = type(self).__name__
        try:
            with self._tracker.run(
                trainer=trainer_name,
                config=self._config_snapshot(),
                note=self.note,
                n_params=n_params,
                metric_key=self._metric_to_max,
            ) as handle:
                logger.info(f"metrics run_id: {handle.run_id}")
                for epoch in range(self.epochs):
                    if self._run_one_epoch(epoch, train_ctx, val_ctx, handle):
                        break
        except BaseException as e:
            # tracker 的 with-block 已经把 status 标 CRASHED, 这里只推送
            notify(
                title=f"{trainer_name} CRASHED",
                body=f"{type(e).__name__}: {str(e)[:200]} | note={self.note or '-'}",
            )
            raise

        logger.info(f"完成 ✓ best val {self._metric_to_max}={self._best_metric:.3f}")
        notify(
            title=f"{trainer_name} done",
            body=(
                f"best {self._metric_to_max}={self._best_metric:.3f} "
                f"({self.epochs} epochs) | note={self.note or '-'}"
            ),
        )
        return self._best_metric

    def _run_one_epoch(
        self,
        epoch: int,
        train_ctx: Any,
        val_ctx: Any,
        handle: RunHandle,
    ) -> bool:
        """一个 epoch — train + optional eval + tracker log. True = 早停."""
        t0 = time.time()
        train_m = self._run_train_epoch(epoch, train_ctx)
        train_t = time.time() - t0
        cur_lr = self.optimizer.param_groups[0]["lr"]

        val_m: dict[str, float] | None = None
        eval_t = 0.0
        stopped = False
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
            stopped = self._update_best_and_check_stop(val_m, epoch, handle)
        else:
            logger.info(
                f"E{epoch:02d} | {self._format_train_log(train_m)} | {train_t:.1f}s",
            )

        handle.log_epoch(
            epoch,
            train_metrics=train_m,
            val_metrics=val_m,
            lr=cur_lr,
            train_time_s=train_t,
            eval_time_s=eval_t,
        )
        return stopped

    def _config_snapshot(self) -> dict[str, Any]:
        """训练配置快照 (存进 metrics.db, 方便后续对比)."""
        return {
            "model": dataclasses.asdict(self.model_cfg),
            "data": dataclasses.asdict(self.data_cfg),
            "train": dataclasses.asdict(self.train_cfg),
            "epochs": self.epochs,
            "eval_every": self.eval_every,
        }

    def _update_best_and_check_stop(
        self,
        val_m: dict[str, float],
        epoch: int,
        handle: RunHandle,
    ) -> bool:
        """检查是否早停. True = 停, False = 继续."""
        cur = val_m[self._metric_to_max]
        if cur > self._best_metric:
            self._best_metric = cur
            self._no_improve = 0
            path = self.ckpt_dir / self._ckpt_name
            self._save_ckpt(path, val_m)
            handle.update_best(cur, epoch)
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
