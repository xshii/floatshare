"""PopTrainer — Phase 3 抓涨停 Stage A 监督预训.

核心 tricks (vs 纯 BCE baseline):
    pos_weight = sqrt(neg/pos) ≈ 4.93 — 缓和 24x 类不平衡, 防过度预测正
    EMA decay 0.995                   — eval 用 EMA (raw 训练抖动大)
    Warmup 5% + cosine to 0.1×base_lr — 稳定起步
    Label smoothing 0.05              — 1→0.95, 0→0.05 缓过拟合
    valid_starts 排除最后 reward_horizon=3 天 (避免拿到 NaN label)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.nn import functional as F

from floatshare.ml.data.dataset import build_cube
from floatshare.ml.data.universe import select_per_industry_top_k
from floatshare.ml.evaluation.cube import classifier_metrics
from floatshare.ml.labels import label_stats
from floatshare.ml.model.agent import ActorCritic
from floatshare.ml.profiling import SectionTimer, maybe_section
from floatshare.ml.training.base import BaseTrainer
from floatshare.observability import logger

if TYPE_CHECKING:
    from pathlib import Path

    from floatshare.ml.data.dataset import MarketCube

# 持仓 K 天: 跟 HitLabelConfig sell_offset 对齐
_REWARD_HORIZON = 3


@dataclass(slots=True)
class _PopTrainCtx:
    """Pop 训练循环需要的全部数据.

    features / labels 一次性上 MPS, 避免每 batch 的 CPU np.stack + host→MPS 复制.
    mask/tt/ind 模板也预分配, slice 复用 — 减少 MPS allocator 压力.
    """

    cube_tr: MarketCube
    cube_va: MarketCube
    tr_labels: np.ndarray
    va_labels: np.ndarray
    valid_starts: np.ndarray  # (n_valid,) 可做 train step 的 day indices
    pos_weight: torch.Tensor
    ema_model: ActorCritic | None
    batch_days: int
    # GPU 预加载的特征 / 标签 — 训练/验证循环全程在 device 上切 window
    features_tr_gpu: torch.Tensor  # (n_days, n_tokens, F)
    features_va_gpu: torch.Tensor
    labels_tr_gpu: torch.Tensor  # (n_days, n_tokens) int
    # 预分配的 mask/tt/ind 模板, batch_days 条 — 可 slice 复用
    mask_tmpl: torch.Tensor  # (batch_days, n_tokens) bool
    tt_tmpl: torch.Tensor  # (batch_days, n_tokens) long
    ind_tmpl: torch.Tensor  # (batch_days, n_tokens) long


class PopTrainer(BaseTrainer):
    """Phase 3 监督预训 — BCE + pos_weight + EMA + warmup+cosine."""

    def __init__(
        self,
        model_cfg,
        data_cfg,
        train_cfg,
        *,
        epochs: int = 20,
        batch_days: int = 16,
        label_smooth: float = 0.05,
        use_ema: bool = True,
        ema_decay: float = 0.995,
        warmup_frac: float = 0.05,
        resume_from_ckpt: str | None = None,
        note: str | None = None,
        eval_every: int = 2,
        timer: SectionTimer | None = None,
        ckpt_name_override: str | None = None,
    ) -> None:
        super().__init__(
            model_cfg,
            data_cfg,
            train_cfg,
            epochs=epochs,
            eval_every=eval_every,
            note=note,
            ckpt_name_override=ckpt_name_override,
        )
        self.batch_days = batch_days
        self.label_smooth = label_smooth
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.warmup_frac = warmup_frac
        self._timer = timer  # None = 零开销; Not None = 分段打点 (profile 模式)

        if resume_from_ckpt:
            skipped = self.model.load_backbone(resume_from_ckpt, skip_shape_mismatch=True)
            n_loaded = sum(1 for _ in self.model.state_dict() if _) - len(skipped)
            logger.info(
                f"warm-start from {resume_from_ckpt}: {len(skipped)} 层跳过 (shape mismatch), "
                f"~{n_loaded} 层复用",
            )
            if skipped:
                for k, reason in skipped.items():
                    logger.info(f"  skip {k}: {reason}")

    # --- Hooks -----------------------------------------------------------

    @property
    def _metric_to_max(self) -> str:
        return "auc"

    @property
    def _ckpt_name(self) -> str:
        return "phase3_pretrain_best.pt"

    def _build_train_ctx(self) -> _PopTrainCtx:
        cube_tr, cube_va = self._build_cubes()
        tr_labels = cube_tr.hit_labels
        va_labels = cube_va.hit_labels
        assert tr_labels is not None, "train phase=3 cube 缺 hit_labels"
        assert va_labels is not None, "val phase=3 cube 缺 hit_labels"
        logger.info(f"  train: {cube_tr.n_days} days × {cube_tr.n_tokens} stocks")
        logger.info(f"  labels: {label_stats(tr_labels)}")
        logger.info(f"  val:   {cube_va.n_days} days × {cube_va.n_tokens} stocks")
        logger.info(f"  labels: {label_stats(va_labels)}")

        seq_len = self.model_cfg.seq_len
        valid_starts = np.arange(seq_len - 1, cube_tr.n_days - _REWARD_HORIZON)
        pos_weight = self._compute_pos_weight(tr_labels)

        ema_model = None
        if self.use_ema:
            ema_model = ActorCritic(self.model_cfg).to(self.device)
            # torch.compile 会给 state_dict key 加 _orig_mod. 前缀, 这里 strip 掉再 load
            src_sd = {k.removeprefix("_orig_mod."): v for k, v in self.model.state_dict().items()}
            ema_model.load_state_dict(src_sd)
            for p in ema_model.parameters():
                p.requires_grad = False

        # 一次性把 features/labels 搬到 device — 后续 slice 在 MPS 上做, 避免
        # 每 batch CPU→MPS copy (55 MB×100 batch/epoch ≈ 5.5 GB 带宽)
        features_tr_gpu = torch.from_numpy(cube_tr.features).to(self.device)
        features_va_gpu = torch.from_numpy(cube_va.features).to(self.device)
        labels_tr_gpu = torch.from_numpy(tr_labels).to(self.device)
        n_tokens = cube_tr.n_tokens
        bd = self.batch_days
        mask_tmpl = torch.ones(bd, n_tokens, dtype=torch.bool, device=self.device)
        tt_tmpl = torch.ones(bd, n_tokens, dtype=torch.long, device=self.device)
        ind_tmpl = torch.zeros(bd, n_tokens, dtype=torch.long, device=self.device)

        return _PopTrainCtx(
            cube_tr=cube_tr,
            cube_va=cube_va,
            tr_labels=tr_labels,
            va_labels=va_labels,
            valid_starts=valid_starts,
            pos_weight=pos_weight,
            ema_model=ema_model,
            batch_days=self.batch_days,
            features_tr_gpu=features_tr_gpu,
            features_va_gpu=features_va_gpu,
            labels_tr_gpu=labels_tr_gpu,
            mask_tmpl=mask_tmpl,
            tt_tmpl=tt_tmpl,
            ind_tmpl=ind_tmpl,
        )

    def _build_val_ctx(self, train_ctx: _PopTrainCtx) -> _PopTrainCtx:
        return train_ctx  # val 数据也在 _PopTrainCtx 里

    def _total_steps(self, train_ctx: _PopTrainCtx) -> int:
        return max(1, len(train_ctx.valid_starts) // train_ctx.batch_days) * self.epochs

    def _build_scheduler(self, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
        warmup_steps = int(total_steps * self.warmup_frac)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.1 + 0.45 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _run_train_epoch(self, epoch: int, ctx: _PopTrainCtx) -> dict[str, float]:
        del epoch
        np.random.shuffle(ctx.valid_starts)
        epoch_loss, epoch_pos, n_batch = 0.0, 0, 0
        seq_len = self.model_cfg.seq_len

        for i in range(0, len(ctx.valid_starts), ctx.batch_days):
            batch_idx = ctx.valid_starts[i : i + ctx.batch_days]
            if len(batch_idx) < 2:
                continue
            loss_val, n_pos = self._train_one_batch(batch_idx, ctx, seq_len)
            if loss_val is None:
                continue
            epoch_loss += loss_val
            epoch_pos += n_pos
            n_batch += 1

        return {
            "train_loss": epoch_loss / max(n_batch, 1),
            "pos": float(epoch_pos),
        }

    def _run_eval(self, ctx: _PopTrainCtx) -> dict[str, float]:
        eval_model = ctx.ema_model if ctx.ema_model is not None else self.model
        m = classifier_metrics(
            eval_model,
            ctx.features_va_gpu,
            ctx.va_labels,
            self.model_cfg.seq_len,
            self.device,
            top_k=10,
        )
        return {
            "val_auc": m["auc"],
            "val_p@10": m["p@10"],
            "base": m["base_rate"],
            # 让 base class 能拿 `auc` 来判最佳
            "auc": m["auc"],
        }

    def _save_ckpt(self, path: Path, val_m: dict[str, float]) -> None:
        """Pop 存 EMA model (eval 用的就是它), 无 EMA 时存 raw."""
        del val_m
        ctx = getattr(self, "_last_ctx", None)
        model_to_save = ctx.ema_model if ctx and ctx.ema_model is not None else self.model
        model_to_save.save(path)

    # --- Helpers ---------------------------------------------------------

    def _build_cubes(self) -> tuple[MarketCube, MarketCube]:
        universe = select_per_industry_top_k(
            self.data_cfg.db_path,
            as_of_date=self.data_cfg.train_end,
            top_k=15,
            turnover_window=20,
            vola_window=60,
            circ_mv_min=20_0000.0,
            circ_mv_max=500_0000.0,
            w_turnover=0.5,
            w_vola=0.35,
            w_cmv=0.15,
        )
        logger.info(
            f"Per-industry top-15 universe @ {self.data_cfg.train_end}: "
            f"{len(universe)} 股 (理论 31×15=465, 冷门行业不够 K 的有多少取多少); "
            f"前 3: {universe[:3]}",
        )
        logger.info("构建 train cube (phase=3) …")
        cube_tr = build_cube(
            self.data_cfg,
            self.data_cfg.train_start,
            self.data_cfg.train_end,
            phase=3,
            universe=universe,
        )
        logger.info("构建 val cube (phase=3) …")
        cube_va = build_cube(
            self.data_cfg,
            self.data_cfg.val_start,
            self.data_cfg.val_end,
            phase=3,
            universe=universe,
        )
        return cube_tr, cube_va

    def _compute_pos_weight(self, labels: np.ndarray) -> torch.Tensor:
        pos = int((labels == 1).sum())
        neg = int((labels == 0).sum())
        raw = neg / max(pos, 1)
        eff = float(np.sqrt(raw))
        logger.info(
            f"Class imbalance — pos:{pos:,} neg:{neg:,} raw pos_weight={raw:.1f} "
            f"使用 sqrt={eff:.2f}",
        )
        return torch.tensor(eff, device=self.device, dtype=torch.float32)

    def _train_one_batch(
        self,
        batch_idx: np.ndarray,
        ctx: _PopTrainCtx,
        seq_len: int,
    ) -> tuple[float | None, int]:
        """一个 batch: forward → BCE loss → backward + EMA update.

        Returns (loss_item, n_pos_samples). None 表示 batch 无有效样本 (跳过).
        """
        b = len(batch_idx)
        with maybe_section(self._timer, "data_prep"):
            # 全在 GPU 上构 window: (B, T, N, F) → (B, N, T, F)
            windows = torch.stack(
                [ctx.features_tr_gpu[t - seq_len + 1 : t + 1] for t in batch_idx],
            )
            x_t = windows.permute(0, 2, 1, 3).contiguous()
            y_t = ctx.labels_tr_gpu[batch_idx]  # (B, N)
            # mask/tt/ind 预分配模板的前 b 行 (batch_days 是上限, 末尾 batch 可能不足)
            mask_t = ctx.mask_tmpl[:b]
            tt = ctx.tt_tmpl[:b]
            ind = ctx.ind_tmpl[:b]

        # bf16 autocast: model forward 走 bfloat16 (MPS native), loss 强制 fp32
        # 包 matmul/attn/FFN; softmax/layernorm/loss 内部会自动 upcast to fp32
        with (
            maybe_section(self._timer, "forward"),
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16),
        ):
            out = self.model(x_t, tt, ind, mask_t)
        valid_mask = y_t >= 0
        if valid_mask.sum() == 0:
            return None, 0

        with maybe_section(self._timer, "loss"):
            # Label smoothing: 0.05 → 1→0.95, 0→0.05
            y_f = y_t.float().clamp(0, 1)
            y_smooth = y_f * (1 - self.label_smooth) + 0.5 * self.label_smooth

            # BCE with logits + pos_weight — 显式 .float() 把 bf16 p_hit upcast 回 fp32
            # 避免 log/clamp 在 bf16 下数值不稳 (p_hit 接近 0/1 时 log 会炸)
            p_clamped = out.p_hit.float().clamp(1e-7, 1 - 1e-7)
            logits = torch.log(p_clamped / (1 - p_clamped))
            loss = F.binary_cross_entropy_with_logits(
                logits[valid_mask],
                y_smooth[valid_mask],
                pos_weight=ctx.pos_weight,
            )

        with maybe_section(self._timer, "backward"):
            self.optimizer.zero_grad()
            loss.backward()

        with maybe_section(self._timer, "optim_step"):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            self.scheduler.step()

            if ctx.ema_model is not None:
                self._ema_update(ctx.ema_model)
        self._last_ctx = ctx  # 供 _save_ckpt 取 EMA
        return float(loss.item()), int(y_f[valid_mask].sum().item())

    def _ema_update(self, ema_model: ActorCritic) -> None:
        with torch.no_grad():
            for ep, mp in zip(
                ema_model.parameters(),
                self.model.parameters(),
                strict=True,
            ):
                ep.mul_(self.ema_decay).add_(mp.data, alpha=1 - self.ema_decay)
