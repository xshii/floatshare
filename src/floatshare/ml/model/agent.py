"""ActorCritic — backbone (TemporalEncoder + CrossTokenEncoder | DualAxisEncoder) + 可热插拔 head.

ckpt 支持 backbone-only save/load → Phase 1 → Phase 2 迁移时 backbone 权重直接复用,
head 重新训.

Phase → Head 分派走 HEAD_REGISTRY, forward 路径统一无 isinstance 分支.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from floatshare.ml.model.encoder import (
    CrossTokenEncoder,
    DualAxisEncoder,
    TemporalEncoder,
)
from floatshare.ml.model.heads import (
    FlatStockHead,
    FlatStockHeadOut,
    Head,
    HeadOut,
    HierarchicalHead,
    HierarchicalHeadOut,
    IndustryHead,
    IndustryHeadOut,
)
from floatshare.ml.types import (
    ActionOut,
    HierarchicalActionOut,
    IndustryActionOut,
    PopActionOut,
)

if TYPE_CHECKING:
    from floatshare.ml.config import ModelConfig


# --- HEAD_REGISTRY (phase → head class + wrapper) ----------------------------


def _wrap_industry(out: HeadOut, value: Tensor) -> IndustryActionOut:
    assert isinstance(out, IndustryHeadOut)
    return IndustryActionOut(weights=out.weights, logits=out.logits, value=value)


def _wrap_hierarchical(out: HeadOut, value: Tensor) -> HierarchicalActionOut:
    assert isinstance(out, HierarchicalHeadOut)
    return HierarchicalActionOut(
        industry_weights=out.industry_weights,
        stock_weights=out.stock_weights,
        ind_logits=out.ind_logits,
        stock_logits=out.stock_logits,
        value=value,
    )


def _wrap_pop(out: HeadOut, value: Tensor) -> PopActionOut:
    assert isinstance(out, FlatStockHeadOut)
    return PopActionOut(
        action_logits=out.action_logits,
        action_probs=out.action_probs,
        stock_logits=out.stock_logits,
        stock_probs=out.stock_probs,
        p_hit=out.p_hit,
        value=value,
    )


@dataclass(frozen=True, slots=True)
class HeadWiring:
    """每个 phase 的 head 组件组合 — 构造类 + HeadOut → ActionOut 的包装函数."""

    cls: type[Head]
    wrap: Callable[[HeadOut, Tensor], ActionOut]


HEAD_REGISTRY: dict[int, HeadWiring] = {
    1: HeadWiring(IndustryHead, _wrap_industry),
    2: HeadWiring(HierarchicalHead, _wrap_hierarchical),
    3: HeadWiring(FlatStockHead, _wrap_pop),
}


# --- ActorCritic -------------------------------------------------------------


class ActorCritic(nn.Module):
    """统一架构: backbone + head (按 phase 注册) + value head."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.temporal_enc: TemporalEncoder | None = None
        self.cross_enc: CrossTokenEncoder | None = None
        self.dual_enc: DualAxisEncoder | None = None
        self._build_backbone(cfg)

        if cfg.phase not in HEAD_REGISTRY:
            raise ValueError(f"unknown phase: {cfg.phase} (expect 1/2/3)")
        wiring = HEAD_REGISTRY[cfg.phase]
        self.head: Head = wiring.cls(cfg)
        self._wrap_head_out = wiring.wrap

        self.value_head = nn.Linear(cfg.embed_dim, 1)

        if cfg.pretrained_backbone:
            self.load_backbone(cfg.pretrained_backbone)

    def _build_backbone(self, cfg: ModelConfig) -> None:
        if cfg.encoder_mode == "dual_axis":
            self.dual_enc = DualAxisEncoder(cfg)
        else:
            self.temporal_enc = TemporalEncoder(cfg)
            self.cross_enc = CrossTokenEncoder(cfg)

    def encode(
        self,
        x: Tensor,
        token_types: Tensor,
        industry_ids: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """(B, N, T, F) → (B, N, D). cfg.encoder_mode 决定 single_axis vs dual_axis."""
        if self.dual_enc is not None:
            return self.dual_enc(x, mask, industry_ids)
        assert self.temporal_enc is not None
        assert self.cross_enc is not None
        b, n, t, f = x.shape
        h_token = self.temporal_enc(x.reshape(b * n, t, f)).reshape(b, n, -1)
        return self.cross_enc(h_token, token_types, industry_ids, mask)

    def forward(
        self,
        x: Tensor,
        token_types: Tensor,
        industry_ids: Tensor,
        mask: Tensor,
    ) -> ActionOut:
        h = self.encode(x, token_types, industry_ids, mask)
        value = self._compute_value(h, mask)
        head_out = self.head(h, token_types=token_types, industry_ids=industry_ids, mask=mask)
        return self._wrap_head_out(head_out, value)

    def _compute_value(self, h: Tensor, mask: Tensor) -> Tensor:
        """Mean-pool over valid tokens → value scalar (B,)."""
        valid_w = mask.float().unsqueeze(-1)
        h_pool = (h * valid_w).sum(dim=1) / valid_w.sum(dim=1).clamp(min=1)
        return self.value_head(h_pool).squeeze(-1)

    # --- ckpt I/O (cfg 单独存 JSON, state_dict 用 weights_only=True 安全) ---

    def save(self, path: str | Path) -> None:
        p = Path(path)
        torch.save(self.state_dict(), p)
        cfg_dict = dataclasses.asdict(self.cfg)
        p.with_suffix(".json").write_text(json.dumps(cfg_dict, indent=2))

    def load_backbone(
        self,
        path: str | Path,
        *,
        skip_shape_mismatch: bool = True,
    ) -> dict[str, str]:
        """仅加载 backbone, head/value 重新初始化. 安全模式 weights_only=True.

        shape-aware 加载 (skip_shape_mismatch=True, 默认):
            - 跳过 shape 不匹配的层 (如 feat_proj: (64, 37) vs (64, 39))
            - 返回 {key: reason} 字典表明哪些层被跳过 (reason 如 'shape_mismatch: (64,37)→(64,39)')
            - attention / FFN / pos_emb (不依赖 n_features) 全部复用

        对 37→39 维 warm-start: feat_proj 跳过 (仅 2 维参数差异), 其它 ~90% 参数继续使用.
        """
        sd = torch.load(path, map_location="cpu", weights_only=True)
        prefixes = ("temporal_enc.", "cross_enc.", "dual_enc.")
        current_sd = self.state_dict()
        backbone_sd: dict[str, Any] = {}
        skipped: dict[str, str] = {}
        for k, v in sd.items():
            if not k.startswith(prefixes):
                continue
            if k not in current_sd:
                skipped[k] = "not_in_current_model"
                continue
            if current_sd[k].shape != v.shape:
                if skip_shape_mismatch:
                    skipped[k] = f"shape_mismatch: {tuple(v.shape)}→{tuple(current_sd[k].shape)}"
                    continue
                raise RuntimeError(f"{k}: shape mismatch {v.shape} vs {current_sd[k].shape}")
            backbone_sd[k] = v
        _missing, unexpected = self.load_state_dict(backbone_sd, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected backbone keys: {unexpected}")
        return skipped

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_ckpt(path: str | Path) -> ActorCritic:
    """配套 save() — 加载 .pt + 同名 .json → 恢复 ActorCritic."""
    from floatshare.ml.config import ModelConfig

    p = Path(path)
    cfg_path = p.with_suffix(".json")
    if not cfg_path.exists():
        raise FileNotFoundError(f"缺 cfg JSON: {cfg_path}")
    cfg_dict = json.loads(cfg_path.read_text())
    # 过滤未知字段 (跨版本兼容)
    valid_keys = {f.name for f in dataclasses.fields(ModelConfig)}
    cfg = ModelConfig(**{k: v for k, v in cfg_dict.items() if k in valid_keys})

    model = ActorCritic(cfg)
    sd = torch.load(p, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    return model
