"""Action heads — 每个 phase 一个.

统一接口:
    Head.forward(h, token_types=None, industry_ids=None, mask=None) -> HeadOut

各 head 的 HeadOut 只含 head 自身产物 (weights/logits/p_hit 等), 不含 value.
ActorCritic 负责接 value_head 并把 HeadOut 包成对应的 ActionOut (见 agent.py).

mask 通过把 logit 设 -inf 实现, softmax 后该位置自动 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from floatshare.ml.config import ModelConfig


# --- Head outputs (head-only fields, 不含 value) -----------------------------


@dataclass(frozen=True, slots=True)
class IndustryHeadOut:
    weights: Tensor
    logits: Tensor


@dataclass(frozen=True, slots=True)
class HierarchicalHeadOut:
    industry_weights: Tensor
    stock_weights: Tensor
    ind_logits: Tensor
    stock_logits: Tensor


@dataclass(frozen=True, slots=True)
class FlatStockHeadOut:
    """Phase 3 抓涨停 — 4 类 action + N 维 stock softmax + N 维 P(hit)."""

    action_logits: Tensor  # (B, 4) {不买, 买1只, 买2只, 买3只}
    action_probs: Tensor  # (B, 4) softmax
    stock_logits: Tensor  # (B, N) 个股排序 logit
    stock_probs: Tensor  # (B, N) softmax
    p_hit: Tensor  # (B, N) sigmoid (BCE 监督)


HeadOut = IndustryHeadOut | HierarchicalHeadOut | FlatStockHeadOut


# --- Base class --------------------------------------------------------------


class Head(nn.Module):
    """所有 action head 的基类 — 提供统一 kwargs 接口给 ActorCritic.

    子类实现 forward(h, **kwargs), 通过 **kw 吞掉自己用不上的参数, 这样
    ActorCritic 可以用单一调用形式 head(h, token_types=.., industry_ids=.., mask=..).
    """

    def forward(
        self,
        h: Tensor,
        token_types: Tensor | None = None,
        industry_ids: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> HeadOut:
        raise NotImplementedError


# --- Phase 1: IndustryHead ---------------------------------------------------


class IndustryHead(Head):
    """Phase 1: N 个 token 全是行业, 直接 softmax over N."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(cfg.embed_dim, 1)

    def forward(
        self,
        h: Tensor,
        token_types: Tensor | None = None,
        industry_ids: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> IndustryHeadOut:
        del token_types, industry_ids
        logits = self.proj(h).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        return IndustryHeadOut(weights=F.softmax(logits, dim=-1), logits=logits)


# --- Phase 2: HierarchicalHead ----------------------------------------------


class HierarchicalHead(Head):
    """Phase 2: 顶层 softmax over industries + 底层 scatter softmax over stocks per industry.

    最终 per-stock 权重 = w_ind(parent_industry) × w_stock_in_industry
    所有行业的 sum(w_per_stock) = sum(w_ind) = 1.

    约定: industry tokens 排前 N_industries, stock tokens 排后.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.industry_proj = nn.Linear(cfg.embed_dim, 1)
        self.stock_proj = nn.Linear(cfg.embed_dim, 1)
        self.n_industries = cfg.n_industries

    def forward(
        self,
        h: Tensor,
        token_types: Tensor | None = None,
        industry_ids: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> HierarchicalHeadOut:
        del token_types
        assert industry_ids is not None, "HierarchicalHead 需要 industry_ids"
        assert mask is not None, "HierarchicalHead 需要 mask"

        n_ind = self.n_industries
        ind_h = h[:, :n_ind]
        stock_h = h[:, n_ind:]
        ind_mask = mask[:, :n_ind]
        stock_mask = mask[:, n_ind:]
        stock_ind_ids = industry_ids[:, n_ind:]

        ind_mask_eff = _effective_industry_mask(ind_mask, stock_ind_ids, stock_mask, n_ind)

        ind_logits = self.industry_proj(ind_h).squeeze(-1)
        ind_logits = ind_logits.masked_fill(~ind_mask_eff, float("-inf"))
        w_ind = F.softmax(ind_logits, dim=-1)

        stock_logits = self.stock_proj(stock_h).squeeze(-1)
        w_stock_in_ind = _scatter_softmax(stock_logits, stock_ind_ids, n_ind, stock_mask)

        parent_w = w_ind.gather(1, stock_ind_ids.long())
        w_stock = parent_w * w_stock_in_ind

        return HierarchicalHeadOut(
            industry_weights=w_ind,
            stock_weights=w_stock,
            ind_logits=ind_logits,
            stock_logits=stock_logits,
        )


def _effective_industry_mask(
    ind_mask: Tensor,
    stock_ind_ids: Tensor,
    stock_mask: Tensor,
    n_ind: int,
) -> Tensor:
    """行业 mask ∧ "至少有一只有效 stock 属该行业" — 否则行业仓位会变 NaN."""
    ind_membership = F.one_hot(stock_ind_ids.long(), n_ind).bool() & stock_mask.unsqueeze(-1)
    has_valid_stock = ind_membership.any(dim=1)
    return ind_mask & has_valid_stock


def _scatter_softmax(
    logits: Tensor,
    group_ids: Tensor,
    n_groups: int,
    valid_mask: Tensor,
) -> Tensor:
    """每个 group 内做 softmax, 输出形状同 logits, 非有效位置 = 0."""
    b, n = logits.shape
    group_mask = F.one_hot(group_ids.long(), n_groups).bool().permute(0, 2, 1)
    group_mask = group_mask & valid_mask.unsqueeze(1)

    expanded = logits.unsqueeze(1).expand(b, n_groups, n)
    masked = expanded.masked_fill(~group_mask, float("-inf"))
    max_val = masked.max(dim=-1, keepdim=True).values
    # 空组 max=-inf → 换 0 防 NaN (sum_exp=0 → softmax 全 0)
    safe_max = torch.where(torch.isinf(max_val), torch.zeros_like(max_val), max_val)
    exp_vals = (masked - safe_max).exp() * group_mask.float()
    sum_exp = exp_vals.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    softmax_per_group = exp_vals / sum_exp
    return softmax_per_group.sum(dim=1)


# --- Phase 3: FlatStockHead -------------------------------------------------


class FlatStockHead(Head):
    """Phase 3 抓涨停 head: action 4 类 + 个股 softmax + P(hit) BCE.

    Action head: 4 类 categorical {不买, 买1只, 买2只, 买3只}
    Stock head:  N 维 softmax → 排序选 top-K (K 由 action 决定)
    Hit head:    N 维 sigmoid → P(D+1→D+2 涨 ≥5%) (BCE 监督, Stage A 主任务)

    Stage A: 只训 p_hit (BCE), action/stock 跟着 backbone 走
    Stage B: 加 PPO loss 训 action+stock+value
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.stock_proj = nn.Linear(cfg.embed_dim, 1)
        self.hit_proj = nn.Linear(cfg.embed_dim, 1)
        self.action_proj = nn.Linear(cfg.embed_dim, 4)

    def forward(
        self,
        h: Tensor,
        token_types: Tensor | None = None,
        industry_ids: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> FlatStockHeadOut:
        del token_types, industry_ids
        stock_logits = self.stock_proj(h).squeeze(-1)
        hit_logits = self.hit_proj(h).squeeze(-1)
        if mask is not None:
            stock_logits = stock_logits.masked_fill(~mask, float("-inf"))
            hit_logits = hit_logits.masked_fill(~mask, -1e9)

        h_pool = _masked_mean_pool(h, mask)
        action_logits = self.action_proj(h_pool)

        return FlatStockHeadOut(
            action_logits=action_logits,
            action_probs=F.softmax(action_logits, dim=-1),
            stock_logits=stock_logits,
            stock_probs=F.softmax(stock_logits, dim=-1),
            p_hit=torch.sigmoid(hit_logits),
        )


def _masked_mean_pool(h: Tensor, mask: Tensor | None) -> Tensor:
    """(B, N, D) → (B, D), 按 mask 加权平均 (无 mask 就退化为普通 mean)."""
    if mask is None:
        return h.mean(dim=1)
    valid_w = mask.float().unsqueeze(-1)
    return (h * valid_w).sum(dim=1) / valid_w.sum(dim=1).clamp(min=1)
