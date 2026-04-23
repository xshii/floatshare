"""特征评价模块 — 训练过程中跟踪每个 feature 的表现。

每个 epoch (或每 N epoch) 调用一次, 记录:
    - IC (Pearson):     feature 与 next-K-day return 的截面相关性
    - RankIC (Spearman): rank 的相关性 (鲁棒)
    - 熵 (Shannon):      连续变量分桶后的香农熵, 反映分布丰富度
    - CV (变异系数):     std / |mean|, 反映波动幅度
    - abs_mean:          中心化程度

输出:
    history CSV: epoch, feature, ic, rank_ic, entropy, cv, abs_mean
    可后接 web UI 看趋势 / 找 dead 特征。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    from floatshare.ml.data.dataset import MarketCube


@dataclass(frozen=True, slots=True)
class FeatureEvalRow:
    """单 feature × 单 epoch 的指标快照."""

    epoch: int
    feature: str
    ic: float  # cross-sectional Pearson IC, 日均
    rank_ic: float  # Spearman IC, 日均
    entropy: float  # 分桶熵
    cv: float  # std / |mean|
    abs_mean: float


class FeatureEvaluator:
    """跟踪 K 维特征在训练过程中的表现 → CSV history."""

    def __init__(
        self,
        feature_names: Sequence[str],
        out_path: str | Path,
        reward_horizon: int = 5,
    ) -> None:
        self.names = list(feature_names)
        self.out_path = Path(out_path)
        self.K = reward_horizon
        self.history: list[FeatureEvalRow] = []

    def evaluate(self, cube: MarketCube, epoch: int) -> list[FeatureEvalRow]:
        """对一个 cube 算所有 feature 的指标。返回当 epoch 的所有行 (并 append 到 history)。"""
        _n_days, n_tokens, n_feat = cube.features.shape
        if n_feat != len(self.names):
            raise ValueError(f"cube.features 维度 {n_feat} != names {len(self.names)}")

        # K-day 前向收益: forward_K[t, i] = sum(log_return[t : t+K])
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ret = np.nan_to_num(
                np.log(cube.prices[1:] / cube.prices[:-1]),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float64)
        valid_t = len(log_ret) - self.K
        if valid_t <= 0:
            return []
        cum = np.concatenate(
            [
                np.zeros((1, n_tokens), dtype=np.float64),
                np.cumsum(log_ret, axis=0),
            ]
        )
        forward_K = cum[self.K : self.K + valid_t] - cum[:valid_t]  # (valid_t, n_tokens)

        rows: list[FeatureEvalRow] = []
        for f_idx, fname in enumerate(self.names):
            feat = cube.features[:valid_t, :, f_idx].astype(np.float64)
            ic = _daily_xs_corr(feat, forward_K, kind="pearson")
            rank_ic = _daily_xs_corr(feat, forward_K, kind="spearman")
            flat = cube.features[:, :, f_idx].ravel()
            rows.append(
                FeatureEvalRow(
                    epoch=epoch,
                    feature=fname,
                    ic=ic,
                    rank_ic=rank_ic,
                    entropy=_entropy(flat),
                    cv=float(flat.std() / (abs(flat.mean()) + 1e-8)),
                    abs_mean=float(abs(flat.mean())),
                )
            )
        self.history.extend(rows)
        return rows

    def save(self) -> Path:
        """覆盖写当前 history 到 CSV (单调累加, 每次写全量)."""
        if not self.history:
            return self.out_path
        df = pd.DataFrame(
            [
                {
                    "epoch": r.epoch,
                    "feature": r.feature,
                    "ic": r.ic,
                    "rank_ic": r.rank_ic,
                    "entropy": r.entropy,
                    "cv": r.cv,
                    "abs_mean": r.abs_mean,
                }
                for r in self.history
            ]
        )
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.out_path, index=False)
        return self.out_path

    def top_n(self, n: int = 5, by: str = "rank_ic") -> list[FeatureEvalRow]:
        """最近 epoch 中按指定 metric 绝对值排序前 N。"""
        if not self.history:
            return []
        last_epoch = max(r.epoch for r in self.history)
        last = [r for r in self.history if r.epoch == last_epoch]
        return sorted(last, key=lambda r: abs(getattr(r, by)), reverse=True)[:n]


# === Helpers ===


def _daily_xs_corr(
    feat: np.ndarray,
    ret: np.ndarray,
    kind: str = "pearson",
) -> float:
    """每天截面 corr, 跨天求均值。

    feat / ret: (T, N) — 同 shape, 按天 (轴 0) 算 cross-sectional corr。
    """
    cors: list[float] = []
    for d in range(feat.shape[0]):
        f, r = feat[d], ret[d]
        valid = ~np.isnan(f) & ~np.isnan(r)
        if valid.sum() < 5:
            continue
        fv = f[valid]
        rv = r[valid]
        if kind == "spearman":
            fv = pd.Series(fv).rank().to_numpy()
            rv = pd.Series(rv).rank().to_numpy()
        fc = fv - fv.mean()
        rc = rv - rv.mean()
        denom = np.sqrt((fc**2).sum() * (rc**2).sum())
        if denom > 1e-12:
            cors.append(float((fc * rc).sum() / denom))
    return float(np.mean(cors)) if cors else 0.0


def _entropy(x: np.ndarray, bins: int = 20) -> float:
    """连续变量分桶后的 Shannon 熵 (nats)。空 / 全 NaN → 0。"""
    x = x[~np.isnan(x)]
    if x.size == 0:
        return 0.0
    hist, _ = np.histogram(x, bins=bins)
    s = hist.sum()
    if s == 0:
        return 0.0
    p = hist / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())
