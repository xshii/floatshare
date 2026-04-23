"""涨停 label 生成 — 给 Phase 2 抓涨停任务用。

Label 定义:
    hit[t, i] = 1  if  open[t+buy, i] / open[t+sell, i] - 1 >= threshold
              = 0  otherwise
              = -1 if data 缺失 (停盘 / 一字板 / 数据末尾, 训练时 mask 掉)

默认: buy_offset=1, sell_offset=2, threshold=0.05
    含义: D 日决策 → D+1 开盘买 → D+2 开盘卖, 涨幅 ≥ 5% 算"打中" (1 天超短线)

业界经验过滤 (避免乐观偏差):
    - 一字板买不到: D+1 open == high == low → 视为不可买 (label=-1)
    - 停牌: 任一价格 NaN → label=-1
    - ST / 新股: 在 universe 选择时排除, labels 不管
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_YI_ZI_TOL = 1e-4  # 一字板浮点容差 ≈ 一分钱


@dataclass(frozen=True, slots=True)
class HitLabelConfig:
    """涨停 label 参数 — 默认超短线 1 天 5%.

    D 日决策 → D+1 开盘买 → D+2 开盘卖, 涨幅 ≥ 5%.
    A 股 T+1 制度下持仓 1 天是最短可行 (隔夜).
    """

    buy_offset: int = 1  # D+buy_offset 日开盘买
    sell_offset: int = 2  # D+sell_offset 日开盘卖 (持 1 天)
    threshold: float = 0.05  # 涨幅阈值 (5%)
    exclude_yi_zi: bool = True  # 一字板视为不可买 (open == high == low)


def make_hit_labels(
    opens: np.ndarray,
    highs: np.ndarray,
    cfg: HitLabelConfig | None = None,
    lows: np.ndarray | None = None,
) -> np.ndarray:
    """对 (n_days, n_tokens) 价格矩阵生成 hit label.

    Args:
        opens: (n_days, n_tokens) 开盘价, NaN = 停盘
        highs: (n_days, n_tokens) 最高价, 用于一字板检测
        cfg: HitLabelConfig
        lows: (n_days, n_tokens) 最低价, 精确检测一字板封死 (None=旧宽松判定)

    Returns:
        labels: (n_days, n_tokens) int8
            1  = 命中 (期间涨幅 ≥ threshold 且可买)
            0  = 未命中 (能买但没涨够)
           -1  = 不可用 (NaN / 一字板 / 末端不够 sell_offset 天)
    """
    cfg = cfg or HitLabelConfig()
    n_days, _ = opens.shape
    labels = np.full(opens.shape, -1, dtype=np.int8)

    last_valid = n_days - cfg.sell_offset
    if last_valid <= cfg.buy_offset:
        return labels

    buy_p, sell_p, buy_h, buy_l = _slice_window(
        opens,
        highs,
        lows,
        cfg.buy_offset,
        cfg.sell_offset,
        last_valid,
    )
    valid = _compute_buyable_mask(buy_p, sell_p, buy_h, buy_l, cfg)
    ret = _compute_period_return(buy_p, sell_p)

    hit = (ret >= cfg.threshold) & valid
    miss = ~hit & valid
    labels[:last_valid][hit] = 1
    labels[:last_valid][miss] = 0
    return labels


def label_stats(labels: np.ndarray) -> dict[str, float]:
    """label 分布统计 — 调试 + 监控数据健康度用."""
    n_total = labels.size
    n_valid = int((labels >= 0).sum())
    n_hit = int((labels == 1).sum())
    return {
        "total": n_total,
        "valid": n_valid,
        "valid_pct": n_valid / n_total if n_total else 0.0,
        "hit": n_hit,
        "hit_rate": n_hit / n_valid if n_valid else 0.0,  # 阳性率 (基线)
    }


# --- helpers -----------------------------------------------------------------


def _slice_window(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray | None,
    buy_off: int,
    sell_off: int,
    last_valid: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """按 buy/sell offset 切出对应的 (buy_p, sell_p, buy_h, buy_l) 窗口."""
    buy_p = opens[buy_off : last_valid + buy_off]
    sell_p = opens[sell_off : last_valid + sell_off]
    buy_h = highs[buy_off : last_valid + buy_off]
    buy_l = lows[buy_off : last_valid + buy_off] if lows is not None else None
    return buy_p, sell_p, buy_h, buy_l


def _compute_buyable_mask(
    buy_p: np.ndarray,
    sell_p: np.ndarray,
    buy_h: np.ndarray,
    buy_l: np.ndarray | None,
    cfg: HitLabelConfig,
) -> np.ndarray:
    """可买 mask — 非 NaN 且非一字板封死."""
    valid = ~(np.isnan(buy_p) | np.isnan(sell_p) | np.isnan(buy_h))
    if not cfg.exclude_yi_zi:
        return valid
    if buy_l is not None:
        # 全天封死: open == high == low. T 字板 / 破板 (low < high) 仍可买
        yi_zi_locked = (
            (np.abs(buy_p - buy_h) < _YI_ZI_TOL)
            & (np.abs(buy_p - buy_l) < _YI_ZI_TOL)
            & ~np.isnan(buy_l)
        )
        return valid & ~yi_zi_locked
    # 兼容旧调用 (没传 lows): 退回旧的宽松判定 (open == high)
    return valid & ~(np.abs(buy_p - buy_h) < _YI_ZI_TOL)


def _compute_period_return(buy_p: np.ndarray, sell_p: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return sell_p / buy_p - 1
