"""目标函数 — 把 BacktestResult → 单一 score (越大越好)。

不要直接最大化 Sharpe — 容易选出"高 Sharpe 但回撤恐怖"的伪策略。
推荐 Calmar 主导 (年化收益 / 最大回撤) + Sharpe 次权 + 换手惩罚 + 回撤硬约束。

权重通过 ScoreWeights dataclass 显式参数化, 避免 magic number。
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from floatshare.application.backtest import BacktestResult


def _nan_safe(v: float, default: float = 0.0) -> float:
    """NaN → default (quantstats 在样本不足时常返 NaN)。"""
    return default if math.isnan(v) else v


@dataclass(frozen=True, slots=True)
class ScoreWeights:
    """组合 score 的权重 — 显式参数化避免 magic number。

    默认值的设计逻辑:
      - calmar_weight=1.0: 以 Calmar (年化/回撤) 为主, 偏好"稳健"而非"激进"
      - sharpe_weight=0.3: Sharpe 作为次要参考 (回撤更重要)
      - trade_penalty=0.001: 每笔交易扣 0.001, 在常见交易数 100-1000 范围内
        构成 0.1-1.0 的总惩罚, 跟 Calmar 同量级 → 抑制过度交易
      - max_drawdown_kill=-0.40: 回撤超 40% 直接淘汰, A股回测常见心理底线
    """

    calmar_weight: float = 1.0
    sharpe_weight: float = 0.3
    trade_penalty: float = 0.001
    max_drawdown_kill: float = -0.40


def composite_score(
    result: BacktestResult,
    weights: ScoreWeights | None = None,
) -> float:
    """多目标融合: Calmar + Sharpe - 换手惩罚, 回撤超限直接淘汰。

    回撤超限返回 -1e9 (optuna 等会自动避开)。
    """
    w = weights or ScoreWeights()
    m = result.metrics
    dd = _nan_safe(float(m.max_drawdown))
    if dd < w.max_drawdown_kill:
        return -1e9
    annual = _nan_safe(result.annual_return)
    sharpe = _nan_safe(float(m.sharpe))
    n_trades = len(result.trades) if not result.trades.empty else 0
    calmar = annual / max(abs(dd), 0.01)
    return w.calmar_weight * calmar + w.sharpe_weight * sharpe - w.trade_penalty * n_trades


# 类型: BacktestResult → score
ScoreFn = Callable[["BacktestResult"], float]
DEFAULT_OBJECTIVE: ScoreFn = composite_score
