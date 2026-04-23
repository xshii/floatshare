"""超参 / Walk-Forward Optimization — 通用框架，所有策略统一接入。

每个策略只需在 class 上加 `search_space(trial)` classmethod 即可被自动调参:

    @register("my_strategy")
    class MyStrategy(bt.Strategy):
        params = (("lookback", 5), ("threshold", 0.02))

        @classmethod
        def search_space(cls, trial):
            return dict(
                lookback=trial.suggest_int("lookback", 3, 30),
                threshold=trial.suggest_float("threshold", 0.005, 0.05),
            )

调用:

    from floatshare.application.optimization import walk_forward_optimize
    results = walk_forward_optimize(MyStrategy, data, splits)

防 overfit 三件套:
- WFO 切分 (train/test 时间错开)
- 多目标 score (Calmar 主导 + 回撤约束 + 换手惩罚)
- guardrails (单 split 必须 OOS Sharpe > 0)
"""

from floatshare.application.optimization.objectives import (
    DEFAULT_OBJECTIVE,
    composite_score,
)
from floatshare.application.optimization.tuner import optimize_in_window
from floatshare.application.optimization.walk_forward import (
    WFResult,
    WFSplit,
    make_walk_forward_splits,
    walk_forward_optimize,
)

__all__ = [
    "DEFAULT_OBJECTIVE",
    "WFResult",
    "WFSplit",
    "composite_score",
    "make_walk_forward_splits",
    "optimize_in_window",
    "walk_forward_optimize",
]
