"""单窗口 optuna 调参 — WFO 引擎在每个 train 段调用它。"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import optuna
    import pandas as pd

    from floatshare.application.optimization.objectives import ScoreFn


def optimize_in_window(
    strategy_cls: type,
    train_data: pd.DataFrame,
    search_space: Callable[[optuna.Trial], dict[str, Any]],
    objective: ScoreFn,
    n_trials: int = 50,
    initial_capital: float = 100_000,
    study_name: str | None = None,
) -> dict[str, Any]:
    """对单一 train 窗口跑 optuna, 返回 best_params。

    objective 评估在 train 内部, 防 overfit 由 WFO 切分保证。
    """
    import optuna

    from floatshare.application.backtest import run_backtest

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def trial_fn(trial: optuna.Trial) -> float:
        params = search_space(trial)
        try:
            result = run_backtest(
                strategy_cls=strategy_cls,
                data=train_data,
                initial_capital=initial_capital,
                strategy_params=params,
            )
            return objective(result)
        except Exception:
            return -1e9

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(trial_fn, n_trials=n_trials, show_progress_bar=False)
    return dict(study.best_params)
