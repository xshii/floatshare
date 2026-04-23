"""Walk-Forward Optimization 引擎 — 防过拟合的标准范式。

每个 split = (train, test) 时间不重叠对:
  1. train 段跑 optuna 找最优 params
  2. test 段用这组 params 跑回测得 OOS 指标
  3. 滑动 train 窗口重复

最终评分 = mean(OOS Sharpe), std(OOS Sharpe) 反映稳定性。

设计:
- 不绑定具体策略 — 任何 backtrader Strategy 子类都可用
- 策略自带 search_space 时自动取; 否则 caller 显式传入
- 多目标 score 由 objectives.composite_score 提供
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any

import pandas as pd
from dateutil.relativedelta import relativedelta

from floatshare.application.backtest import run_backtest
from floatshare.application.optimization.objectives import (
    DEFAULT_OBJECTIVE,
    ScoreFn,
)
from floatshare.application.optimization.tuner import optimize_in_window
from floatshare.observability import logger

if TYPE_CHECKING:
    import optuna


@dataclass(frozen=True, slots=True)
class WFSplit:
    """单个 walk-forward 切分 — train 时间段 + test 时间段。"""

    train_start: date
    train_end: date
    test_start: date
    test_end: date


@dataclass(slots=True)
class WFResult:
    """单 split 的优化 + OOS 评估结果。"""

    split: WFSplit
    best_params: dict[str, Any]
    oos_total_return: float
    oos_annual_return: float
    oos_sharpe: float
    oos_max_drawdown: float
    oos_n_trades: int


def make_walk_forward_splits(
    start: date,
    end: date,
    *,
    train_years: int = 3,
    test_years: int = 1,
    step_years: int = 1,
) -> list[WFSplit]:
    """从 [start, end] 生成滑动窗口切分。

    用 dateutil.relativedelta 自动处理闰年 (2/29 + 1 year → 2/28)。
    """
    splits: list[WFSplit] = []
    cursor = start
    while True:
        train_end = cursor + relativedelta(years=train_years)
        test_end = train_end + relativedelta(years=test_years)
        if test_end > end:
            break
        splits.append(WFSplit(cursor, train_end, train_end, test_end))
        cursor = cursor + relativedelta(years=step_years)
    return splits


def walk_forward_optimize(
    strategy_cls: type,
    data: pd.DataFrame,
    splits: list[WFSplit],
    *,
    search_space: Callable[[optuna.Trial], dict[str, Any]] | None = None,
    objective: ScoreFn = DEFAULT_OBJECTIVE,
    n_trials: int = 50,
    initial_capital: float = 100_000,
) -> list[WFResult]:
    """对每个 split 在 train 上调参, 在 test 上 OOS 评估。

    search_space 默认从 `strategy_cls.search_space` classmethod 取,
    用户可显式传入覆盖。
    """
    space_fn = search_space or getattr(strategy_cls, "search_space", None)
    if space_fn is None:
        raise ValueError(
            f"{strategy_cls.__name__} 没有 search_space classmethod, 且未传入 search_space 参数",
        )

    results: list[WFResult] = []
    for i, sp in enumerate(splits, start=1):
        logger.info(
            f"[WFO {i}/{len(splits)}] train={sp.train_start}~{sp.train_end} "
            f"test={sp.test_start}~{sp.test_end}",
        )
        train_df = _slice_data(data, sp.train_start, sp.train_end)
        test_df = _slice_data(data, sp.test_start, sp.test_end)

        best = optimize_in_window(
            strategy_cls=strategy_cls,
            train_data=train_df,
            search_space=space_fn,
            objective=objective,
            n_trials=n_trials,
            initial_capital=initial_capital,
            study_name=f"{strategy_cls.__name__}_{sp.train_start}_{sp.train_end}",
        )
        logger.info(f"  best params: {best}")

        oos = run_backtest(
            strategy_cls=strategy_cls,
            data=test_df,
            initial_capital=initial_capital,
            strategy_params=best,
        )
        m = oos.metrics
        results.append(
            WFResult(
                split=sp,
                best_params=best,
                oos_total_return=oos.total_return,
                oos_annual_return=oos.annual_return,
                oos_sharpe=float(m.sharpe) if m.sharpe == m.sharpe else 0.0,
                oos_max_drawdown=float(m.max_drawdown) if m.max_drawdown == m.max_drawdown else 0.0,
                oos_n_trades=len(oos.trades) if not oos.trades.empty else 0,
            )
        )
        logger.info(
            f"  OOS: ret={oos.total_return:+.2%} "
            f"sharpe={results[-1].oos_sharpe:.2f} "
            f"dd={results[-1].oos_max_drawdown:.2%}",
        )
    return results


def _slice_data(data: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """按 trade_date 切片 (排除 end 当天，避免相邻 split 重叠)。"""
    ts_start = pd.Timestamp(start)
    ts_end = pd.Timestamp(end)
    df = data.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df[(df["trade_date"] >= ts_start) & (df["trade_date"] < ts_end)]
