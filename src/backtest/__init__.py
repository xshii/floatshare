"""回测工具 — 基于 backtrader 的薄封装。"""

from .runner import BacktestResult, run_backtest

__all__ = ["BacktestResult", "run_backtest"]
