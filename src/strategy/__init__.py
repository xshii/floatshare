"""策略系统 — 直接基于 backtrader.Strategy。"""

import backtrader as bt

from .registry import StrategyRegistry

# 重新导出 backtrader.Strategy 作为项目内的策略基类别名
Strategy = bt.Strategy

__all__ = ["Strategy", "StrategyRegistry"]
