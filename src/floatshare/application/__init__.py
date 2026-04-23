"""应用层 — 用例编排：组合 interfaces / strategy / analytics / infrastructure。

应用层是组合根（composition root），允许 import infrastructure 来构造默认实例。
其它"上层"模块（除 cli 之外）禁止 import infrastructure。
"""

from floatshare.application.backtest import BacktestResult, run_backtest
from floatshare.application.data_loader import (
    AllSourcesFailed,
    DataLoader,
    create_default_loader,
)
from floatshare.application.data_syncer import DataSyncer as DataSyncer
from floatshare.application.data_syncer import apply_adjustment as apply_adjustment
from floatshare.application.treasury import (
    AccountNotFound,
    InsufficientFunds,
    PlanNotActive,
    PlanNotFound,
    Treasury,
)

__all__ = [
    "AccountNotFound",
    "AllSourcesFailed",
    "BacktestResult",
    "DataLoader",
    "DataSyncer",
    "InsufficientFunds",
    "PlanNotActive",
    "PlanNotFound",
    "Treasury",
    "apply_adjustment",
    "create_default_loader",
    "run_backtest",
]
