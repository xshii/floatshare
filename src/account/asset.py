"""资产管理"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import date, datetime
import pandas as pd


@dataclass
class AssetSnapshot:
    """资产快照"""

    date: date
    cash: float
    position_value: float
    total_value: float
    positions: Dict[str, Dict] = field(default_factory=dict)


class AssetManager:
    """资产管理器"""

    def __init__(self, initial_capital: float = 1_000_000):
        self.initial_capital = initial_capital
        self._snapshots: List[AssetSnapshot] = []
        self._peak_value: float = initial_capital

    def take_snapshot(
        self,
        date_: date,
        cash: float,
        positions: Dict,
    ) -> AssetSnapshot:
        """记录资产快照"""
        position_value = sum(
            p.get("market_value", 0) if isinstance(p, dict) else p.market_value
            for p in positions.values()
        )
        total_value = cash + position_value

        snapshot = AssetSnapshot(
            date=date_,
            cash=cash,
            position_value=position_value,
            total_value=total_value,
            positions={
                code: {
                    "quantity": p.get("quantity", 0) if isinstance(p, dict) else p.quantity,
                    "market_value": p.get("market_value", 0) if isinstance(p, dict) else p.market_value,
                }
                for code, p in positions.items()
            },
        )

        self._snapshots.append(snapshot)

        # 更新峰值
        if total_value > self._peak_value:
            self._peak_value = total_value

        return snapshot

    def get_history(self) -> pd.DataFrame:
        """获取历史记录"""
        if not self._snapshots:
            return pd.DataFrame()

        data = [
            {
                "date": s.date,
                "cash": s.cash,
                "position_value": s.position_value,
                "total_value": s.total_value,
            }
            for s in self._snapshots
        ]

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])

        # 计算收益率
        df["return"] = df["total_value"].pct_change()
        df["cumulative_return"] = (df["total_value"] / self.initial_capital) - 1

        # 计算回撤
        df["peak"] = df["total_value"].expanding().max()
        df["drawdown"] = (df["total_value"] - df["peak"]) / df["peak"]

        return df

    @property
    def current_value(self) -> float:
        """当前市值"""
        if not self._snapshots:
            return self.initial_capital
        return self._snapshots[-1].total_value

    @property
    def current_drawdown(self) -> float:
        """当前回撤"""
        if self._peak_value <= 0:
            return 0.0
        return (self.current_value - self._peak_value) / self._peak_value

    @property
    def max_drawdown(self) -> float:
        """最大回撤"""
        history = self.get_history()
        if history.empty:
            return 0.0
        return history["drawdown"].min()

    @property
    def total_return(self) -> float:
        """总收益率"""
        return (self.current_value / self.initial_capital) - 1

    def get_monthly_returns(self) -> pd.Series:
        """获取月度收益"""
        history = self.get_history()
        if history.empty:
            return pd.Series()

        history = history.set_index("date")
        monthly = history["total_value"].resample("M").last()
        return monthly.pct_change()

    def get_annual_returns(self) -> pd.Series:
        """获取年度收益"""
        history = self.get_history()
        if history.empty:
            return pd.Series()

        history = history.set_index("date")
        annual = history["total_value"].resample("Y").last()
        return annual.pct_change()

    def clear(self) -> None:
        """清空记录"""
        self._snapshots.clear()
        self._peak_value = self.initial_capital
