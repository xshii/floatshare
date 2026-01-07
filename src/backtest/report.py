"""回测报告"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np


@dataclass
class BacktestReport:
    """回测报告"""

    daily_data: pd.DataFrame
    trades: pd.DataFrame
    initial_capital: float
    final_value: float
    benchmark: Optional[pd.DataFrame] = None

    @property
    def total_return(self) -> float:
        """总收益率"""
        return (self.final_value / self.initial_capital) - 1

    @property
    def annual_return(self) -> float:
        """年化收益率"""
        if self.daily_data.empty:
            return 0.0

        days = len(self.daily_data)
        if days <= 0:
            return 0.0

        return (1 + self.total_return) ** (252 / days) - 1

    @property
    def max_drawdown(self) -> float:
        """最大回撤"""
        if self.daily_data.empty:
            return 0.0

        portfolio_values = self.daily_data["portfolio_value"]
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return drawdown.min()

    @property
    def sharpe_ratio(self) -> float:
        """夏普比率（假设无风险利率为3%）"""
        if self.daily_data.empty or len(self.daily_data) < 2:
            return 0.0

        daily_returns = self.daily_data["portfolio_value"].pct_change().dropna()

        if daily_returns.std() == 0:
            return 0.0

        risk_free_rate = 0.03 / 252  # 日无风险利率
        excess_returns = daily_returns - risk_free_rate
        return np.sqrt(252) * excess_returns.mean() / daily_returns.std()

    @property
    def sortino_ratio(self) -> float:
        """索提诺比率"""
        if self.daily_data.empty or len(self.daily_data) < 2:
            return 0.0

        daily_returns = self.daily_data["portfolio_value"].pct_change().dropna()
        risk_free_rate = 0.03 / 252

        excess_returns = daily_returns - risk_free_rate
        downside_returns = daily_returns[daily_returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()

    @property
    def calmar_ratio(self) -> float:
        """卡尔玛比率"""
        if self.max_drawdown == 0:
            return 0.0
        return self.annual_return / abs(self.max_drawdown)

    @property
    def win_rate(self) -> float:
        """胜率"""
        if self.trades.empty:
            return 0.0

        # 计算每笔交易的盈亏
        # 简化处理：假设trades中已有profit字段，或通过买卖配对计算
        wins = len(self.trades[self.trades.get("profit", 0) > 0])
        total = len(self.trades)

        return wins / total if total > 0 else 0.0

    @property
    def trade_count(self) -> int:
        """交易次数"""
        return len(self.trades)

    @property
    def volatility(self) -> float:
        """年化波动率"""
        if self.daily_data.empty or len(self.daily_data) < 2:
            return 0.0

        daily_returns = self.daily_data["portfolio_value"].pct_change().dropna()
        return daily_returns.std() * np.sqrt(252)

    def summary(self) -> Dict[str, Any]:
        """获取摘要统计"""
        return {
            "初始资金": self.initial_capital,
            "最终市值": self.final_value,
            "总收益率": f"{self.total_return:.2%}",
            "年化收益率": f"{self.annual_return:.2%}",
            "最大回撤": f"{self.max_drawdown:.2%}",
            "夏普比率": f"{self.sharpe_ratio:.2f}",
            "索提诺比率": f"{self.sortino_ratio:.2f}",
            "卡尔玛比率": f"{self.calmar_ratio:.2f}",
            "年化波动率": f"{self.volatility:.2%}",
            "交易次数": self.trade_count,
            "胜率": f"{self.win_rate:.2%}",
        }

    def print_summary(self) -> None:
        """打印摘要"""
        print("\n" + "=" * 50)
        print("回测报告")
        print("=" * 50)

        for key, value in self.summary().items():
            print(f"{key}: {value}")

        print("=" * 50)

    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        return pd.DataFrame([self.summary()])

    def get_monthly_returns(self) -> pd.DataFrame:
        """获取月度收益"""
        if self.daily_data.empty:
            return pd.DataFrame()

        df = self.daily_data.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        monthly = df["portfolio_value"].resample("M").last()
        monthly_returns = monthly.pct_change()

        return monthly_returns.to_frame("return")

    def get_drawdown_series(self) -> pd.Series:
        """获取回撤序列"""
        if self.daily_data.empty:
            return pd.Series()

        portfolio_values = self.daily_data["portfolio_value"]
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak

        return drawdown
