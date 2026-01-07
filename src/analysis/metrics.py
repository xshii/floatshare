"""绩效指标计算"""

from typing import Optional, Dict
import pandas as pd
import numpy as np


class PerformanceMetrics:
    """绩效指标计算器"""

    def __init__(self, risk_free_rate: float = 0.03):
        """
        初始化

        Args:
            risk_free_rate: 年化无风险利率
        """
        self.risk_free_rate = risk_free_rate
        self.daily_rf = risk_free_rate / 252

    def calculate_all(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> Dict:
        """计算所有指标"""
        metrics = {
            "total_return": self.total_return(returns),
            "annual_return": self.annual_return(returns),
            "volatility": self.volatility(returns),
            "sharpe_ratio": self.sharpe_ratio(returns),
            "sortino_ratio": self.sortino_ratio(returns),
            "max_drawdown": self.max_drawdown(returns),
            "calmar_ratio": self.calmar_ratio(returns),
            "win_rate": self.win_rate(returns),
            "profit_loss_ratio": self.profit_loss_ratio(returns),
            "var_95": self.value_at_risk(returns, 0.95),
            "cvar_95": self.conditional_var(returns, 0.95),
        }

        if benchmark_returns is not None:
            metrics.update({
                "alpha": self.alpha(returns, benchmark_returns),
                "beta": self.beta(returns, benchmark_returns),
                "information_ratio": self.information_ratio(returns, benchmark_returns),
                "tracking_error": self.tracking_error(returns, benchmark_returns),
            })

        return metrics

    def total_return(self, returns: pd.Series) -> float:
        """总收益率"""
        return (1 + returns).prod() - 1

    def annual_return(self, returns: pd.Series) -> float:
        """年化收益率"""
        total = self.total_return(returns)
        days = len(returns)
        if days <= 0:
            return 0.0
        return (1 + total) ** (252 / days) - 1

    def volatility(self, returns: pd.Series) -> float:
        """年化波动率"""
        return returns.std() * np.sqrt(252)

    def sharpe_ratio(self, returns: pd.Series) -> float:
        """夏普比率"""
        excess_returns = returns - self.daily_rf
        if returns.std() == 0:
            return 0.0
        return np.sqrt(252) * excess_returns.mean() / returns.std()

    def sortino_ratio(self, returns: pd.Series) -> float:
        """索提诺比率"""
        excess_returns = returns - self.daily_rf
        downside = returns[returns < 0]
        if len(downside) == 0 or downside.std() == 0:
            return 0.0
        return np.sqrt(252) * excess_returns.mean() / downside.std()

    def max_drawdown(self, returns: pd.Series) -> float:
        """最大回撤"""
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()

    def calmar_ratio(self, returns: pd.Series) -> float:
        """卡尔玛比率"""
        mdd = abs(self.max_drawdown(returns))
        if mdd == 0:
            return 0.0
        return self.annual_return(returns) / mdd

    def win_rate(self, returns: pd.Series) -> float:
        """胜率"""
        if len(returns) == 0:
            return 0.0
        return (returns > 0).sum() / len(returns)

    def profit_loss_ratio(self, returns: pd.Series) -> float:
        """盈亏比"""
        gains = returns[returns > 0]
        losses = returns[returns < 0]

        if len(losses) == 0 or losses.mean() == 0:
            return float("inf") if len(gains) > 0 else 0.0

        return abs(gains.mean() / losses.mean())

    def value_at_risk(
        self, returns: pd.Series, confidence: float = 0.95
    ) -> float:
        """VaR（风险价值）"""
        return np.percentile(returns, (1 - confidence) * 100)

    def conditional_var(
        self, returns: pd.Series, confidence: float = 0.95
    ) -> float:
        """CVaR（条件风险价值）"""
        var = self.value_at_risk(returns, confidence)
        return returns[returns <= var].mean()

    def alpha(
        self, returns: pd.Series, benchmark_returns: pd.Series
    ) -> float:
        """Alpha（超额收益）"""
        beta = self.beta(returns, benchmark_returns)
        return self.annual_return(returns) - (
            self.risk_free_rate + beta * (self.annual_return(benchmark_returns) - self.risk_free_rate)
        )

    def beta(
        self, returns: pd.Series, benchmark_returns: pd.Series
    ) -> float:
        """Beta（市场敏感度）"""
        # 对齐数据
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned) < 2:
            return 1.0

        cov = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
        var = aligned.iloc[:, 1].var()

        if var == 0:
            return 1.0
        return cov / var

    def information_ratio(
        self, returns: pd.Series, benchmark_returns: pd.Series
    ) -> float:
        """信息比率"""
        te = self.tracking_error(returns, benchmark_returns)
        if te == 0:
            return 0.0

        excess = returns - benchmark_returns
        return np.sqrt(252) * excess.mean() / te

    def tracking_error(
        self, returns: pd.Series, benchmark_returns: pd.Series
    ) -> float:
        """跟踪误差"""
        excess = returns - benchmark_returns
        return excess.std() * np.sqrt(252)

    def downside_deviation(self, returns: pd.Series) -> float:
        """下行标准差"""
        downside = returns[returns < 0]
        if len(downside) == 0:
            return 0.0
        return downside.std() * np.sqrt(252)

    def skewness(self, returns: pd.Series) -> float:
        """偏度"""
        return returns.skew()

    def kurtosis(self, returns: pd.Series) -> float:
        """峰度"""
        return returns.kurtosis()

    def recovery_time(self, returns: pd.Series) -> Optional[int]:
        """最大回撤恢复时间（天）"""
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak

        # 找到最大回撤位置
        max_dd_idx = drawdown.idxmin()
        max_dd_value = drawdown[max_dd_idx]

        # 找到恢复位置
        recovery = cumulative[max_dd_idx:][cumulative[max_dd_idx:] >= peak[max_dd_idx]]

        if len(recovery) == 0:
            return None

        recovery_idx = recovery.index[0]
        return (recovery_idx - max_dd_idx).days if hasattr(recovery_idx - max_dd_idx, 'days') else None
