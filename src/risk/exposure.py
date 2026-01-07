"""敞口计算"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.execution.position import PositionDetail


@dataclass
class ExposureReport:
    """敞口报告"""

    total_exposure: float  # 总敞口
    net_exposure: float  # 净敞口
    long_exposure: float  # 多头敞口
    short_exposure: float  # 空头敞口（A股一般为0）
    beta_exposure: float  # Beta敞口
    industry_exposure: Dict[str, float]  # 行业敞口
    factor_exposure: Dict[str, float]  # 因子敞口


class ExposureCalculator:
    """敞口计算器"""

    def __init__(self):
        self._industry_map: Dict[str, str] = {}  # code -> industry
        self._beta_map: Dict[str, float] = {}  # code -> beta

    def set_industry_map(self, industry_map: Dict[str, str]) -> None:
        """设置行业映射"""
        self._industry_map = industry_map

    def set_beta_map(self, beta_map: Dict[str, float]) -> None:
        """设置Beta映射"""
        self._beta_map = beta_map

    def calculate(
        self,
        positions: Dict[str, PositionDetail],
        portfolio_value: float,
    ) -> ExposureReport:
        """计算敞口"""
        if not positions or portfolio_value <= 0:
            return ExposureReport(
                total_exposure=0,
                net_exposure=0,
                long_exposure=0,
                short_exposure=0,
                beta_exposure=0,
                industry_exposure={},
                factor_exposure={},
            )

        # 计算总敞口
        long_exposure = sum(p.market_value for p in positions.values() if p.quantity > 0)
        short_exposure = 0  # A股无空头

        total_exposure = long_exposure + short_exposure
        net_exposure = long_exposure - short_exposure

        # 计算行业敞口
        industry_exposure = self._calculate_industry_exposure(positions, portfolio_value)

        # 计算Beta敞口
        beta_exposure = self._calculate_beta_exposure(positions, portfolio_value)

        return ExposureReport(
            total_exposure=total_exposure / portfolio_value,
            net_exposure=net_exposure / portfolio_value,
            long_exposure=long_exposure / portfolio_value,
            short_exposure=short_exposure / portfolio_value,
            beta_exposure=beta_exposure,
            industry_exposure=industry_exposure,
            factor_exposure={},
        )

    def _calculate_industry_exposure(
        self,
        positions: Dict[str, PositionDetail],
        portfolio_value: float,
    ) -> Dict[str, float]:
        """计算行业敞口"""
        industry_values: Dict[str, float] = {}

        for code, pos in positions.items():
            industry = self._industry_map.get(code, "未知")
            if industry not in industry_values:
                industry_values[industry] = 0
            industry_values[industry] += pos.market_value

        return {
            ind: val / portfolio_value
            for ind, val in industry_values.items()
        }

    def _calculate_beta_exposure(
        self,
        positions: Dict[str, PositionDetail],
        portfolio_value: float,
    ) -> float:
        """计算加权Beta"""
        total_beta = 0

        for code, pos in positions.items():
            beta = self._beta_map.get(code, 1.0)
            weight = pos.market_value / portfolio_value if portfolio_value > 0 else 0
            total_beta += beta * weight

        return total_beta

    def calculate_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        position_value: float = 1_000_000,
    ) -> float:
        """
        计算VaR（风险价值）

        Args:
            returns: 历史收益率序列
            confidence: 置信度
            position_value: 持仓价值
        """
        if returns.empty:
            return 0.0

        var_pct = np.percentile(returns, (1 - confidence) * 100)
        return abs(var_pct * position_value)

    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        position_value: float = 1_000_000,
    ) -> float:
        """
        计算CVaR（条件风险价值）

        Args:
            returns: 历史收益率序列
            confidence: 置信度
            position_value: 持仓价值
        """
        if returns.empty:
            return 0.0

        var_pct = np.percentile(returns, (1 - confidence) * 100)
        cvar_pct = returns[returns <= var_pct].mean()
        return abs(cvar_pct * position_value)

    def calculate_correlation_matrix(
        self,
        returns_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """计算相关性矩阵"""
        return returns_df.corr()

    def check_concentration_risk(
        self,
        positions: Dict[str, PositionDetail],
        portfolio_value: float,
        threshold: float = 0.20,
    ) -> List[str]:
        """检查集中度风险"""
        warnings = []

        for code, pos in positions.items():
            concentration = pos.market_value / portfolio_value if portfolio_value > 0 else 0
            if concentration > threshold:
                warnings.append(
                    f"{code} 持仓占比 {concentration:.2%} 超过阈值 {threshold:.2%}"
                )

        return warnings
