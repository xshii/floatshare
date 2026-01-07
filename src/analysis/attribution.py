"""归因分析"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np


class AttributionAnalyzer:
    """归因分析器"""

    def __init__(self):
        pass

    def brinson_attribution(
        self,
        portfolio_weights: pd.DataFrame,
        portfolio_returns: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        benchmark_returns: pd.DataFrame,
    ) -> Dict:
        """
        Brinson归因分析

        Args:
            portfolio_weights: 组合权重 (date x sector)
            portfolio_returns: 组合收益 (date x sector)
            benchmark_weights: 基准权重 (date x sector)
            benchmark_returns: 基准收益 (date x sector)

        Returns:
            归因结果字典
        """
        # 配置效应（Allocation Effect）
        # 选择效应（Selection Effect）
        # 交互效应（Interaction Effect）

        allocation = (portfolio_weights - benchmark_weights) * benchmark_returns
        selection = benchmark_weights * (portfolio_returns - benchmark_returns)
        interaction = (portfolio_weights - benchmark_weights) * (portfolio_returns - benchmark_returns)

        return {
            "allocation": allocation.sum().sum(),
            "selection": selection.sum().sum(),
            "interaction": interaction.sum().sum(),
            "total_active": allocation.sum().sum() + selection.sum().sum() + interaction.sum().sum(),
            "allocation_by_sector": allocation.sum().to_dict(),
            "selection_by_sector": selection.sum().to_dict(),
        }

    def factor_attribution(
        self,
        returns: pd.Series,
        factor_returns: pd.DataFrame,
    ) -> Dict:
        """
        因子归因分析

        Args:
            returns: 组合收益序列
            factor_returns: 因子收益DataFrame

        Returns:
            因子暴露和贡献
        """
        # 对齐数据
        aligned = pd.concat([returns, factor_returns], axis=1).dropna()

        if len(aligned) < 10:
            return {"error": "数据不足"}

        y = aligned.iloc[:, 0]
        X = aligned.iloc[:, 1:]

        # 添加常数项
        X = pd.concat([pd.Series(1, index=X.index, name="alpha"), X], axis=1)

        # 回归
        try:
            from numpy.linalg import lstsq

            coeffs, residuals, rank, s = lstsq(X.values, y.values, rcond=None)

            factor_names = X.columns.tolist()
            exposures = dict(zip(factor_names, coeffs))

            # 计算因子贡献
            contributions = {}
            for i, factor in enumerate(factor_names):
                contributions[factor] = coeffs[i] * X[factor].mean()

            return {
                "exposures": exposures,
                "contributions": contributions,
                "r_squared": 1 - residuals[0] / ((y - y.mean()) ** 2).sum() if len(residuals) > 0 else 0,
            }
        except Exception as e:
            return {"error": str(e)}

    def sector_attribution(
        self,
        positions: pd.DataFrame,
        returns: pd.DataFrame,
        sector_map: Dict[str, str],
    ) -> Dict:
        """
        行业归因

        Args:
            positions: 持仓市值 (date x code)
            returns: 个股收益 (date x code)
            sector_map: 股票-行业映射
        """
        # 按行业分组
        sector_returns = {}
        sector_weights = {}

        for code in positions.columns:
            sector = sector_map.get(code, "其他")

            if sector not in sector_returns:
                sector_returns[sector] = []
                sector_weights[sector] = []

            # 计算该股票对行业的贡献
            stock_weight = positions[code] / positions.sum(axis=1)
            stock_return = returns.get(code, pd.Series(0, index=positions.index))

            sector_returns[sector].append(stock_weight * stock_return)
            sector_weights[sector].append(stock_weight)

        # 汇总
        result = {}
        for sector in sector_returns:
            total_return = sum(sector_returns[sector])
            total_weight = sum(sector_weights[sector])

            result[sector] = {
                "weight": total_weight.mean(),
                "return": total_return.sum(),
                "contribution": (total_return).sum(),
            }

        return result

    def timing_attribution(
        self,
        position_ratios: pd.Series,
        market_returns: pd.Series,
    ) -> Dict:
        """
        择时归因

        Args:
            position_ratios: 仓位比例时间序列
            market_returns: 市场收益时间序列
        """
        # 对齐数据
        aligned = pd.concat([position_ratios, market_returns], axis=1).dropna()
        aligned.columns = ["position", "market"]

        # 计算择时收益
        # 如果在市场上涨时高仓位，下跌时低仓位，则有正的择时收益
        avg_position = aligned["position"].mean()
        timing_return = ((aligned["position"] - avg_position) * aligned["market"]).sum()

        # 分析仓位与市场的相关性
        correlation = aligned["position"].corr(aligned["market"])

        return {
            "timing_return": timing_return,
            "position_market_correlation": correlation,
            "avg_position": avg_position,
            "position_std": aligned["position"].std(),
        }
