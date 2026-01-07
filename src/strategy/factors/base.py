"""因子基类"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np


class Factor(ABC):
    """因子基类"""

    name: str = "BaseFactor"
    description: str = ""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算因子值

        Args:
            data: 输入数据，包含OHLCV等字段

        Returns:
            因子值Series
        """
        pass

    def normalize(self, values: pd.Series, method: str = "zscore") -> pd.Series:
        """
        因子值标准化

        Args:
            values: 原始因子值
            method: 标准化方法 (zscore, minmax, rank)
        """
        if method == "zscore":
            return (values - values.mean()) / values.std()
        elif method == "minmax":
            return (values - values.min()) / (values.max() - values.min())
        elif method == "rank":
            return values.rank(pct=True)
        else:
            return values

    def winsorize(
        self, values: pd.Series, lower: float = 0.01, upper: float = 0.99
    ) -> pd.Series:
        """去极值"""
        lower_bound = values.quantile(lower)
        upper_bound = values.quantile(upper)
        return values.clip(lower_bound, upper_bound)

    def neutralize(
        self, values: pd.Series, groups: pd.Series
    ) -> pd.Series:
        """行业/市值中性化"""
        result = values.copy()

        for group in groups.unique():
            mask = groups == group
            group_values = values[mask]
            result[mask] = group_values - group_values.mean()

        return result

    def get_param(self, key: str, default: Any = None) -> Any:
        """获取参数"""
        return self.params.get(key, default)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({self.name})>"
