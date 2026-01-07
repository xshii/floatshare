"""信号生成器"""

from typing import List, Dict, Optional, Callable
import pandas as pd
import numpy as np

from ..base import Signal


class SignalGenerator:
    """交易信号生成器"""

    def __init__(self):
        self._rules: List[Dict] = []

    def add_rule(
        self,
        name: str,
        condition: Callable[[pd.DataFrame], pd.Series],
        direction: str,
        strength: float = 1.0,
    ) -> "SignalGenerator":
        """
        添加信号规则

        Args:
            name: 规则名称
            condition: 条件函数，返回布尔Series
            direction: 交易方向 (buy/sell)
            strength: 信号强度

        Returns:
            self，支持链式调用
        """
        self._rules.append(
            {
                "name": name,
                "condition": condition,
                "direction": direction,
                "strength": strength,
            }
        )
        return self

    def generate(
        self, data: pd.DataFrame, code_column: str = "code"
    ) -> List[Signal]:
        """
        生成交易信号

        Args:
            data: 市场数据
            code_column: 股票代码列名

        Returns:
            信号列表
        """
        signals = []

        for rule in self._rules:
            try:
                mask = rule["condition"](data)

                if mask.any():
                    triggered_data = data[mask]

                    for idx, row in triggered_data.iterrows():
                        code = row[code_column] if code_column in row else str(idx)

                        signal = Signal(
                            code=code,
                            direction=rule["direction"],
                            strength=rule["strength"],
                            price=row.get("close"),
                            reason=rule["name"],
                        )
                        signals.append(signal)

            except Exception as e:
                print(f"规则 {rule['name']} 执行失败: {e}")

        return signals

    def clear_rules(self) -> None:
        """清空所有规则"""
        self._rules.clear()


class CrossSignal:
    """交叉信号工具"""

    @staticmethod
    def golden_cross(fast: pd.Series, slow: pd.Series) -> pd.Series:
        """金叉信号"""
        return (fast > slow) & (fast.shift(1) <= slow.shift(1))

    @staticmethod
    def death_cross(fast: pd.Series, slow: pd.Series) -> pd.Series:
        """死叉信号"""
        return (fast < slow) & (fast.shift(1) >= slow.shift(1))

    @staticmethod
    def break_above(price: pd.Series, level: pd.Series) -> pd.Series:
        """向上突破"""
        return (price > level) & (price.shift(1) <= level.shift(1))

    @staticmethod
    def break_below(price: pd.Series, level: pd.Series) -> pd.Series:
        """向下突破"""
        return (price < level) & (price.shift(1) >= level.shift(1))


class ThresholdSignal:
    """阈值信号工具"""

    @staticmethod
    def above_threshold(values: pd.Series, threshold: float) -> pd.Series:
        """高于阈值"""
        return values > threshold

    @staticmethod
    def below_threshold(values: pd.Series, threshold: float) -> pd.Series:
        """低于阈值"""
        return values < threshold

    @staticmethod
    def in_range(
        values: pd.Series, lower: float, upper: float
    ) -> pd.Series:
        """在区间内"""
        return (values >= lower) & (values <= upper)

    @staticmethod
    def out_of_range(
        values: pd.Series, lower: float, upper: float
    ) -> pd.Series:
        """在区间外"""
        return (values < lower) | (values > upper)
