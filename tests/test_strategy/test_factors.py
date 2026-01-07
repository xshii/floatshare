"""因子测试"""

import pytest
import pandas as pd
import numpy as np
from src.strategy.factors.momentum import MomentumFactor, RSIFactor
from src.strategy.factors.technical import MAFactor, MACDFactor, BollFactor


class TestMomentumFactors:
    """动量因子测试"""

    def test_momentum_factor(self, sample_daily_data):
        """测试动量因子"""
        factor = MomentumFactor(params={"period": 20})
        result = factor.calculate(sample_daily_data)

        assert len(result) == len(sample_daily_data)
        assert result.iloc[:20].isna().all()  # 前20个应该是NaN
        assert not result.iloc[20:].isna().all()

    def test_rsi_factor(self, sample_daily_data):
        """测试RSI因子"""
        factor = RSIFactor(params={"period": 14})
        result = factor.calculate(sample_daily_data)

        # RSI应该在0-100之间
        valid_values = result.dropna()
        assert (valid_values >= 0).all()
        assert (valid_values <= 100).all()


class TestTechnicalFactors:
    """技术指标因子测试"""

    def test_ma_factor(self, sample_daily_data):
        """测试均线因子"""
        factor = MAFactor(params={"period": 20})
        result = factor.calculate(sample_daily_data)

        assert len(result) == len(sample_daily_data)

    def test_macd_factor(self, sample_daily_data):
        """测试MACD因子"""
        factor = MACDFactor()
        result = factor.calculate(sample_daily_data)

        assert len(result) == len(sample_daily_data)

    def test_macd_full(self, sample_daily_data):
        """测试完整MACD"""
        factor = MACDFactor()
        dif, dea, macd = factor.calculate_full(sample_daily_data)

        assert len(dif) == len(sample_daily_data)
        assert len(dea) == len(sample_daily_data)
        assert len(macd) == len(sample_daily_data)

    def test_boll_factor(self, sample_daily_data):
        """测试布林带因子"""
        factor = BollFactor(params={"period": 20, "std_dev": 2})
        result = factor.calculate(sample_daily_data)

        # 布林带位置应该主要在0-1之间
        valid_values = result.dropna()
        assert len(valid_values) > 0

    def test_boll_bands(self, sample_daily_data):
        """测试布林带三轨"""
        factor = BollFactor(params={"period": 20, "std_dev": 2})
        upper, middle, lower = factor.calculate_bands(sample_daily_data)

        # 上轨应该大于中轨，中轨应该大于下轨
        valid_idx = ~upper.isna()
        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()
