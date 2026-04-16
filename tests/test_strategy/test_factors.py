"""因子测试 — 验证 ta 薄封装能算出预期的形状和值域。"""

from __future__ import annotations

from src.strategy.factors.momentum import MomentumFactor, RSIFactor
from src.strategy.factors.technical import BollFactor, MACDFactor, MAFactor


class TestMomentumFactors:
    def test_momentum_factor(self, sample_daily_data):
        factor = MomentumFactor(params={"period": 20})
        result = factor.calculate(sample_daily_data)
        assert len(result) == len(sample_daily_data)
        assert result.iloc[:20].isna().all()
        assert result.iloc[20:].notna().any()

    def test_rsi_factor(self, sample_daily_data):
        factor = RSIFactor(params={"period": 14})
        result = factor.calculate(sample_daily_data)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()


class TestTechnicalFactors:
    def test_ma_factor(self, sample_daily_data):
        factor = MAFactor(params={"period": 20})
        result = factor.calculate(sample_daily_data)
        assert len(result) == len(sample_daily_data)
        assert result.iloc[20:].notna().any()

    def test_macd_factor(self, sample_daily_data):
        factor = MACDFactor()
        result = factor.calculate(sample_daily_data)
        assert len(result) == len(sample_daily_data)

    def test_macd_full(self, sample_daily_data):
        factor = MACDFactor()
        dif, dea, macd = factor.calculate_full(sample_daily_data)
        assert len(dif) == len(sample_daily_data)
        assert len(dea) == len(sample_daily_data)
        assert len(macd) == len(sample_daily_data)

    def test_boll_factor(self, sample_daily_data):
        factor = BollFactor(params={"period": 20, "std_dev": 2})
        result = factor.calculate(sample_daily_data)
        assert result.dropna().size > 0

    def test_boll_bands(self, sample_daily_data):
        factor = BollFactor(params={"period": 20, "std_dev": 2})
        upper, middle, lower = factor.calculate_bands(sample_daily_data)
        valid = ~upper.isna()
        assert (upper[valid] >= middle[valid]).all()
        assert (middle[valid] >= lower[valid]).all()
