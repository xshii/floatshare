"""因子契约 + 单元测试 — 用 parametrize 自动覆盖所有 Factor。"""

from __future__ import annotations

import pytest

from floatshare.factors import Factor
from floatshare.factors.momentum import (
    MomentumFactor,
    ReversalFactor,
    RSIFactor,
    VolumeMomentumFactor,
)
from floatshare.factors.technical import (
    ATRFactor,
    BollFactor,
    KDJFactor,
    MACDFactor,
    MAFactor,
)

ALL_FACTORS = [
    MAFactor(),
    MACDFactor(),
    BollFactor(),
    ATRFactor(),
    KDJFactor(),
    MomentumFactor(),
    RSIFactor(),
    ReversalFactor(),
    VolumeMomentumFactor(),
]


@pytest.mark.parametrize("factor", ALL_FACTORS, ids=lambda f: f.name)
class TestFactorContract:
    """所有因子必须满足的契约：返回长度对齐 + 是 Factor Protocol。"""

    def test_is_factor_protocol(self, factor):
        assert isinstance(factor, Factor)

    def test_returns_series_aligned(self, factor, sample_daily_data):
        result = factor.calculate(sample_daily_data)
        assert len(result) == len(sample_daily_data)

    def test_has_some_non_na_values(self, factor, sample_daily_data):
        result = factor.calculate(sample_daily_data)
        assert result.notna().any()


class TestSpecificFactors:
    def test_rsi_in_range(self, sample_daily_data):
        result = RSIFactor(period=14).calculate(sample_daily_data)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_macd_full_three_series(self, sample_daily_data):
        dif, dea, macd = MACDFactor().calculate_full(sample_daily_data)
        assert len(dif) == len(sample_daily_data)
        assert len(dea) == len(sample_daily_data)
        assert len(macd) == len(sample_daily_data)

    def test_boll_bands_ordered(self, sample_daily_data):
        upper, middle, lower = BollFactor(period=20).calculate_bands(sample_daily_data)
        valid = ~upper.isna()
        assert (upper[valid] >= middle[valid]).all()
        assert (middle[valid] >= lower[valid]).all()

    def test_momentum_first_n_are_nan(self, sample_daily_data):
        result = MomentumFactor(period=20).calculate(sample_daily_data)
        assert result.iloc[:20].isna().all()
        assert result.iloc[20:].notna().any()
