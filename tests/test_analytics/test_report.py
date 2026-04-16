"""quantstats 封装的 Metrics 测试。"""

from __future__ import annotations

import math

from floatshare.analytics import Metrics, metrics


class TestMetrics:
    def test_metrics_returns_typed_dataclass(self, sample_returns):
        m = metrics(sample_returns)
        assert isinstance(m, Metrics)
        assert isinstance(m.sharpe, float)
        assert isinstance(m.max_drawdown, float)

    def test_to_dict_contains_all_fields(self, sample_returns):
        d = metrics(sample_returns).to_dict()
        for k in (
            "total_return",
            "cagr",
            "volatility",
            "sharpe",
            "sortino",
            "max_drawdown",
            "calmar",
            "win_rate",
            "profit_factor",
            "var_95",
            "cvar_95",
        ):
            assert k in d

    def test_with_benchmark_populates_alpha_beta(self, sample_returns):
        m = metrics(sample_returns, benchmark=sample_returns * 0.9 + 0.0001)
        # benchmark 提供时应该有 alpha/beta
        assert m.alpha is not None
        assert m.beta is not None

    def test_short_series_returns_nan_safely(self):
        import pandas as pd

        m = metrics(pd.Series([0.001, -0.002], index=pd.date_range("2024-01-01", periods=2)))
        assert isinstance(m, Metrics)
        # 不抛异常即可，部分指标可能 NaN
        assert math.isnan(m.cvar_95) or isinstance(m.cvar_95, float)
