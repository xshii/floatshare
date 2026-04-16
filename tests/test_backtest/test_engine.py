"""回测引擎测试 — 基于 backtrader runner。"""

from __future__ import annotations

import backtrader as bt

from src.backtest import BacktestResult, run_backtest
from src.strategy.registry import StrategyRegistry


class BuyAndHoldStrategy(bt.Strategy):
    """最简单的 buy-and-hold：第一根 bar 就买。"""

    name = "BuyAndHold"

    def next(self) -> None:
        for d in self.datas:
            if not self.getposition(d).size:
                cash = self.broker.getcash()
                target = cash * 0.5 / len(self.datas)
                size = int(target / d.close[0] / 100) * 100
                if size > 0:
                    self.buy(data=d, size=size)


class TestBacktestRunner:
    def test_run_single_stock(self, sample_daily_data):
        result = run_backtest(
            strategy_cls=BuyAndHoldStrategy,
            data=sample_daily_data,
            initial_capital=1_000_000,
        )
        assert isinstance(result, BacktestResult)
        assert result.initial_capital == 1_000_000
        assert result.final_value > 0
        assert len(result.returns) > 0

    def test_run_multi_stock(self, multi_stock_data):
        result = run_backtest(
            strategy_cls=BuyAndHoldStrategy,
            data=multi_stock_data,
            initial_capital=1_000_000,
        )
        assert result.final_value > 0
        assert len(result.daily_data) > 0

    def test_metrics_are_floats(self, sample_daily_data):
        result = run_backtest(
            strategy_cls=BuyAndHoldStrategy,
            data=sample_daily_data,
            initial_capital=1_000_000,
        )
        assert isinstance(result.total_return, float)
        assert isinstance(result.annual_return, float)
        assert isinstance(result.max_drawdown, float)
        assert isinstance(result.sharpe_ratio, float)


class TestRegisteredStrategies:
    def test_ma_cross_runs(self, sample_daily_data):
        # 触发注册
        import strategies  # noqa: F401

        cls = StrategyRegistry.get("ma_cross")
        assert cls is not None
        result = run_backtest(
            strategy_cls=cls,
            data=sample_daily_data,
            initial_capital=1_000_000,
            strategy_params={"short_period": 5, "long_period": 20},
        )
        assert result.final_value > 0

    def test_dual_thrust_runs(self, sample_daily_data):
        import strategies  # noqa: F401

        cls = StrategyRegistry.get("dual_thrust")
        assert cls is not None
        result = run_backtest(
            strategy_cls=cls,
            data=sample_daily_data,
            initial_capital=1_000_000,
        )
        assert result.final_value > 0
