"""回测引擎测试"""

import pytest
import pandas as pd
from datetime import date

from src.backtest.engine import BacktestEngine
from src.strategy.base import Strategy, Signal, StrategyContext


class SimpleStrategy(Strategy):
    """简单测试策略"""

    name = "SimpleTest"

    def init(self, context: StrategyContext) -> None:
        pass

    def handle_data(self, context: StrategyContext, data: pd.DataFrame) -> list:
        signals = []

        for code in data["code"].unique():
            has_position = code in context.positions and context.positions[code] > 0

            # 简单逻辑：没持仓就买，有持仓就持有
            if not has_position:
                signals.append(Signal(
                    code=code,
                    direction="buy",
                    strength=0.5,
                    price=data[data["code"] == code]["close"].iloc[0],
                ))

        return signals


class TestBacktestEngine:
    """回测引擎测试"""

    def test_engine_initialization(self):
        """测试引擎初始化"""
        engine = BacktestEngine(
            initial_capital=1_000_000,
            commission=0.0003,
            slippage=0.001,
        )

        assert engine.initial_capital == 1_000_000
        assert engine.trading_config.commission_rate == 0.0003
        assert engine.trading_config.slippage == 0.001

    def test_run_backtest(self, sample_daily_data):
        """测试运行回测"""
        engine = BacktestEngine(initial_capital=1_000_000)
        strategy = SimpleStrategy()

        report = engine.run(
            strategy=strategy,
            data=sample_daily_data,
        )

        assert report is not None
        assert report.initial_capital == 1_000_000
        assert report.final_value > 0

    def test_backtest_with_multiple_stocks(self, multi_stock_data):
        """测试多股票回测"""
        engine = BacktestEngine(initial_capital=1_000_000)
        strategy = SimpleStrategy()

        report = engine.run(
            strategy=strategy,
            data=multi_stock_data,
        )

        assert report is not None
        assert len(report.daily_data) > 0

    def test_backtest_report_metrics(self, sample_daily_data):
        """测试回测报告指标"""
        engine = BacktestEngine(initial_capital=1_000_000)
        strategy = SimpleStrategy()

        report = engine.run(
            strategy=strategy,
            data=sample_daily_data,
        )

        # 检查指标计算
        assert isinstance(report.total_return, float)
        assert isinstance(report.annual_return, float)
        assert isinstance(report.max_drawdown, float)
        assert isinstance(report.sharpe_ratio, float)
