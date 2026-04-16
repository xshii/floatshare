"""回测用例 — 把 floatshare 的标准 OHLCV DataFrame 喂给 backtrader。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, cast

import backtrader as bt
import pandas as pd

from floatshare.analytics import Metrics, metrics
from floatshare.domain.trading import TradingConfig
from floatshare.infrastructure.broker import AStockCommission
from floatshare.observability import logger


def _df_to_feed(df: pd.DataFrame, name: str) -> bt.feeds.PandasData:
    """把 floatshare 标准 OHLCV DataFrame 转成 backtrader feed。"""
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").set_index("trade_date")
    return bt.feeds.PandasData(
        dataname=df,
        name=name,
        datetime=None,
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        openinterest=-1,
    )


@dataclass(slots=True)
class BacktestResult:
    """回测结果。"""

    initial_capital: float
    final_value: float
    returns: pd.Series
    daily_data: pd.DataFrame
    metrics: Metrics
    cerebro: bt.Cerebro | None = field(default=None, repr=False)

    @property
    def total_return(self) -> float:
        return float(self.metrics.total_return)

    @property
    def annual_return(self) -> float:
        return float(self.metrics.cagr)

    @property
    def max_drawdown(self) -> float:
        return float(self.metrics.max_drawdown)

    @property
    def sharpe_ratio(self) -> float:
        return float(self.metrics.sharpe)

    def print_summary(self) -> None:
        logger.info("=" * 50)
        logger.info(f"初始资金: {self.initial_capital:,.2f}")
        logger.info(f"最终市值: {self.final_value:,.2f}")
        for k, v in self.metrics.to_dict().items():
            if isinstance(v, (int, float)):
                logger.info(f"  {k}: {v:.4f}")
            else:
                logger.info(f"  {k}: {v}")
        logger.info("=" * 50)


def run_backtest(
    strategy_cls: type[bt.Strategy],
    data: pd.DataFrame,
    initial_capital: float = 1_000_000,
    strategy_params: dict[str, Any] | None = None,
    trading_config: TradingConfig | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> BacktestResult:
    """运行 backtrader 回测。

    Args:
        strategy_cls: backtrader.Strategy 子类
        data: 标准 OHLCV DataFrame，必须含 code/trade_date/open/high/low/close/volume 列
        initial_capital: 初始资金
        strategy_params: 策略参数 dict
        trading_config: A 股交易费率配置
        start_date / end_date: 可选过滤窗口
    """
    cfg = trading_config or TradingConfig()
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.addcommissioninfo(
        AStockCommission(
            commission=cfg.commission_rate,
            stamp_duty=cfg.stamp_duty,
            transfer_fee=cfg.transfer_fee,
            min_commission=cfg.min_commission,
        )
    )
    cerebro.broker.set_slippage_perc(cfg.slippage)

    if start_date is not None or end_date is not None:
        data = data.copy()
        data["trade_date"] = pd.to_datetime(data["trade_date"])
        if start_date is not None:
            data = cast(pd.DataFrame, data[data["trade_date"] >= pd.Timestamp(start_date)])
        if end_date is not None:
            data = cast(pd.DataFrame, data[data["trade_date"] <= pd.Timestamp(end_date)])

    for code in data["code"].unique():
        sub = cast(pd.DataFrame, data[data["code"] == code])
        if not sub.empty:
            cerebro.adddata(_df_to_feed(sub, name=str(code)))

    cerebro.addstrategy(strategy_cls, **(strategy_params or {}))
    cerebro.addanalyzer(
        bt.analyzers.TimeReturn,
        _name="timereturn",
        timeframe=bt.TimeFrame.Days,
    )

    results = cerebro.run()
    strat = results[0]
    # backtrader 在 strategy 实例上动态挂 .analyzers，stub 没法静态描述
    analyzer = strat.analyzers.timereturn  # pyright: ignore[reportAttributeAccessIssue]
    timereturn: dict[Any, float] = analyzer.get_analysis()
    if timereturn:
        returns = pd.Series(timereturn).sort_index()
        returns.index = pd.to_datetime(returns.index)
    else:
        returns = pd.Series(dtype=float)

    final_value = float(cerebro.broker.getvalue())
    daily_data = pd.DataFrame(
        {
            "date": returns.index,
            "return": returns.values,
            "cum_return": (1 + returns).cumprod().values if len(returns) else [],
        }
    )
    from dataclasses import replace

    snapshot = metrics(returns) if len(returns) > 1 else _empty_metrics()
    if len(returns) > 1:
        # quantstats 在某些指标上不会带 total_return，用真实终值覆盖
        snapshot = replace(
            snapshot, total_return=(final_value - initial_capital) / initial_capital
        )

    return BacktestResult(
        initial_capital=initial_capital,
        final_value=final_value,
        returns=returns,
        daily_data=daily_data,
        metrics=snapshot,
        cerebro=cerebro,
    )


def _empty_metrics() -> Metrics:
    nan = float("nan")
    return Metrics(
        total_return=0.0,
        cagr=nan,
        volatility=nan,
        sharpe=nan,
        sortino=nan,
        max_drawdown=nan,
        calmar=nan,
        win_rate=nan,
        profit_factor=nan,
        var_95=nan,
        cvar_95=nan,
    )
