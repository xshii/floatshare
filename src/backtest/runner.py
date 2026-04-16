"""backtrader 运行器 — 把 floatshare 的 OHLCV DataFrame 喂给 cerebro。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Iterable, List, Optional, Type

import backtrader as bt
import pandas as pd

from config.trading import TradingConfig
from src.analysis import metrics as compute_metrics
from src.monitor import logger


def _df_to_feed(df: pd.DataFrame, name: str) -> bt.feeds.PandasData:
    """把 floatshare 标准 OHLCV DataFrame 转成 backtrader feed。"""
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").set_index("trade_date")
    # backtrader 需要 datetime 索引 + 标准列名
    return bt.feeds.PandasData(
        dataname=df,
        name=name,
        datetime=None,  # 用 index
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        openinterest=-1,
    )


@dataclass
class BacktestResult:
    """回测结果。"""

    initial_capital: float
    final_value: float
    returns: pd.Series
    daily_data: pd.DataFrame
    metrics: Dict[str, Any] = field(default_factory=dict)
    cerebro: Optional[bt.Cerebro] = None

    @property
    def total_return(self) -> float:
        return float(self.metrics.get("total_return", 0.0))

    @property
    def annual_return(self) -> float:
        return float(self.metrics.get("cagr", 0.0))

    @property
    def max_drawdown(self) -> float:
        return float(self.metrics.get("max_drawdown", 0.0))

    @property
    def sharpe_ratio(self) -> float:
        return float(self.metrics.get("sharpe", 0.0))

    def print_summary(self) -> None:
        logger.info("=" * 50)
        logger.info(f"初始资金: {self.initial_capital:,.2f}")
        logger.info(f"最终市值: {self.final_value:,.2f}")
        for k, v in self.metrics.items():
            if isinstance(v, (int, float)):
                logger.info(f"  {k}: {v:.4f}")
            else:
                logger.info(f"  {k}: {v}")
        logger.info("=" * 50)


def run_backtest(
    strategy_cls: Type[bt.Strategy],
    data: pd.DataFrame,
    initial_capital: float = 1_000_000,
    strategy_params: Optional[Dict[str, Any]] = None,
    trading_config: Optional[TradingConfig] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> BacktestResult:
    """运行回测。

    Args:
        strategy_cls: backtrader.Strategy 子类
        data: 标准 OHLCV DataFrame，必须含 code / trade_date / open / high / low / close / volume
        initial_capital: 初始资金
        strategy_params: 策略参数字典
        trading_config: 手续费/印花税配置
        start_date / end_date: 可选时间窗
    """
    cfg = trading_config or TradingConfig()
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(initial_capital)
    # A 股佣金（双边）+ 印花税（卖出）近似：用 backtrader 的 commission 字段近似双边费率
    cerebro.broker.setcommission(commission=cfg.commission_rate + cfg.stamp_duty / 2)
    cerebro.broker.set_slippage_perc(cfg.slippage)

    # 注入数据
    if start_date is not None or end_date is not None:
        data = data.copy()
        data["trade_date"] = pd.to_datetime(data["trade_date"])
        if start_date is not None:
            data = data[data["trade_date"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            data = data[data["trade_date"] <= pd.Timestamp(end_date)]

    codes: Iterable[str] = data["code"].unique()
    for code in codes:
        sub = data[data["code"] == code]
        if sub.empty:
            continue
        cerebro.adddata(_df_to_feed(sub, name=code))

    # 注入策略
    cerebro.addstrategy(strategy_cls, **(strategy_params or {}))

    # 分析器
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn", timeframe=bt.TimeFrame.Days)

    results = cerebro.run()
    strat = results[0]

    daily_returns_dict: Dict[Any, float] = strat.analyzers.timereturn.get_analysis()
    if daily_returns_dict:
        returns = pd.Series(daily_returns_dict).sort_index()
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

    m: Dict[str, Any] = {
        "total_return": (final_value - initial_capital) / initial_capital,
    }
    if len(returns) > 5:
        try:
            m.update(compute_metrics(returns))
        except Exception as e:  # quantstats 在极短序列上可能失败
            logger.warning(f"绩效指标计算失败: {e}")

    return BacktestResult(
        initial_capital=initial_capital,
        final_value=final_value,
        returns=returns,
        daily_data=daily_data,
        metrics=m,
        cerebro=cerebro,
    )
