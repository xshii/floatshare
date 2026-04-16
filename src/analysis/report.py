"""绩效报告 — 全部走 quantstats，不再自己实现指标。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd
import quantstats as qs

# 注入 pandas 扩展（qs.stats / qs.plots）
qs.extend_pandas()


def metrics(
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    rf: float = 0.0,
) -> dict:
    """常用绩效指标一次性算出。

    Args:
        returns: 日收益率序列（index 为日期）
        benchmark: 基准日收益率序列，可选
        rf: 年化无风险利率

    Returns:
        指标字典
    """
    out = {
        "total_return": float(qs.stats.comp(returns)),
        "cagr": float(qs.stats.cagr(returns, rf=rf)),
        "volatility": float(qs.stats.volatility(returns)),
        "sharpe": float(qs.stats.sharpe(returns, rf=rf)),
        "sortino": float(qs.stats.sortino(returns, rf=rf)),
        "max_drawdown": float(qs.stats.max_drawdown(returns)),
        "calmar": float(qs.stats.calmar(returns)),
        "win_rate": float(qs.stats.win_rate(returns)),
        "profit_factor": float(qs.stats.profit_factor(returns)),
        "var_95": float(qs.stats.value_at_risk(returns)),
        "cvar_95": float(qs.stats.cvar(returns)),
    }
    if benchmark is not None:
        out["alpha_beta"] = tuple(map(float, qs.stats.greeks(returns, benchmark)))
        out["information_ratio"] = float(qs.stats.information_ratio(returns, benchmark))
    return out


def html_report(
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    output: Union[str, Path] = "report.html",
    title: str = "FloatShare Backtest Report",
) -> Path:
    """生成完整 HTML tearsheet。"""
    output = Path(output)
    qs.reports.html(returns, benchmark=benchmark, output=str(output), title=title)
    return output
