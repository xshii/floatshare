"""绩效报告 — 全部走 quantstats，输出强类型 Metrics。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import quantstats as qs

qs.extend_pandas()


@dataclass(frozen=True, slots=True)
class Metrics:
    """统一的绩效指标快照。"""

    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    profit_factor: float
    var_95: float
    cvar_95: float
    alpha: float | None = None
    beta: float | None = None
    information_ratio: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


def _safe(value: object) -> float:
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return f


def metrics(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    rf: float = 0.0,
) -> Metrics:
    """从日收益率序列计算 Metrics。极短或全零序列返回 NaN，不抛异常。"""
    alpha: float | None = None
    beta: float | None = None
    info_ratio: float | None = None
    if benchmark is not None and len(returns) > 1:
        try:
            greeks = qs.stats.greeks(returns, benchmark)
            alpha = _safe(greeks.get("alpha", float("nan")))
            beta = _safe(greeks.get("beta", float("nan")))
            info_ratio = _safe(qs.stats.information_ratio(returns, benchmark))
        except Exception:
            pass

    return Metrics(
        total_return=_safe(qs.stats.comp(returns)),
        cagr=_safe(qs.stats.cagr(returns, rf=rf)),
        volatility=_safe(qs.stats.volatility(returns)),
        sharpe=_safe(qs.stats.sharpe(returns, rf=rf)),
        sortino=_safe(qs.stats.sortino(returns, rf=rf)),
        max_drawdown=_safe(qs.stats.max_drawdown(returns)),
        calmar=_safe(qs.stats.calmar(returns)),
        win_rate=_safe(qs.stats.win_rate(returns)),
        profit_factor=_safe(qs.stats.profit_factor(returns)),
        var_95=_safe(qs.stats.value_at_risk(returns)),
        cvar_95=_safe(qs.stats.cvar(returns)),
        alpha=alpha,
        beta=beta,
        information_ratio=info_ratio,
    )


def html_report(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    output: str | Path = "report.html",
    title: str = "FloatShare Backtest Report",
) -> Path:
    """生成完整 HTML tearsheet。"""
    out = Path(output)
    qs.reports.html(returns, benchmark=benchmark, output=str(out), title=title)
    return out
