"""策略 tab callbacks — 跑回测 + 渲染 K 线/equity/指标。"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback
from plotly.subplots import make_subplots

from floatshare import notify
from floatshare import registry as strategy_registry
from floatshare.application.backtest import BacktestResult, run_backtest
from floatshare.domain.trading import TradingConfig
from floatshare.web.components import empty_fig, feedback, metric_card, safe_float
from floatshare.web.data import load_klines


@callback(
    Output("strat-fees-collapse", "is_open"),
    Input("strat-fees-toggle", "n_clicks"),
    State("strat-fees-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_fees_panel(_n: int | None, is_open: bool) -> bool:
    return not is_open


def _coalesce(v: float | None, default: float) -> float:
    """空/None 回落到默认值，简化 UI 字段处理。"""
    return float(v) if v is not None else default


@callback(
    Output("strat-chart", "figure"),
    Output("strat-equity", "figure"),
    Output("strat-metrics", "children"),
    Output("strat-live-hint", "children"),
    Input("strat-run", "n_clicks"),
    State("strat-select", "value"),
    State("strat-code", "value"),
    State("strat-dates", "start_date"),
    State("strat-dates", "end_date"),
    State("strat-mode", "value"),
    State("tc-commission", "value"),
    State("tc-stamp", "value"),
    State("tc-transfer", "value"),
    State("tc-min-comm", "value"),
    State("tc-slippage", "value"),
    prevent_initial_call=True,
)
def run_strategy(
    _n: int | None,
    strategy_name: str | None,
    code: str | None,
    start: str | None,
    end: str | None,
    mode: str,
    tc_commission: float | None,
    tc_stamp: float | None,
    tc_transfer: float | None,
    tc_min_comm: float | None,
    tc_slippage: float | None,
) -> tuple:
    if not all([strategy_name, code, start, end]):
        return empty_fig("请完整填写"), empty_fig(""), "", ""

    strategy_cls = strategy_registry.get(strategy_name) if strategy_name else None
    if strategy_cls is None:
        return empty_fig(f"找不到策略 {strategy_name}"), empty_fig(""), "", ""

    start_d = date.fromisoformat(start)  # type: ignore[arg-type]
    end_d = date.today() if mode == "live" else date.fromisoformat(end)  # type: ignore[arg-type]
    klines = load_klines(code, start_d, end_d) if code else pd.DataFrame()
    if klines.empty:
        return empty_fig(f"{code} 无本地 K 线数据"), empty_fig(""), "", ""

    # 构造 TradingConfig — UI 留空时回落到 dataclass 默认值
    defaults = TradingConfig()
    cfg = TradingConfig(
        commission_rate=_coalesce(tc_commission, defaults.commission_rate),
        stamp_duty=_coalesce(tc_stamp, defaults.stamp_duty),
        transfer_fee=_coalesce(tc_transfer, defaults.transfer_fee),
        min_commission=_coalesce(tc_min_comm, defaults.min_commission),
        slippage=_coalesce(tc_slippage, defaults.slippage),
    )
    try:
        result = run_backtest(strategy_cls, klines.assign(code=code), trading_config=cfg)
    except Exception as exc:
        return empty_fig(f"回测失败: {exc}"), empty_fig(""), "", ""

    live_hint: Any = ""
    if mode == "live" and not result.trades.empty:
        latest = result.trades.iloc[-1]
        signal_msg = (
            f"最新信号: {latest['action'].upper()} {latest['size']:.0f}股 "
            f"@ {latest['price']:.2f} 于 {latest['date'].date()}"
        )
        live_hint = feedback(signal_msg, color="warning")
        # 信号在最近 7 天内才推送 (避免点开历史回测刷屏)
        if (date.today() - latest["date"].date()).days <= 7:
            notify(
                f"📈 {strategy_name} / {code} {latest['action'].upper()}",
                f"{signal_msg}\n@ {datetime.now().strftime('%H:%M:%S')} 触发",
            )

    return (
        _build_candlestick(klines, result.trades, str(code)),
        _build_equity_curve(result),
        _build_metrics_card(result),
        live_hint,
    )


def _build_candlestick(
    klines: pd.DataFrame,
    trades: pd.DataFrame,
    code: str,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
        subplot_titles=(f"{code} K 线", "成交量"),
    )
    fig.add_trace(
        go.Candlestick(
            x=klines["trade_date"],
            open=klines["open"],
            high=klines["high"],
            low=klines["low"],
            close=klines["close"],
            name="K 线",
            increasing_line_color="#e74c3c",
            decreasing_line_color="#27ae60",
        ),
        row=1,
        col=1,
    )
    for action, symbol, color, label in (
        ("buy", "triangle-up", "#e74c3c", "买入"),
        ("sell", "triangle-down", "#27ae60", "卖出"),
    ):
        pts = trades[trades["action"] == action] if not trades.empty else pd.DataFrame()
        if not pts.empty:
            fig.add_trace(
                go.Scatter(
                    x=pts["date"],
                    y=pts["price"],
                    mode="markers",
                    name=label,
                    marker={
                        "symbol": symbol,
                        "size": 14,
                        "color": color,
                        "line": {"color": "white", "width": 1},
                    },
                    hovertemplate=f"{label} %{{x|%Y-%m-%d}}<br>价 %{{y:.2f}}<br>量 %{{text}}股",
                    text=pts["size"].astype(str),
                ),
                row=1,
                col=1,
            )
    fig.add_trace(
        go.Bar(x=klines["trade_date"], y=klines["volume"], name="成交量", marker_color="#95a5a6"),
        row=2,
        col=1,
    )
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        showlegend=True,
        hovermode="x unified",
        margin={"l": 40, "r": 40, "t": 40, "b": 40},
    )
    return fig


def _build_equity_curve(result: BacktestResult) -> go.Figure:
    fig = go.Figure()
    if not result.daily_data.empty:
        fig.add_trace(
            go.Scatter(
                x=result.daily_data["date"],
                y=result.daily_data["cum_return"],
                mode="lines",
                name="累计收益",
                line={"color": "#3498db", "width": 2},
                fill="tozeroy",
                fillcolor="rgba(52,152,219,0.15)",
            )
        )
        fig.add_hline(y=1.0, line_dash="dot", line_color="#7f8c8d", annotation_text="本金")
    fig.update_layout(
        title="累计收益曲线",
        yaxis_title="1 + 累计收益",
        margin={"l": 40, "r": 40, "t": 40, "b": 40},
    )
    return fig


def _build_metrics_card(result: BacktestResult) -> Any:
    m = result.metrics
    pnl = result.final_value - result.initial_capital
    pnl_pct = pnl / result.initial_capital if result.initial_capital else 0
    return dbc.Row(
        [
            metric_card("初始资金", result.initial_capital),
            metric_card("最终市值", result.final_value, tone="primary"),
            metric_card("总盈亏", pnl, tone="sign"),
            metric_card("总收益率 %", pnl_pct * 100, tone="sign"),
            metric_card("年化 %", safe_float(m.cagr) * 100),
            metric_card("最大回撤 %", safe_float(m.max_drawdown) * 100),
            metric_card("Sharpe", safe_float(m.sharpe)),
        ],
        className="g-2 mb-3",
    )
