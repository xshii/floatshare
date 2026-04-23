"""行情 tab callback — K 线 + 周期重采样 + 多指标叠加。"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback
from plotly.subplots import make_subplots

from floatshare.domain.enums import AdjustType
from floatshare.web.components import empty_fig
from floatshare.web.data import load_klines
from floatshare.web.indicators import (
    IndicatorSpec,
    get,
    resample_klines,
)

_ADJ_LABEL: dict[AdjustType, str] = {
    AdjustType.QFQ: "前复权",
    AdjustType.HFQ: "后复权",
    AdjustType.NONE: "不复权",
}
_PERIOD_LABEL: dict[str, str] = {"D": "日", "W": "周", "M": "月", "Y": "年"}


@callback(
    Output("quotes-chart", "figure"),
    Input("quotes-code", "value"),
    Input("quotes-dates", "start_date"),
    Input("quotes-dates", "end_date"),
    Input("quotes-period", "value"),
    Input("quotes-adj", "value"),
    Input("quotes-indicators", "value"),
    Input("quotes-shift", "value"),
)
def render_quotes(
    code: str | None,
    start: str | None,
    end: str | None,
    period: str | None,
    adj_value: str | None,
    indicators: list[str] | None,
    shift_on: bool | None,
) -> go.Figure:
    if not (code and start and end):
        return empty_fig("请选择股票与日期窗口")
    adj = AdjustType(adj_value or "qfq")
    df = load_klines(code, date.fromisoformat(start[:10]), date.fromisoformat(end[:10]), adj=adj)
    if df.empty:
        return empty_fig(f"{code} 无本地 K 线数据")
    df = resample_klines(df, period or "D")
    return _build_figure(df, code, adj, period or "D", indicators or [], bool(shift_on))


# ==============================================================================
# 渲染
# ==============================================================================


def _compute(spec: IndicatorSpec, df: pd.DataFrame, shift_on: bool) -> pd.DataFrame:
    """跑指标; 全局 shift 开启时给非 forward 指标 shift(1) 变成无前视。"""
    out = spec.compute(df)
    if shift_on and not spec.forward_only:
        out = out.shift(1)
    return out


def _build_figure(
    df: pd.DataFrame,
    code: str,
    adj: AdjustType,
    period: str,
    indicator_names: list[str],
    shift_on: bool,
) -> go.Figure:
    specs = [s for s in (get(n) for n in indicator_names) if s is not None]
    overlays = [s for s in specs if s.panel == "overlay"]
    subplots = [s for s in specs if s.panel == "subplot"]

    rows = 1 + len(subplots)
    main_h = 0.6 if subplots else 1.0
    sub_h = (1 - main_h) / len(subplots) if subplots else 0.0
    row_heights = [main_h, *([sub_h] * len(subplots))]
    shift_tag = " · 🔵 shift(1)" if shift_on else ""
    titles = [
        f"{code} · {_PERIOD_LABEL[period]}线 · {_ADJ_LABEL[adj]}{shift_tag}",
        *(s.label for s in subplots),
    ]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.04,
        subplot_titles=titles,
    )
    fig.add_trace(
        go.Candlestick(
            x=df["trade_date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K 线",
            increasing_line_color="#e74c3c",
            decreasing_line_color="#27ae60",
        ),
        row=1,
        col=1,
    )

    for spec in overlays:
        _add_overlay(fig, df, spec, shift_on)
    for i, spec in enumerate(subplots, start=2):
        _add_subplot(fig, df, spec, row=i, shift_on=shift_on)

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        showlegend=True,
        hovermode="x unified",
        margin={"l": 40, "r": 40, "t": 40, "b": 40},
    )
    return fig


def _add_overlay(fig: go.Figure, df: pd.DataFrame, spec: IndicatorSpec, shift_on: bool) -> None:
    """主图叠加 — 每一列加一条线。"""
    out = _compute(spec, df, shift_on)
    for col in out.columns:
        fig.add_trace(
            go.Scatter(
                x=df["trade_date"],
                y=out[col],
                mode="lines",
                name=col,
                line={"width": 1, "color": spec.color},
                legendgroup=spec.name,
            ),
            row=1,
            col=1,
        )


def _add_subplot(
    fig: go.Figure, df: pd.DataFrame, spec: IndicatorSpec, row: int, shift_on: bool
) -> None:
    """副图独立行 — vol/HIST 用 Bar, 其它 Scatter。"""
    out = _compute(spec, df, shift_on)
    for col in out.columns:
        is_bar = spec.name == "vol" or col == "HIST"
        trace_cls = go.Bar if is_bar else go.Scatter
        kw: dict[str, Any] = {
            "x": df["trade_date"],
            "y": out[col],
            "name": col,
            "legendgroup": spec.name,
        }
        if is_bar:
            kw["marker_color"] = spec.color
        else:
            kw["mode"] = "lines"
            kw["line"] = {"width": 1}
        fig.add_trace(trace_cls(**kw), row=row, col=1)
