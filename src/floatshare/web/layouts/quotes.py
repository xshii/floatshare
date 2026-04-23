"""行情 tab — K 线 + 周期切换 + 指标叠加。"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import dash_bootstrap_components as dbc
from dash import dcc, html

from floatshare.web.data import listed_codes
from floatshare.web.indicators import all_indicators, display_label

_DEFAULT_OVERLAYS = ["ma5", "ma20"]
_DEFAULT_SUBPLOT = "vol"


def layout_quotes() -> Any:
    code_opts = [{"label": lbl, "value": c} for c, lbl in listed_codes()]
    ind_opts = [{"label": display_label(s), "value": s.name} for s in all_indicators()]
    today = date.today()
    default_start = today - timedelta(days=365)
    return dbc.Container(
        [
            html.H3("行情 K 线", className="mt-3 mb-3"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("股票"),
                            dcc.Dropdown(
                                id="quotes-code",
                                options=code_opts,
                                value=code_opts[0]["value"] if code_opts else None,
                                searchable=True,
                                clearable=False,
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Label("窗口"),
                            dcc.DatePickerRange(
                                id="quotes-dates",
                                start_date=default_start.isoformat(),
                                end_date=today.isoformat(),
                                display_format="YYYY-MM-DD",
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Label("周期"),
                            dcc.RadioItems(
                                id="quotes-period",
                                options=[
                                    {"label": f"  {x}", "value": x} for x in ("D", "W", "M", "Y")
                                ],
                                value="D",
                                inline=True,
                            ),
                        ],
                        md=2,
                    ),
                    dbc.Col(
                        [
                            html.Label("复权"),
                            dcc.RadioItems(
                                id="quotes-adj",
                                options=[
                                    {"label": "  前复权", "value": "qfq"},
                                    {"label": "  后复权", "value": "hfq"},
                                    {"label": "  不复权", "value": "none"},
                                ],
                                value="qfq",
                                inline=True,
                            ),
                        ],
                        width="auto",
                    ),
                ],
                className="g-2 mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label(
                                [
                                    "指标 ",
                                    html.Small(
                                        "🔵=严格因果(无前视) ⚠️=含当日 📊=副图",
                                        className="text-muted ms-2",
                                    ),
                                ]
                            ),
                            dcc.Dropdown(
                                id="quotes-indicators",
                                options=ind_opts,
                                value=[*_DEFAULT_OVERLAYS, _DEFAULT_SUBPLOT],
                                multi=True,
                                clearable=True,
                            ),
                        ],
                        md=10,
                    ),
                    dbc.Col(
                        [
                            html.Label(" "),  # 与 dropdown 对齐
                            dbc.Switch(
                                id="quotes-shift",
                                label="🔵 全局 shift(1)  (所有 ⚠️ 指标变成无前视, 默认开)",
                                value=True,
                                className="mt-2",
                            ),
                        ],
                        md=2,
                    ),
                ],
                className="g-2 mb-2",
            ),
            dcc.Loading(dcc.Graph(id="quotes-chart", style={"height": "750px"})),
        ],
        fluid=True,
    )
