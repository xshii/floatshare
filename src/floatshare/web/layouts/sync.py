"""同步/健康 tab — 进度卡 + 手动拉取面板 + 日志 + 表行数 + 日热图。"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import dash_bootstrap_components as dbc
from dash import dcc, html

from floatshare.domain.enums import DataKind


def layout_sync() -> Any:
    today = date.today()
    yesterday = today - timedelta(days=1)
    type_opts = [{"label": k.value, "value": k.value} for k in DataKind.all()]
    return dbc.Container(
        [
            html.H3("同步 / 健康监控", className="mt-3 mb-3"),
            # 默认 disabled, intervals.py 的 gate callback 会按 active_tab 切换
            dcc.Interval(id="sync-tick", interval=5000, disabled=True),
            html.Div(id="sync-progress-card", className="mb-3", style={"minHeight": "180px"}),
            # === 日数据拉取热图 (顶部关键状态) ====================================
            html.Div(
                [
                    html.H5("近一周日数据拉取情况", className="d-inline-block me-2 mb-0"),
                    dbc.Button(
                        "⟳ 刷新",
                        id="daily-status-refresh",
                        size="sm",
                        color="secondary",
                        outline=True,
                    ),
                    html.Small(
                        " (单元格=当日有数据的 code 数, 满量约 5500)", className="text-muted ms-2"
                    ),
                ],
                className="mb-2 d-flex align-items-center",
            ),
            dcc.Loading(html.Div(id="daily-status-grid", style={"minHeight": "220px"})),
            html.Hr(className="my-3"),
            # === 拉取数据面板 ====================================================
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("📥 数据拉取", className="mb-3"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Small("起"),
                                        dcc.DatePickerSingle(
                                            id="pull-start",
                                            date=yesterday.isoformat(),
                                            display_format="YYYY-MM-DD",
                                        ),
                                    ],
                                    width="auto",
                                ),
                                dbc.Col(
                                    [
                                        html.Small("止"),
                                        dcc.DatePickerSingle(
                                            id="pull-end",
                                            date=today.isoformat(),
                                            display_format="YYYY-MM-DD",
                                        ),
                                    ],
                                    width="auto",
                                ),
                                dbc.Col(
                                    [
                                        html.Small("数据类型 (留空=全部)"),
                                        dcc.Dropdown(
                                            id="pull-types",
                                            options=type_opts,
                                            multi=True,
                                            placeholder="全部",
                                        ),
                                    ],
                                    width=5,
                                ),
                                dbc.Col(
                                    dbc.Button("📥 拉取", id="pull-trigger", color="primary"),
                                    width="auto",
                                    className="d-flex align-items-end",
                                ),
                            ],
                            className="g-2",
                        ),
                        html.Div(id="pull-status", className="text-muted small mt-2"),
                    ]
                ),
                className="mb-3",
            ),
            # === 主面板：日志 + 表行数 + 日热图 ==================================
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H5("Sync 日志 (tail -50)"),
                            html.Pre(
                                id="sync-log",
                                style={
                                    # 用 height 而非 maxHeight: 高度恒定, 避免 5s tick 追加日志时
                                    # 整页 reflow 把下面元素挤来挤去
                                    "height": "400px",
                                    "overflowY": "auto",
                                    "background": "#f6f8fa",
                                    "padding": "1em",
                                    "fontSize": "12px",
                                },
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H5("表行数", className="d-inline-block me-2 mb-0"),
                                    dbc.Button(
                                        "⟳ 刷新",
                                        id="counts-refresh",
                                        size="sm",
                                        color="secondary",
                                        outline=True,
                                    ),
                                    html.Small(id="counts-updated-at", className="text-muted ms-2"),
                                ],
                                className="mb-2 d-flex align-items-center",
                            ),
                            dcc.Loading(html.Div(id="sync-counts")),
                        ],
                        width=6,
                    ),
                ],
                className="mb-3",
            ),
        ],
        fluid=True,
    )
