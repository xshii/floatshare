"""开关 tab — 移动友好的功能开关面板, 显示规则说明 + 联动约束。"""

from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
from dash import dcc, html


def layout_features() -> Any:
    # 触发集中 register (cli + web 共享的 flag 列表)
    from floatshare.application import feature_registry  # noqa: F401

    return dbc.Container(
        [
            html.H3("功能开关", className="mt-3 mb-3"),
            dbc.Alert(
                [
                    html.Strong("注意 "),
                    "切换在当前 web 进程内即时生效, 重启丢失。",
                    html.Br(),
                    "永久生效请编辑 ",
                    html.Code(".env"),
                    " 加 ",
                    html.Code("FLOATSHARE_FEATURES=foo,bar"),
                ],
                color="info",
                className="small mb-3",
            ),
            # 约束错误提示条 (依赖未满足/互斥)
            html.Div(id="ff-validation"),
            # 主面板 (按 category 分组卡片)
            html.Div(id="ff-grid"),
            # 自动刷新约束状态
            # 默认 disabled, intervals.py 的 gate callback 会按 active_tab 切换
            dcc.Interval(id="ff-tick", interval=2000, disabled=True),
        ],
        fluid=True,
    )
