"""开关 tab callbacks — 读/写 flag, 渲染分组卡片, 校验依赖/互斥。"""

from __future__ import annotations

import os
from typing import Any

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, html

from floatshare.observability import features as ff


@callback(
    Output("ff-grid", "children"),
    Output("ff-validation", "children"),
    Input("ff-tick", "n_intervals"),
    Input({"type": "ff-toggle", "name": dash.ALL}, "value"),
    State({"type": "ff-toggle", "name": dash.ALL}, "id"),
)
def update_features_panel(_n: int | None, values: list, ids: list) -> tuple:
    """渲染 flag 网格 + 应用 toggle 到 os.environ + 显示约束违规。"""
    # 1. Apply toggle changes (in-memory env override)
    for v, i in zip(values or [], ids or [], strict=False):
        name = i["name"]
        env_key = f"FLOATSHARE_FEATURE_{name.upper()}"
        os.environ[env_key] = "1" if v else "0"

    # 2. Render grid by category
    grid = _render_features_grid()

    # 3. Validation status
    errors = ff.validate_enabled() + ff.validate_registry()
    if errors:
        bar = dbc.Alert(
            [html.Strong("⚠️ 当前启用集合违反约束:"), html.Ul([html.Li(e) for e in errors])],
            color="warning",
            className="mb-3",
        )
    else:
        bar = ""
    return grid, bar


def _render_features_grid() -> Any:
    by_cat: dict[str, list] = {}
    for f in ff.all_flags():
        by_cat.setdefault(f.category, []).append(f)

    cards = []
    for cat, flags in sorted(by_cat.items()):
        rows = []
        for f in flags:
            on = ff.is_enabled(f.name)
            badges = []
            if f.requires:
                badges.append(
                    dbc.Badge(f"依赖 {','.join(f.requires)}", color="info", className="me-1 small")
                )
            if f.conflicts:
                badges.append(
                    dbc.Badge(
                        f"互斥 {','.join(f.conflicts)}", color="danger", className="me-1 small"
                    )
                )
            if f.default:
                badges.append(dbc.Badge("默认开", color="success", className="me-1 small"))
            rows.append(
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Switch(
                                id={"type": "ff-toggle", "name": f.name},
                                value=on,
                                label=f.name,
                                label_style={"font-weight": "600", "font-family": "monospace"},
                            ),
                            xs=12,
                            md=4,
                        ),
                        dbc.Col(
                            [
                                html.Div(f.description, className="small"),
                                html.Div(
                                    ["影响: ", html.Span(f.impact, className="text-muted")],
                                    className="small mt-1",
                                )
                                if f.impact
                                else "",
                                html.Div(badges, className="mt-1") if badges else "",
                            ],
                            xs=12,
                            md=8,
                        ),
                    ],
                    className="g-2 py-2 border-bottom",
                )
            )
        cards.append(
            dbc.Card(
                [
                    dbc.CardHeader(html.Strong(f"[{cat}]")),
                    dbc.CardBody(rows),
                ],
                className="mb-3",
            )
        )
    return cards
