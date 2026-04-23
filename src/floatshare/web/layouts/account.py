"""账户 tab — 多账户选择 + 入金/出金 modal + 持仓表。"""

from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
from dash import dcc, html

from floatshare.web.components import account_options


def layout_account() -> Any:
    opts = account_options()
    default = opts[0]["value"] if opts else None
    return dbc.Container(
        [
            html.H3("账户总览", className="mt-3 mb-3"),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            id="acct-dropdown", options=opts, value=default, placeholder="选择账户"
                        ),
                        width=4,
                    ),
                    dbc.Col(
                        dbc.Button("新建账户", id="acct-new-btn", color="secondary", size="sm"),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button("入金", id="acct-deposit-btn", color="primary", size="sm"),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button("出金", id="acct-withdraw-btn", color="danger", size="sm"),
                        width="auto",
                    ),
                ],
                className="mb-3 g-2",
            ),
            html.Div(id="acct-summary"),
            html.H5("持仓", className="mt-4"),
            html.Div(id="acct-holdings"),
            # 新建账户 modal
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("新建账户")),
                    dbc.ModalBody(
                        [
                            dbc.Input(
                                id="modal-acct-id", placeholder="账户 ID (英文)", className="mb-2"
                            ),
                            dbc.Input(
                                id="modal-acct-name", placeholder="账户名称", className="mb-2"
                            ),
                            dbc.Input(
                                id="modal-acct-memo", placeholder="备注 (可选)", className="mb-2"
                            ),
                            html.Div(id="modal-acct-feedback"),
                        ]
                    ),
                    dbc.ModalFooter(dbc.Button("创建", id="modal-acct-save", color="primary")),
                ],
                id="modal-new-acct",
                is_open=False,
            ),
            # 出入金 modal
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("入金 / 出金")),
                    dbc.ModalBody(
                        [
                            dcc.RadioItems(
                                id="modal-money-direction",
                                options=[
                                    {"label": "  入金", "value": "deposit"},
                                    {"label": "  出金", "value": "withdraw"},
                                ],
                                value="deposit",
                                inline=True,
                                className="mb-2",
                            ),
                            dbc.Input(
                                id="modal-money-amount",
                                type="number",
                                placeholder="金额 (元)",
                                className="mb-2",
                            ),
                            dbc.Input(
                                id="modal-money-memo", placeholder="备注 (可选)", className="mb-2"
                            ),
                            html.Div(id="modal-money-feedback"),
                        ]
                    ),
                    dbc.ModalFooter(dbc.Button("确认", id="modal-money-save", color="primary")),
                ],
                id="modal-money",
                is_open=False,
            ),
        ],
        fluid=True,
    )
