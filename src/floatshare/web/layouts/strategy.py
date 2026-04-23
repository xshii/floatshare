"""策略 tab — 选策略+股票+窗口, K 线 + equity + 指标。"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import dash_bootstrap_components as dbc
from dash import dcc, html

from floatshare import registry as strategy_registry
from floatshare.domain.trading import TradingConfig
from floatshare.web.data import listed_codes

_DEFAULT_TC = TradingConfig()


def _fee_input(label: str, input_id: str, value: float, step: float, unit: str = "") -> Any:
    """费率/金额输入: label + 单位提示 + number input。"""
    return dbc.Col(
        [
            html.Small(f"{label} {unit}".strip(), className="text-muted"),
            dcc.Input(
                id=input_id,
                type="number",
                value=value,
                step=step,
                min=0,
                debounce=True,
                className="form-control form-control-sm",
            ),
        ],
        xs=6,
        sm=4,
        md=2,
    )


def layout_strategy() -> Any:
    strat_opts = [{"label": s, "value": s} for s in strategy_registry.list_strategies()]
    code_opts = [{"label": lbl, "value": c} for c, lbl in listed_codes()]
    today = date.today()
    default_start = today - timedelta(days=365 * 2)
    return dbc.Container(
        [
            html.H3("策略回测 / 模拟 / 实盘", className="mt-3 mb-3"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("策略"),
                            dcc.Dropdown(
                                id="strat-select",
                                options=strat_opts,
                                value=strat_opts[0]["value"] if strat_opts else None,
                            ),
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            html.Label("股票"),
                            dcc.Dropdown(
                                id="strat-code",
                                options=code_opts,
                                value=code_opts[0]["value"] if code_opts else None,
                            ),
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            html.Label("窗口"),
                            dcc.DatePickerRange(
                                id="strat-dates",
                                start_date=default_start.isoformat(),
                                end_date=today.isoformat(),
                                display_format="YYYY-MM-DD",
                            ),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            html.Label("模式"),
                            dcc.RadioItems(
                                id="strat-mode",
                                options=[
                                    {"label": "  模拟", "value": "sim"},
                                    {"label": "  实盘信号", "value": "live"},
                                ],
                                value="sim",
                                inline=True,
                            ),
                        ],
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button("运行", id="strat-run", color="primary"),
                        width="auto",
                        className="d-flex align-items-end",
                    ),
                    dbc.Col(
                        dbc.Button("⚙ 费率/滑点", id="strat-fees-toggle", color="link", size="sm"),
                        width="auto",
                        className="d-flex align-items-end",
                    ),
                ],
                className="g-2 mb-3",
            ),
            # === 费率/滑点配置 (折叠, 默认 A 股标准) ============================
            dbc.Collapse(
                dbc.Card(
                    dbc.CardBody(
                        dbc.Row(
                            [
                                _fee_input(
                                    "佣金率",
                                    "tc-commission",
                                    _DEFAULT_TC.commission_rate,
                                    0.00001,
                                    "(单边, 默认万 3)",
                                ),
                                _fee_input(
                                    "印花税",
                                    "tc-stamp",
                                    _DEFAULT_TC.stamp_duty,
                                    0.00001,
                                    "(卖出, 默认万 5)",
                                ),
                                _fee_input(
                                    "过户费",
                                    "tc-transfer",
                                    _DEFAULT_TC.transfer_fee,
                                    0.000001,
                                    "(双边, 默认万 0.1)",
                                ),
                                _fee_input(
                                    "最低佣金",
                                    "tc-min-comm",
                                    _DEFAULT_TC.min_commission,
                                    0.5,
                                    "(元)",
                                ),
                                _fee_input(
                                    "滑点",
                                    "tc-slippage",
                                    _DEFAULT_TC.slippage,
                                    0.0001,
                                    "(默认千 1)",
                                ),
                            ],
                            className="g-2",
                        )
                    )
                ),
                id="strat-fees-collapse",
                is_open=False,
                className="mb-3",
            ),
            html.Div(id="strat-metrics"),
            dcc.Loading(dcc.Graph(id="strat-chart", style={"height": "600px"})),
            dcc.Loading(dcc.Graph(id="strat-equity", style={"height": "300px"})),
            html.Div(id="strat-live-hint"),
        ],
        fluid=True,
    )
