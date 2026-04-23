"""账户 tab callbacks — 切账户/开户/出入金。"""

from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, callback

from floatshare.web.components import account_options, feedback, metric_card
from floatshare.web.data import BOOK, account_summary, holdings_with_prices


def _render_summary(s: dict) -> Any:
    return dbc.Row(
        [
            metric_card("现金余额", s["cash"]),
            metric_card("持仓市值", s["market_value"]),
            metric_card("总资产", s["total_asset"], tone="primary"),
            metric_card("总本金", s["invested"]),
            metric_card("浮动盈亏", s["floating_pnl"], tone="sign"),
            metric_card("累计盈亏", s["total_pnl"], tone="sign"),
        ],
        className="g-2",
    )


def _render_holdings(df: pd.DataFrame) -> Any:
    if df.empty:
        return dbc.Alert("尚无持仓", color="secondary")
    shown = df.copy()
    for c in ("shares", "avg_cost", "last_price", "market_value", "total_cost", "pnl"):
        shown[c] = df[c].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "-")
    shown["pnl_pct"] = df["pnl_pct"].map(
        lambda x: f"{x * 100:+.2f}%" if pd.notna(x) else "-",
    )
    shown = shown[
        ["code", "shares", "avg_cost", "last_price", "market_value", "total_cost", "pnl", "pnl_pct"]
    ]
    shown.columns = ["代码", "份额", "成本价", "最新价", "市值", "累计成本", "浮盈", "浮盈 %"]
    return dbc.Table.from_dataframe(shown, striped=True, bordered=True, hover=True, size="sm")


@callback(
    Output("acct-summary", "children"),
    Output("acct-holdings", "children"),
    Input("acct-dropdown", "value"),
    Input("acct-refresh", "data"),
)
def update_account(account_id: str | None, _refresh: int) -> tuple:
    if not account_id:
        return dbc.Alert("请先在右上角创建账户", color="info"), ""
    return _render_summary(account_summary(account_id)), _render_holdings(
        holdings_with_prices(account_id)
    )


@callback(
    Output("modal-new-acct", "is_open", allow_duplicate=True),
    Input("acct-new-btn", "n_clicks"),
    prevent_initial_call=True,
)
def open_new_acct(_n: int | None) -> bool:
    return True


@callback(
    Output("modal-money", "is_open", allow_duplicate=True),
    Input("acct-deposit-btn", "n_clicks"),
    Input("acct-withdraw-btn", "n_clicks"),
    prevent_initial_call=True,
)
def open_money_modal(_n_dep: int | None, _n_wd: int | None) -> bool:
    return True


@callback(
    Output("acct-refresh", "data", allow_duplicate=True),
    Output("acct-dropdown", "options", allow_duplicate=True),
    Output("modal-new-acct", "is_open", allow_duplicate=True),
    Output("modal-acct-feedback", "children"),
    Input("modal-acct-save", "n_clicks"),
    State("modal-acct-id", "value"),
    State("modal-acct-name", "value"),
    State("modal-acct-memo", "value"),
    State("acct-refresh", "data"),
    prevent_initial_call=True,
)
def save_new_account(
    _n: int | None,
    acct_id: str | None,
    name: str | None,
    memo: str | None,
    refresh: int,
) -> tuple:
    if not acct_id or not name:
        return refresh, account_options(), True, feedback("账户 ID 和名称都必填", "warning")
    try:
        BOOK.open_account(acct_id.strip(), name.strip(), memo=(memo or None))
    except Exception as exc:
        return refresh, account_options(), True, feedback(f"创建失败: {exc}")
    return refresh + 1, account_options(), False, ""


@callback(
    Output("acct-refresh", "data", allow_duplicate=True),
    Output("modal-money", "is_open", allow_duplicate=True),
    Output("modal-money-feedback", "children"),
    Input("modal-money-save", "n_clicks"),
    State("acct-dropdown", "value"),
    State("modal-money-direction", "value"),
    State("modal-money-amount", "value"),
    State("modal-money-memo", "value"),
    State("acct-refresh", "data"),
    prevent_initial_call=True,
)
def save_money(
    _n: int | None,
    account_id: str | None,
    direction: str,
    amount: float | None,
    memo: str | None,
    refresh: int,
) -> tuple:
    if not account_id:
        return refresh, True, feedback("请先选择账户", "warning")
    if not amount or amount <= 0:
        return refresh, True, feedback("金额必须 > 0", "warning")
    try:
        action = BOOK.deposit if direction == "deposit" else BOOK.withdraw
        action(account_id, amount, memo=memo or None)
    except Exception as exc:
        label = "入金" if direction == "deposit" else "出金"
        return refresh, True, feedback(f"{label}失败: {exc}")
    return refresh + 1, False, ""
