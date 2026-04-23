"""复用 UI 助手 — 跨 layouts/callbacks 共享。

放这里的标准: 不依赖 callback 状态, 纯函数 (输入 → Dash component)。
有副作用 / 跨进程 IO 的助手放对应 callbacks/ 子模块。
"""

from __future__ import annotations

import math
import re
from typing import Any

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import html

from floatshare.web.data import list_accounts

# loguru / colorama / ANSI 转义清理 (sync 日志染过色, 网页里要去掉)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def feedback(msg: str, color: str = "danger") -> Any:
    """统一样式的提示条 (modal 内 / 顶部 alert)。"""
    return dbc.Alert(msg, color=color, className="mt-2")


def metric_card(label: str, value: float, tone: str = "neutral") -> Any:
    """tone: neutral / primary / sign (自动标红绿)。

    响应式宽度: 手机 (xs) 半屏 / 平板 (md) 1/3 / 桌面 (lg) 1/6。
    """
    color = "primary" if tone == "primary" else "light"
    inverse = tone == "primary"
    text_class = ""
    if tone == "sign" and value:
        text_class = "text-success" if value > 0 else "text-danger"
    return dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.Small(label, className="text-muted"),
                    html.H4(f"{value:,.2f}", className=text_class),
                ]
            ),
            color=color,
            inverse=inverse,
        ),
        xs=6,
        md=4,
        lg=2,
        className="mb-2",
    )


def empty_fig(msg: str) -> go.Figure:
    """占位 figure — 提示用户填表/无数据。"""
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font={"size": 16, "color": "#888"})
    fig.update_layout(
        margin={"l": 40, "r": 40, "t": 40, "b": 40},
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def safe_float(v: float) -> float:
    """NaN → 0.0 (quantstats 在样本不足时返 NaN, 防止前端崩)。"""
    return 0.0 if math.isnan(v) else float(v)


def fmt_duration(secs: int | None) -> str:
    """秒数 → 人类友好的 1h 23m / 5m 12s / 30s。"""
    if not secs:
        return "0s"
    h, rem = divmod(int(secs), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def account_options() -> list[dict]:
    """账户下拉框选项 — layout 初始化和创建账户后都要用。"""
    df = list_accounts()
    return [{"label": f"{r.name} ({r.account_id})", "value": r.account_id} for r in df.itertuples()]
