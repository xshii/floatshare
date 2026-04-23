"""FloatShare 可视化看板 — Dash 入口。

Tabs:
    1. 账户总览  — 多账户下拉, 现金+市值+总资产+盈亏 + 持仓
    2. 行情 K 线 — 选股+窗口+复权(qfq/hfq/none), candlestick + 成交量副图
    3. 策略回测  — 选策略+股票+窗口, K 线带买卖点 + equity curve + 指标
    4. 同步监控  — sync 进度卡 + 日志 tail + 表行数 + 日热图 + 手动拉取
    5. 功能开关  — flag 分组卡片, 依赖/互斥校验

实现按 layouts/ + callbacks/ 拆分 — 本文件只组装。

运行:
    floatshare-web                        # 默认 0.0.0.0:8050
    floatshare-web --port 8080 --debug
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import Dash, dcc, html

from floatshare.web.layouts import (
    layout_account,
    layout_features,
    layout_quotes,
    layout_strategy,
    layout_sync,
)

# 把项目根加到 sys.path, 让 strategies/ 被自动发现 + 注册到 registry
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# 注意: `import strategies` 只 import 包, 不会 import 其下子模块,
# `@register` 装饰器不会触发。必须 discover 递归导入所有 .py。
with contextlib.suppress(ImportError):
    from floatshare.registry import discover

    discover("strategies")


app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="FloatShare Dashboard",
    suppress_callback_exceptions=True,
    # 手机适配: viewport meta tag 让响应式断点生效
    meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1, shrink-to-fit=no",
        }
    ],
)

# @callback 注册必须在 Dash() 实例化之后 (callback 绑定到 dash 全局栈顶 app)
from floatshare.web import callbacks as _callbacks  # noqa: E402, F401

app.layout = dbc.Container(
    [
        html.H1("FloatShare 量化看板", className="mt-3 mb-3"),
        dbc.Tabs(
            [
                dbc.Tab(layout_account(), label="账户", tab_id="tab-account"),
                dbc.Tab(layout_quotes(), label="行情", tab_id="tab-quotes"),
                dbc.Tab(layout_strategy(), label="策略", tab_id="tab-strat"),
                dbc.Tab(layout_sync(), label="同步/健康", tab_id="tab-sync"),
                dbc.Tab(layout_features(), label="开关", tab_id="tab-features"),
            ],
            id="main-tabs",
            active_tab="tab-account",
        ),
        dcc.Store(id="acct-refresh", data=0),
    ],
    fluid=True,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="FloatShare 可视化看板")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
