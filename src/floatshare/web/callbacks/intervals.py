"""Tab-aware Interval gating — 只让活跃 tab 的 Interval 跑。

dcc.Interval 是全局组件, 默认无视 active tab 持续触发回调。后果:
- sync 没在跑时, sync-tick 5s 仍在跑 subprocess(tail) + 读 JSON
- 用户在行情/策略 tab 时, ff-tick 2s 仍在校验所有 flag
- 浏览器 XHR 闪烁、tab 标题转圈、CPU 浪费

策略: active tab != owner tab 时把 Interval `disabled` 置 True。
切回该 tab 立即恢复 (Dash Interval disabled 切回 False 会立刻 fire 一次)。
"""

from __future__ import annotations

from dash import Input, Output, callback


@callback(
    Output("sync-tick", "disabled"),
    Output("ff-tick", "disabled"),
    Input("main-tabs", "active_tab"),
)
def gate_intervals(active_tab: str | None) -> tuple[bool, bool]:
    """只在对应 tab 激活时启用 tick。"""
    return (active_tab != "tab-sync", active_tab != "tab-features")
