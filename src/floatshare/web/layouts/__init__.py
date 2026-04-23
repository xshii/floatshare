"""Tab 布局 — 每个 tab 一个文件, 纯函数返回 Dash component。

callbacks 通过 Output id 与这里的 component id 对接, 互不直接 import。
"""

from floatshare.web.layouts.account import layout_account as layout_account
from floatshare.web.layouts.features import layout_features as layout_features
from floatshare.web.layouts.quotes import layout_quotes as layout_quotes
from floatshare.web.layouts.strategy import layout_strategy as layout_strategy
from floatshare.web.layouts.sync import layout_sync as layout_sync
