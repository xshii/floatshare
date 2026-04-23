"""Dash callbacks — 按 tab 拆分。

import 即注册 (Dash @callback 是全局 registry), 上层只需 `import callbacks`。
"""

from floatshare.web.callbacks import account as account
from floatshare.web.callbacks import features as features
from floatshare.web.callbacks import intervals as intervals
from floatshare.web.callbacks import quotes as quotes
from floatshare.web.callbacks import strategy as strategy
from floatshare.web.callbacks import sync as sync
