"""因子库。

注意：此处不 eager-import 子模块以避免拉起 `ta` 库的副作用；
外部用户按需 `from floatshare.strategy.factors.technical import MAFactor`。
"""

from floatshare.strategy.factors.protocols import Factor

__all__ = ["Factor"]
