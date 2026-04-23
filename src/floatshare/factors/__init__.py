"""因子库 — 给策略开发者使用的开箱即用因子集合。

注意：此处不 eager-import 子模块以避免拉起 `ta` 库的副作用；
按需 `from floatshare.factors.technical import MAFactor`。

Factor 协议从 interfaces 层 re-export 以方便用户类型注解。
"""

from floatshare.interfaces.factor import Factor as Factor

__all__ = ["Factor"]
