"""策略实现集合 — 导入即注册。"""

from .dual_thrust import DualThrustStrategy
from .ma_cross import MACrossStrategy

__all__ = ["DualThrustStrategy", "MACrossStrategy"]
