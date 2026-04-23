"""因子契约 — 用 Protocol 替代 ABC，鸭子类型。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class Factor(Protocol):
    """因子的最小契约：对 OHLCV DataFrame 算出长度对齐的 Series。"""

    name: str

    def calculate(self, data: pd.DataFrame) -> pd.Series: ...
