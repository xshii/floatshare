"""技术指标库 — 行情 tab 用。

每个指标是一个 IndicatorSpec(纯函数 + 元数据), 注册到模块级 _REGISTRY。
新增指标只要 register(IndicatorSpec(...)), 无需改 callbacks/layouts。

约定:
    forward_only=True : 数值仅依赖 trade_date < D 的数据 (无前视, 可作 D 日交易信号)
    forward_only=False: 数值含 trade_date == D 当日 close (典型 MA/RSI, 不能作当日信号)

UI 标记 (display_label):
    🔵 forward-only         — 严格因果, 无前视偏差
    ⚠️ contemporaneous      — 含当日数据, 不能作当日交易信号
    📊 (suffix "副图")       — 副图独立展示
    (无 suffix)              — 主图叠加
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pandas as pd

Panel = Literal["overlay", "subplot"]


@dataclass(frozen=True, slots=True)
class IndicatorSpec:
    name: str
    label: str
    panel: Panel
    forward_only: bool
    compute: Callable[[pd.DataFrame], pd.DataFrame]
    color: str = "#888888"  # subplot 主线 / overlay 单线默认色


_REGISTRY: dict[str, IndicatorSpec] = {}


def register(spec: IndicatorSpec) -> None:
    _REGISTRY[spec.name] = spec


def get(name: str) -> IndicatorSpec | None:
    return _REGISTRY.get(name)


def all_indicators() -> list[IndicatorSpec]:
    return list(_REGISTRY.values())


def display_label(spec: IndicatorSpec) -> str:
    """UI dropdown 显示: emoji 标因果性 + (副图) 后缀。"""
    prefix = "🔵" if spec.forward_only else "⚠️"
    suffix = " 📊副图" if spec.panel == "subplot" else ""
    return f"{prefix} {spec.label}{suffix}"


# ==============================================================================
# 计算函数 (无副作用, 输入 OHLCV df, 输出 indicator-only df)
# ==============================================================================


# Pandas 类型推断在 rolling/ewm 链上会出 Series|DataFrame 模糊,
# 这些 helper 实际返回都是 Series, 用 cast 标注以避免 Pyright 噪声。


def _close(df: pd.DataFrame) -> pd.Series:
    return cast(pd.Series, df["close"])


def _ma(close: pd.Series, n: int, shift: int = 0) -> pd.Series:
    rolled = cast(pd.Series, close.rolling(n, min_periods=n).mean())
    return cast(pd.Series, rolled.shift(shift))


def _ema(close: pd.Series, n: int) -> pd.Series:
    return cast(pd.Series, close.ewm(span=n, adjust=False).mean())


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    dif = _ema(close, fast) - _ema(close, slow)
    dea = cast(pd.Series, dif.ewm(span=signal, adjust=False).mean())
    return pd.DataFrame({"DIF": dif, "DEA": dea, "HIST": (dif - dea) * 2})


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return cast(pd.Series, 100 - 100 / (1 + rs))


def _kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
    low_min = cast(pd.Series, df["low"]).rolling(n).min()
    high_max = cast(pd.Series, df["high"]).rolling(n).max()
    rsv = (_close(df) - low_min) / (high_max - low_min) * 100
    k = cast(pd.Series, rsv.ewm(alpha=1 / m1, adjust=False).mean())
    d = cast(pd.Series, k.ewm(alpha=1 / m2, adjust=False).mean())
    return pd.DataFrame({"K": k, "D": d, "J": 3 * k - 2 * d})


def _boll(close: pd.Series, n: int = 20, k: int = 2) -> pd.DataFrame:
    mid = cast(pd.Series, close.rolling(n).mean())
    std = cast(pd.Series, close.rolling(n).std(ddof=0))
    return pd.DataFrame(
        {
            "BOLL_up": mid + k * std,
            "BOLL_mid": mid,
            "BOLL_dn": mid - k * std,
        }
    )


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = (cast(pd.Series, df["high"]), cast(pd.Series, df["low"]), _close(df))
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(
        axis=1
    )
    return cast(pd.Series, tr.ewm(alpha=1 / n, adjust=False).mean())


# ==============================================================================
# 注册 (8 个标准指标, 含一个 forward-only 示例)
# ==============================================================================

# --- 主图叠加 ---
for _n, _color in [(5, "#ff7f0e"), (20, "#1f77b4"), (60, "#9467bd")]:
    register(
        IndicatorSpec(
            name=f"ma{_n}",
            label=f"MA{_n}",
            panel="overlay",
            forward_only=False,
            compute=lambda df, n=_n: pd.DataFrame({f"MA{n}": _ma(_close(df), n)}),
            color=_color,
        )
    )

register(
    IndicatorSpec(
        name="ma20_prev",
        label="MA20 (前一日)",
        panel="overlay",
        forward_only=True,
        compute=lambda df: pd.DataFrame({"MA20_prev": _ma(_close(df), 20, shift=1)}),
        color="#17becf",
    )
)

register(
    IndicatorSpec(
        name="boll",
        label="BOLL(20,2)",
        panel="overlay",
        forward_only=False,
        compute=lambda df: _boll(_close(df)),
        color="#bcbd22",
    )
)

# --- 副图 ---
register(
    IndicatorSpec(
        name="vol",
        label="成交量",
        panel="subplot",
        forward_only=False,
        compute=lambda df: pd.DataFrame({"vol": df["volume"]}),
        color="#95a5a6",
    )
)
register(
    IndicatorSpec(
        name="macd",
        label="MACD(12,26,9)",
        panel="subplot",
        forward_only=False,
        compute=lambda df: _macd(_close(df)),
    )
)
register(
    IndicatorSpec(
        name="rsi14",
        label="RSI(14)",
        panel="subplot",
        forward_only=False,
        compute=lambda df: pd.DataFrame({"RSI14": _rsi(_close(df), 14)}),
        color="#e377c2",
    )
)
register(
    IndicatorSpec(
        name="kdj",
        label="KDJ(9,3,3)",
        panel="subplot",
        forward_only=False,
        compute=_kdj,
    )
)
register(
    IndicatorSpec(
        name="atr14",
        label="ATR(14)",
        panel="subplot",
        forward_only=False,
        compute=lambda df: pd.DataFrame({"ATR14": _atr(df, 14)}),
        color="#8c564b",
    )
)


# ==============================================================================
# 周期重采样 (D/W/M/Y)
# ==============================================================================

PERIOD_RULE: dict[str, str | None] = {
    "D": None,
    "W": "W-FRI",
    "M": "ME",
    "Y": "YE",
}


def resample_klines(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """daily OHLCV → 周/月/年。空 rule → 原样返回。"""
    rule = PERIOD_RULE.get(period)
    if rule is None or df.empty:
        return df
    work = df.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    out = cast(
        pd.DataFrame,
        work.set_index("trade_date")
        .resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}),
    )
    return out.dropna(subset=["open"]).reset_index()
