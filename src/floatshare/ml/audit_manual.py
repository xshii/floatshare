"""Ground-truth 手算模块 — 独立于 features.py 的纯 numpy 特征重实现.

用途:
    1. tests/test_ml/test_features_values.py 的 expected 值计算 (编译期验证)
    2. ml/audit.py 的 spot-check 对拍 (运行期验证, 真 DB)

不用 pandas rolling/ewm (否则就是自己验自己). 全部纯 numpy 递推.

公式与 features.py 严格对齐. 任何 features.py 的 bugfix 必须同步这里, 否则 audit
会把正确结果标成 mismatch.
"""

from __future__ import annotations

import math

import numpy as np

LIMIT_UP_LOG: float = math.log(1.095)  # 与 features._LIMIT_UP_THRESHOLD=0.095 对齐


# --- Rolling helpers (pandas-compatible) -------------------------------------


def rolling_mean_np(
    arr: np.ndarray,
    window: int,
    min_periods: int | None = None,
) -> np.ndarray:
    mp = window if min_periods is None else min_periods
    out = np.full(len(arr), np.nan, dtype=np.float64)
    for i in range(len(arr)):
        s = arr[max(0, i - window + 1) : i + 1]
        valid = s[~np.isnan(s)]
        if len(valid) >= mp:
            out[i] = float(valid.mean())
    return out


def rolling_std_np(
    arr: np.ndarray,
    window: int,
    min_periods: int | None = None,
    ddof: int = 0,
) -> np.ndarray:
    mp = window if min_periods is None else min_periods
    out = np.full(len(arr), np.nan, dtype=np.float64)
    for i in range(len(arr)):
        s = arr[max(0, i - window + 1) : i + 1]
        valid = s[~np.isnan(s)]
        if len(valid) >= mp and len(valid) > ddof:
            out[i] = float(valid.std(ddof=ddof))
    return out


def rolling_sum_np(
    arr: np.ndarray,
    window: int,
    min_periods: int | None = None,
) -> np.ndarray:
    mp = window if min_periods is None else min_periods
    out = np.full(len(arr), np.nan, dtype=np.float64)
    for i in range(len(arr)):
        s = arr[max(0, i - window + 1) : i + 1]
        valid = s[~np.isnan(s)]
        if len(valid) >= mp:
            out[i] = float(valid.sum())
    return out


def rolling_max_np(
    arr: np.ndarray,
    window: int,
    min_periods: int | None = None,
) -> np.ndarray:
    mp = window if min_periods is None else min_periods
    out = np.full(len(arr), np.nan, dtype=np.float64)
    for i in range(len(arr)):
        s = arr[max(0, i - window + 1) : i + 1]
        valid = s[~np.isnan(s)]
        if len(valid) >= mp:
            out[i] = float(valid.max())
    return out


def rolling_min_np(
    arr: np.ndarray,
    window: int,
    min_periods: int | None = None,
) -> np.ndarray:
    mp = window if min_periods is None else min_periods
    out = np.full(len(arr), np.nan, dtype=np.float64)
    for i in range(len(arr)):
        s = arr[max(0, i - window + 1) : i + 1]
        valid = s[~np.isnan(s)]
        if len(valid) >= mp:
            out[i] = float(valid.min())
    return out


def rolling_corr_np(
    a: np.ndarray,
    b: np.ndarray,
    window: int,
    min_periods: int | None = None,
) -> np.ndarray:
    """pandas-compatible rolling corr (Pearson, ddof=1)."""
    mp = window if min_periods is None else min_periods
    out = np.full(len(a), np.nan, dtype=np.float64)
    for i in range(len(a)):
        lo = max(0, i - window + 1)
        s_a = a[lo : i + 1]
        s_b = b[lo : i + 1]
        mask = ~np.isnan(s_a) & ~np.isnan(s_b)
        if mask.sum() >= mp and mask.sum() > 1:
            va, vb = s_a[mask], s_b[mask]
            if va.std(ddof=1) == 0 or vb.std(ddof=1) == 0:
                out[i] = np.nan
            else:
                out[i] = float(np.corrcoef(va, vb)[0, 1])
    return out


# --- EWM (adjust=False) -------------------------------------------------------


def ewm_adjust_false_np(x: np.ndarray, alpha: float) -> np.ndarray:
    """pandas x.ewm(alpha, adjust=False).mean() 的 numpy 等价递推.

    NaN 处理: 首个有效值作为 y[i0] 初值; 后续 NaN 沿用 y[i-1]
    (与 pandas default ignore_na=False + adjust=False 一致).
    """
    y = np.full(len(x), np.nan, dtype=np.float64)
    initialized = False
    last = np.nan
    for i, v in enumerate(x):
        if np.isnan(v):
            y[i] = last if initialized else np.nan
            continue
        if not initialized:
            last = float(v)
            initialized = True
        else:
            last = (1 - alpha) * last + alpha * float(v)
        y[i] = last
    return y


def ewm_span_np(x: np.ndarray, span: int) -> np.ndarray:
    """pandas x.ewm(span=span, adjust=False).mean() — alpha = 2/(span+1)."""
    return ewm_adjust_false_np(x, alpha=2.0 / (span + 1))


# --- Technical indicators (features._rsi / _macd_hist / _kdj_j / _atr 对应) ---


def manual_rsi(close: np.ndarray, n: int = 12) -> np.ndarray:
    """RSI Wilder — 与 features._rsi 严格同配方 (默认 n=12 对齐 tushare rsi_12).

    2026-04-21 bugfix 后:
        gain=0 & loss=0  →  50 (warmup 首值)
        gain>0 & loss=0  →  100 (纯涨 → 超买)
    """
    delta = np.concatenate([[np.nan], np.diff(close)])
    gain = np.where(np.isnan(delta), np.nan, np.maximum(delta, 0))
    loss = np.where(np.isnan(delta), np.nan, np.maximum(-delta, 0))
    avg_gain = ewm_adjust_false_np(gain, alpha=1 / n)
    avg_loss = ewm_adjust_false_np(loss, alpha=1 / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss == 0, np.nan, avg_gain / avg_loss)
    rsi = 100 - 100 / (1 + rs)
    pure_up = (avg_loss == 0) & (avg_gain > 0)
    rsi = np.where(pure_up, 100.0, rsi)
    return np.where(np.isnan(rsi), 50.0, rsi)


def manual_macd_hist(close: np.ndarray) -> np.ndarray:
    ema_fast = ewm_span_np(close, 12)
    ema_slow = ewm_span_np(close, 26)
    dif = ema_fast - ema_slow
    dea = ewm_span_np(dif, 9)
    return (dif - dea) * 2


def manual_kdj_j(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    n: int = 9,
    m1: int = 3,
    m2: int = 3,
) -> np.ndarray:
    low_min = rolling_min_np(low, n, min_periods=n)
    high_max = rolling_max_np(high, n, min_periods=n)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = high_max - low_min
        rsv = np.where(denom == 0, np.nan, (close - low_min) / denom * 100)
    k = ewm_adjust_false_np(rsv, alpha=1 / m1)
    d = ewm_adjust_false_np(k, alpha=1 / m2)
    return 3 * k - 2 * d


def manual_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    n: int = 14,
) -> np.ndarray:
    prev_close = np.concatenate([[np.nan], close[:-1]])
    tr_stack = np.stack(
        [high - low, np.abs(high - prev_close), np.abs(low - prev_close)],
    )
    tr = np.nanmax(tr_stack, axis=0)
    return ewm_adjust_false_np(tr, alpha=1 / n)


# --- 涨停史专属 ---------------------------------------------------------------


def manual_is_limit(close: np.ndarray) -> np.ndarray:
    """log-return 超 log(1.095) → 1 (涨停), 否则 0. NaN (首日 delta) 当 0."""
    ret_1d = np.concatenate([[np.nan], np.log(close[1:] / close[:-1])])
    return np.where(np.isnan(ret_1d), 0.0, (ret_1d >= LIMIT_UP_LOG).astype(np.float64))


def manual_days_since_limit(is_limit: np.ndarray) -> np.ndarray:
    """2026-04-21 bugfix: 用 where 代替 replace(0, nan) 避免 Day 0 涨停污染."""
    n = len(is_limit)
    days_idx = np.arange(n, dtype=np.float64)
    # 在 is_limit=1 的位置保留 days_idx, 其它位置 NaN → ffill 传递
    masked = np.where(is_limit == 1, days_idx, np.nan)
    last_limit = np.full(n, np.nan)
    prev = np.nan
    for i, v in enumerate(masked):
        if not np.isnan(v):
            prev = v
        last_limit[i] = prev
    days_since = days_idx - last_limit
    days_since = np.where(np.isnan(days_since), 60.0, days_since)
    return np.minimum(days_since, 60.0) / 60.0


def manual_consecutive_up(close: np.ndarray) -> np.ndarray:
    """连续上涨天数 (cap 10) / 10."""
    ret_1d = np.concatenate([[np.nan], np.log(close[1:] / close[:-1])])
    up = np.where(np.isnan(ret_1d), 0, (ret_1d > 0).astype(int))
    streak = np.zeros(len(up), dtype=np.int64)
    for i in range(len(up)):
        if i == 0:
            streak[i] = 1 if up[i] else 0
        elif up[i] == 0:
            streak[i] = 0
        elif up[i] == up[i - 1]:
            streak[i] = streak[i - 1] + 1
        else:
            streak[i] = 1
    return np.minimum(streak, 10).astype(np.float64) / 10.0
