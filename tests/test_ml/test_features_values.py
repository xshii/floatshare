"""37 维 feature per-value 自验证 — 每个 feature 用纯 numpy 手算 ground truth.

## 验证契约

compute_features 末尾 `out.shift(1)` → feats 第 D 行 (0-indexed) 等于 helper 在
第 D-1 天的计算结果. 每个测试都按这个语义手算 expected:

    feats[D, "ret_1d"]     = log(close[D-1] / close[D-2])
    feats[D, "range_pct"]  = (high[D-1] - low[D-1]) / close[D-1]
    feats[D, "ma5_dev"]    = close[D-1] / mean(close[D-5..D-1]) - 1
    ...

## 独立性

- 所有 ground truth 用 **纯 numpy + math** 算, 不用 pandas rolling/ewm
  (否则就是用 pandas 验自己, 退化)
- EWM 类指标 (RSI/MACD/KDJ/ATR) 用独立 `_np_ewm` 递推重新实现
- 合成 panel 用 seed=42 deterministic, 含涨停 / 一字板 / 连续上涨 / 下跌
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from floatshare.ml.audit_manual import (
    LIMIT_UP_LOG as _LIMIT_UP_LOG,
)
from floatshare.ml.audit_manual import (
    manual_atr as _np_atr,
)
from floatshare.ml.audit_manual import (
    manual_kdj_j as _np_kdj_j,
)
from floatshare.ml.audit_manual import (
    manual_macd_hist as _np_macd_hist,
)
from floatshare.ml.audit_manual import (
    manual_rsi as _np_rsi,
)
from floatshare.ml.audit_manual import (
    rolling_corr_np as _rolling_corr,
)
from floatshare.ml.audit_manual import (
    rolling_max_np as _rolling_max,
)
from floatshare.ml.audit_manual import (
    rolling_mean_np as _rolling_mean,
)
from floatshare.ml.audit_manual import (
    rolling_std_np as _rolling_std,
)
from floatshare.ml.audit_manual import (
    rolling_sum_np as _rolling_sum,
)
from floatshare.ml.features import FEATURE_COLS, FEATURE_GROUPS, N_FEATURES, compute_features

# --- 常量 ---------------------------------------------------------------------

_N_DAYS = 80
_ATOL = 1e-10
_ATOL_EWM = 1e-6  # EWM 递推累积浮点误差略大


# --- fixture -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Raw:
    """panel 按 trade_date 排序后的原始列 — numpy, 0-indexed 与 feats 对齐."""

    close: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    volume: np.ndarray
    pe_ttm: np.ndarray
    pb: np.ndarray
    turnover_rate: np.ndarray
    total_mv: np.ndarray
    circ_mv: np.ndarray
    net_mf: np.ndarray
    buy_sm: np.ndarray
    buy_md: np.ndarray
    buy_lg: np.ndarray
    buy_elg: np.ndarray


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    """80 天 1 股合成 panel, seed=42, 含涨停 / 一字板 / 连涨 / 跌.

    关键注入事件 (便于 limit_up_history / is_yi_zi 等测试):
        day 30: ret_1d = log(1.10)  (涨停)
        day 45: ret_1d = log(1.095) (刚好涨停阈值)
        day 55: ret_1d = log(1.10)  (涨停)
        day 65: 一字板 (open == high == low == close 且涨停)
        day 70-73: 连续上涨 4 天
    """
    rng = np.random.default_rng(42)
    log_rets = rng.normal(0.001, 0.015, _N_DAYS)
    log_rets[30] = math.log(1.10)
    log_rets[45] = math.log(1.095)
    log_rets[55] = math.log(1.10)
    log_rets[65] = math.log(1.10)  # 一字板日的收盘 ret
    log_rets[70] = math.log(1.012)
    log_rets[71] = math.log(1.020)
    log_rets[72] = math.log(1.015)
    log_rets[73] = math.log(1.030)
    close = 10.0 * np.exp(np.cumsum(log_rets))

    # OHL: 默认有一定 intraday 波动; day 65 改一字板 (OHLC 全等)
    high = close * 1.005
    low = close * 0.995
    open_ = np.concatenate(
        [[close[0] * 0.998], close[:-1] * (1 + rng.normal(0, 0.003, _N_DAYS - 1))]
    )
    high[65] = close[65]
    low[65] = close[65]
    open_[65] = close[65]

    volume = rng.integers(100_000, 1_000_000, _N_DAYS).astype(np.float64)
    turnover_rate = volume / 1e7 * 100  # 简化的换手率 (百分比)
    total_mv = close * 1e8  # 假设 1 亿股
    circ_mv = total_mv * 0.7  # 流通 70%
    pe_ttm = np.full(_N_DAYS, 15.0) + rng.normal(0, 0.1, _N_DAYS)
    pb = np.full(_N_DAYS, 2.0) + rng.normal(0, 0.05, _N_DAYS)

    # moneyflow — 4 档买 + 4 档卖; 这里用 rng 生成 "大单比 ~ 0.4" 的结构
    buy_sm = rng.integers(1_000_000, 2_000_000, _N_DAYS).astype(np.float64)
    buy_md = rng.integers(500_000, 1_500_000, _N_DAYS).astype(np.float64)
    buy_lg = rng.integers(500_000, 1_200_000, _N_DAYS).astype(np.float64)
    buy_elg = rng.integers(100_000, 800_000, _N_DAYS).astype(np.float64)
    net_mf = (
        buy_sm
        + buy_md
        + buy_lg
        + buy_elg
        - rng.integers(1_000_000, 5_000_000, _N_DAYS).astype(np.float64)
    )

    dates = pd.date_range("2024-01-01", periods=_N_DAYS, freq="D")
    rows = [
        {
            "code": "AAA.SZ",
            "trade_date": dates[i],
            "open": float(open_[i]),
            "high": float(high[i]),
            "low": float(low[i]),
            "close": float(close[i]),
            "volume": float(volume[i]),
            "amount": float(volume[i] * close[i]),
            "pe_ttm": float(pe_ttm[i]),
            "pb": float(pb[i]),
            "turnover_rate": float(turnover_rate[i]),
            "total_mv": float(total_mv[i]),
            "circ_mv": float(circ_mv[i]),
            "net_mf_amount": float(net_mf[i]),
            "buy_sm_amount": float(buy_sm[i]),
            "buy_md_amount": float(buy_md[i]),
            "buy_lg_amount": float(buy_lg[i]),
            "buy_elg_amount": float(buy_elg[i]),
            "sell_sm_amount": float(buy_sm[i] * 0.9),
            "sell_md_amount": float(buy_md[i] * 0.9),
            "sell_lg_amount": float(buy_lg[i] * 0.95),
            "sell_elg_amount": float(buy_elg[i] * 0.95),
        }
        for i in range(_N_DAYS)
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def raw(panel: pd.DataFrame) -> _Raw:
    p = panel.sort_values("trade_date").reset_index(drop=True)
    return _Raw(
        close=p["close"].to_numpy(),
        open=p["open"].to_numpy(),
        high=p["high"].to_numpy(),
        low=p["low"].to_numpy(),
        volume=p["volume"].to_numpy(),
        pe_ttm=p["pe_ttm"].to_numpy(),
        pb=p["pb"].to_numpy(),
        turnover_rate=p["turnover_rate"].to_numpy(),
        total_mv=p["total_mv"].to_numpy(),
        circ_mv=p["circ_mv"].to_numpy(),
        net_mf=p["net_mf_amount"].to_numpy(),
        buy_sm=p["buy_sm_amount"].to_numpy(),
        buy_md=p["buy_md_amount"].to_numpy(),
        buy_lg=p["buy_lg_amount"].to_numpy(),
        buy_elg=p["buy_elg_amount"].to_numpy(),
    )


@pytest.fixture(scope="module")
def feats(panel: pd.DataFrame) -> pd.DataFrame:
    return compute_features(panel).sort_values("trade_date").reset_index(drop=True)


def _assert_col(
    feats: pd.DataFrame,
    col: str,
    expected: np.ndarray,
    atol: float = _ATOL,
) -> None:
    """断言 feats[col] 的非 NaN 位置与 expected 相等 (忽略 NaN 位置).

    NaN 期望处: 断言 actual 也是 NaN.
    """
    actual = feats[col].to_numpy(dtype=np.float64)
    assert len(actual) == len(expected), f"{col}: 长度 {len(actual)} vs {len(expected)}"
    for i in range(len(actual)):
        e, a = expected[i], actual[i]
        if np.isnan(e):
            assert np.isnan(a), f"{col}@{i}: expected NaN, got {a}"
        else:
            assert not np.isnan(a), f"{col}@{i}: expected {e}, got NaN"
            assert abs(a - e) < atol, f"{col}@{i}: expected {e}, got {a} (diff {a - e})"


def _helper_to_shift(helper_arr: np.ndarray) -> np.ndarray:
    """compute_features 末尾 shift(1) — 把 helper 计算值后移一位, 首位置 NaN."""
    out = np.full(len(helper_arr), np.nan, dtype=np.float64)
    out[1:] = helper_arr[:-1]
    return out


# =============================================================================
# Meta
# =============================================================================


class TestMeta:
    def test_feature_cols_count_39(self) -> None:
        # 37 OHLCV/daily_basic 派生 + 2 新闻 (news_mentioned_t / _t1)
        assert N_FEATURES == 39
        assert len(FEATURE_COLS) == 39

    def test_feature_groups_sum_to_feature_cols(self) -> None:
        """FEATURE_GROUPS 是 _DailyDerivedSource (shift_days=1) 的子集.

        FEATURE_COLS 除了这些旧组, 还包含新闻 source (shift_days=0) 的 2 个 col.
        """
        flat = tuple(c for _, cols in FEATURE_GROUPS for c in cols)
        assert flat == FEATURE_COLS[: len(flat)]
        # 新闻 source 的列在末尾
        assert FEATURE_COLS[-2:] == ("news_mentioned_t", "news_mentioned_t1")

    def test_shift_days_1_sources_first_row_nan(self, feats: pd.DataFrame) -> None:
        """shift_days=1 的 source 在第 0 行 NaN; shift_days=0 的 (如 news) 不 NaN."""
        from floatshare.ml.features import FEATURE_SOURCES

        row0 = feats.iloc[0]
        for source in FEATURE_SOURCES:
            if source.shift_days == 1:
                for col in source.output_cols:
                    assert np.isnan(row0[col]), f"{source.name}.{col}@0 应为 NaN (shift_days=1)"
            else:
                # shift_days=0: 第 0 行有值 (shift 不移位, news 全 0 或从 DB 读)
                for col in source.output_cols:
                    assert not np.isnan(row0[col]), (
                        f"{source.name}.{col}@0 不应为 NaN (shift_days=0), got NaN"
                    )


# =============================================================================
# Group 1: 价格派生 (5)
# =============================================================================


class TestPriceDerived:
    def test_ret_1d(self, feats: pd.DataFrame, raw: _Raw) -> None:
        helper = np.concatenate([[np.nan], np.log(raw.close[1:] / raw.close[:-1])])
        _assert_col(feats, "ret_1d", _helper_to_shift(helper))

    def test_ret_5d(self, feats: pd.DataFrame, raw: _Raw) -> None:
        helper = np.full(len(raw.close), np.nan)
        helper[5:] = np.log(raw.close[5:] / raw.close[:-5])
        _assert_col(feats, "ret_5d", _helper_to_shift(helper))

    def test_ret_20d(self, feats: pd.DataFrame, raw: _Raw) -> None:
        helper = np.full(len(raw.close), np.nan)
        helper[20:] = np.log(raw.close[20:] / raw.close[:-20])
        _assert_col(feats, "ret_20d", _helper_to_shift(helper))

    def test_range_pct(self, feats: pd.DataFrame, raw: _Raw) -> None:
        helper = (raw.high - raw.low) / raw.close
        _assert_col(feats, "range_pct", _helper_to_shift(helper))

    def test_gap(self, feats: pd.DataFrame, raw: _Raw) -> None:
        prev_close = np.concatenate([[np.nan], raw.close[:-1]])
        helper = (raw.open - prev_close) / prev_close
        _assert_col(feats, "gap", _helper_to_shift(helper))


# =============================================================================
# Group 2: 均线偏离 (4)
# =============================================================================


class TestMADeviation:
    def test_ma5_dev(self, feats: pd.DataFrame, raw: _Raw) -> None:
        ma5 = _rolling_mean(raw.close, 5, min_periods=5)
        helper = raw.close / ma5 - 1
        _assert_col(feats, "ma5_dev", _helper_to_shift(helper))

    def test_ma20_dev(self, feats: pd.DataFrame, raw: _Raw) -> None:
        ma20 = _rolling_mean(raw.close, 20, min_periods=20)
        helper = raw.close / ma20 - 1
        _assert_col(feats, "ma20_dev", _helper_to_shift(helper))

    def test_ma60_dev(self, feats: pd.DataFrame, raw: _Raw) -> None:
        ma60 = _rolling_mean(raw.close, 60, min_periods=60)
        helper = raw.close / ma60 - 1
        _assert_col(feats, "ma60_dev", _helper_to_shift(helper))

    def test_ma_short_long(self, feats: pd.DataFrame, raw: _Raw) -> None:
        ma5 = _rolling_mean(raw.close, 5, min_periods=5)
        ma20 = _rolling_mean(raw.close, 20, min_periods=20)
        helper = ma5 / ma20 - 1
        _assert_col(feats, "ma_short_long", _helper_to_shift(helper))


# =============================================================================
# Group 3: 量能 (3)
# =============================================================================


class TestVolume:
    def test_vol_z20(self, feats: pd.DataFrame, raw: _Raw) -> None:
        vol_ma20 = _rolling_mean(raw.volume, 20, min_periods=20)
        vol_std20 = _rolling_std(raw.volume, 20, min_periods=20, ddof=0)
        helper = (raw.volume - vol_ma20) / vol_std20
        _assert_col(feats, "vol_z20", _helper_to_shift(helper))

    def test_vol_ratio_5(self, feats: pd.DataFrame, raw: _Raw) -> None:
        vol_ma5 = _rolling_mean(raw.volume, 5, min_periods=5)
        helper = raw.volume / vol_ma5
        _assert_col(feats, "vol_ratio_5", _helper_to_shift(helper))

    def test_vol_ret_corr20(self, feats: pd.DataFrame, raw: _Raw) -> None:
        ret_1d = np.concatenate([[np.nan], np.log(raw.close[1:] / raw.close[:-1])])
        helper = _rolling_corr(ret_1d, raw.volume, 20, min_periods=20)
        _assert_col(feats, "vol_ret_corr20", _helper_to_shift(helper), atol=1e-8)


# =============================================================================
# Group 4: 技术指标 (4) — EWM-based, 用独立 _np_ewm 对拍
# =============================================================================


class TestTechnical:
    def test_rsi12(self, feats: pd.DataFrame, raw: _Raw) -> None:
        helper = _np_rsi(raw.close, n=12) / 100 - 0.5
        _assert_col(feats, "rsi12", _helper_to_shift(helper), atol=_ATOL_EWM)

    def test_rsi12_all_up_boundary(self, feats: pd.DataFrame) -> None:
        """边界: 恒定上涨 → RSI=100 → feats=0.5 (业界标准, 2026-04-21 bugfix).

        周期 n=12 (对齐 tushare stk_factor.rsi_12). `where(loss > 0, nan)` + `pure_up → 100`:
            - gain=0 & loss=0 (warmup 初期) → 50 (无信号)
            - gain>0 & loss=0 (纯涨 = 超买) → 100
        """
        close = 10.0 * (1.01 ** np.arange(50))
        panel = pd.DataFrame(
            [
                {
                    "code": "UP.SZ",
                    "trade_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                    "open": c * 0.999,
                    "high": c * 1.001,
                    "low": c * 0.999,
                    "close": c,
                    "volume": 1e5,
                    "amount": 1e5 * c,
                    "pe_ttm": 15.0,
                    "pb": 2.0,
                    "turnover_rate": 1.0,
                    "total_mv": 1e10,
                    "circ_mv": 7e9,
                    "net_mf_amount": 0.0,
                    "buy_sm_amount": 1e5,
                    "buy_md_amount": 1e5,
                    "buy_lg_amount": 1e5,
                    "buy_elg_amount": 1e5,
                    "sell_sm_amount": 9e4,
                    "sell_md_amount": 9e4,
                    "sell_lg_amount": 9e4,
                    "sell_elg_amount": 9e4,
                }
                for i, c in enumerate(close)
            ]
        )
        fe = compute_features(panel).sort_values("trade_date").reset_index(drop=True)
        tail = fe["rsi12"].iloc[-1]
        # 业界标准: 纯涨 → RSI=100 → feats = 100/100 - 0.5 = 0.5
        assert tail == pytest.approx(0.5, abs=1e-6), (
            f"all-up RSI12 应为 0.5 (bugfix + n=12 对齐 tushare 后), got {tail}"
        )

    def test_macd_hist_norm(self, feats: pd.DataFrame, raw: _Raw) -> None:
        helper = _np_macd_hist(raw.close) / raw.close
        _assert_col(feats, "macd_hist_norm", _helper_to_shift(helper), atol=_ATOL_EWM)

    def test_kdj_j(self, feats: pd.DataFrame, raw: _Raw) -> None:
        helper = _np_kdj_j(raw.high, raw.low, raw.close) / 100 - 0.5
        _assert_col(feats, "kdj_j", _helper_to_shift(helper), atol=_ATOL_EWM)

    def test_atr14_pct(self, feats: pd.DataFrame, raw: _Raw) -> None:
        helper = _np_atr(raw.high, raw.low, raw.close, n=14) / raw.close
        _assert_col(feats, "atr14_pct", _helper_to_shift(helper), atol=_ATOL_EWM)

    def test_atr14_zero_when_flat(self) -> None:
        """边界: H=L=C 恒定 → ATR=0 → atr14_pct=0."""
        n = 40
        close = np.full(n, 10.0)
        panel = pd.DataFrame(
            [
                {
                    "code": "F.SZ",
                    "trade_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                    "open": 10.0,
                    "high": 10.0,
                    "low": 10.0,
                    "close": 10.0,
                    "volume": 1e5,
                    "amount": 1e6,
                    "pe_ttm": 15.0,
                    "pb": 2.0,
                    "turnover_rate": 1.0,
                    "total_mv": 1e10,
                    "circ_mv": 7e9,
                    "net_mf_amount": 0.0,
                    "buy_sm_amount": 1e5,
                    "buy_md_amount": 1e5,
                    "buy_lg_amount": 1e5,
                    "buy_elg_amount": 1e5,
                    "sell_sm_amount": 9e4,
                    "sell_md_amount": 9e4,
                    "sell_lg_amount": 9e4,
                    "sell_elg_amount": 9e4,
                }
                for i in range(n)
            ]
        )
        _ = close  # deterministic constant
        fe = compute_features(panel).sort_values("trade_date").reset_index(drop=True)
        tail = fe["atr14_pct"].iloc[-1]
        assert tail == pytest.approx(0.0, abs=1e-9), f"flat ATR → 0, got {tail}"


# =============================================================================
# Group 5: 基本面 (4)
# =============================================================================


class TestFundamental:
    def test_pe_log(self, feats: pd.DataFrame, raw: _Raw) -> None:
        helper = np.log1p(np.maximum(raw.pe_ttm, 0))
        _assert_col(feats, "pe_log", _helper_to_shift(helper))

    def test_pb_log(self, feats: pd.DataFrame, raw: _Raw) -> None:
        helper = np.log1p(np.maximum(raw.pb, 0))
        _assert_col(feats, "pb_log", _helper_to_shift(helper))

    def test_turnover(self, feats: pd.DataFrame, raw: _Raw) -> None:
        _assert_col(feats, "turnover", _helper_to_shift(raw.turnover_rate))

    def test_mv_log(self, feats: pd.DataFrame, raw: _Raw) -> None:
        helper = np.log(np.maximum(raw.total_mv, 1))
        _assert_col(feats, "mv_log", _helper_to_shift(helper))


# =============================================================================
# Group 6: 资金流 (2)
# =============================================================================


class TestMoneyFlow:
    def test_inflow_pct(self, feats: pd.DataFrame, raw: _Raw) -> None:
        total_mv_safe = np.where(raw.total_mv == 0, np.nan, raw.total_mv)
        helper = raw.net_mf / total_mv_safe
        _assert_col(feats, "inflow_pct", _helper_to_shift(helper))

    def test_big_ratio(self, feats: pd.DataFrame, raw: _Raw) -> None:
        big = raw.buy_lg + raw.buy_elg
        total = big + raw.buy_sm + raw.buy_md
        total_safe = np.where(total == 0, np.nan, total)
        helper = big / total_safe
        _assert_col(feats, "big_ratio", _helper_to_shift(helper))


# =============================================================================
# Group 7: 涨停史 (6) — 抓涨停任务核心
# =============================================================================


class TestLimitUpHistory:
    @staticmethod
    def _is_limit(close: np.ndarray) -> np.ndarray:
        ret_1d = np.concatenate([[np.nan], np.log(close[1:] / close[:-1])])
        # features 里 is_limit 是 float64 (ret_1d >= threshold).astype(float64)
        # NaN >= threshold → False; 这里保留 NaN → 转换时需小心
        return np.where(np.isnan(ret_1d), 0.0, (ret_1d >= _LIMIT_UP_LOG).astype(np.float64))

    def test_days_since_limit(self, feats: pd.DataFrame, raw: _Raw) -> None:
        is_limit = self._is_limit(raw.close)
        days_idx = np.arange(len(raw.close), dtype=np.float64)
        # 模拟: last_limit_idx = (days_idx * is_limit).replace(0, nan).ffill()
        masked = days_idx * is_limit
        masked = np.where(masked == 0, np.nan, masked)
        last_limit = np.full_like(masked, np.nan)
        prev = np.nan
        for i, v in enumerate(masked):
            if not np.isnan(v):
                prev = v
            last_limit[i] = prev
        days_since = days_idx - last_limit
        days_since = np.where(np.isnan(days_since), 60.0, days_since)
        helper = np.minimum(days_since, 60.0) / 60.0
        _assert_col(feats, "days_since_limit", _helper_to_shift(helper))

    def test_n_limits_20d(self, feats: pd.DataFrame, raw: _Raw) -> None:
        is_limit = self._is_limit(raw.close)
        helper = _rolling_sum(is_limit, 20, min_periods=1) / 20.0
        _assert_col(feats, "n_limits_20d", _helper_to_shift(helper))

    def test_n_limits_60d(self, feats: pd.DataFrame, raw: _Raw) -> None:
        is_limit = self._is_limit(raw.close)
        helper = _rolling_sum(is_limit, 60, min_periods=1) / 60.0
        _assert_col(feats, "n_limits_60d", _helper_to_shift(helper))

    def test_max_ret_20d(self, feats: pd.DataFrame, raw: _Raw) -> None:
        ret_1d = np.concatenate([[np.nan], np.log(raw.close[1:] / raw.close[:-1])])
        # min_periods=1 — pandas rolling max 在 ret_1d[0]=NaN 但 window 里有 1 个 non-NaN 时返回它
        helper = _rolling_max(ret_1d, 20, min_periods=1)
        _assert_col(feats, "max_ret_20d", _helper_to_shift(helper))

    def test_consecutive_up(self, feats: pd.DataFrame, raw: _Raw) -> None:
        """连续上涨天数 (cap 10) / 10."""
        ret_1d = np.concatenate([[np.nan], np.log(raw.close[1:] / raw.close[:-1])])
        up = np.where(np.isnan(ret_1d), 0, (ret_1d > 0).astype(int))
        # 模拟: streak_id = (up != up.shift()).cumsum(); streak_count = up.groupby(streak_id).cumsum() * up
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
        helper = np.minimum(streak, 10).astype(np.float64) / 10.0
        _assert_col(feats, "consecutive_up", _helper_to_shift(helper))

    def test_is_yi_zi(self, feats: pd.DataFrame, raw: _Raw) -> None:
        ret_1d = np.concatenate([[np.nan], np.log(raw.close[1:] / raw.close[:-1])])
        eq_oh = raw.open == raw.high
        eq_hl = raw.high == raw.low
        eq_lc = raw.low == raw.close
        limit = np.where(np.isnan(ret_1d), False, ret_1d >= _LIMIT_UP_LOG)
        helper = (eq_oh & eq_hl & eq_lc & limit).astype(np.float64)
        _assert_col(feats, "is_yi_zi", _helper_to_shift(helper))

    def test_yi_zi_detected_at_day_65(self, feats: pd.DataFrame) -> None:
        """明确: day 65 是一字板 → feats[66, is_yi_zi] = 1 (shift 后向后一位)."""
        assert feats.loc[66, "is_yi_zi"] == 1.0, (
            f"day 66 (shift 后对应 day 65 一字板) 应为 1.0, got {feats.loc[66, 'is_yi_zi']}"
        )


# =============================================================================
# Group 8: 规模 / 换手 / 波动 多尺度 (6)
# =============================================================================


class TestScaleTurnoverVola:
    def test_circ_mv_log(self, feats: pd.DataFrame, raw: _Raw) -> None:
        helper = np.log(np.maximum(raw.circ_mv, 1))
        _assert_col(feats, "circ_mv_log", _helper_to_shift(helper))

    def test_circ_mv_z_self60(self, feats: pd.DataFrame, raw: _Raw) -> None:
        mean60 = _rolling_mean(raw.circ_mv, 60, min_periods=20)
        std60 = _rolling_std(raw.circ_mv, 60, min_periods=20, ddof=0)
        helper = (raw.circ_mv - mean60) / (std60 + 1e-9)
        _assert_col(feats, "circ_mv_z_self60", _helper_to_shift(helper), atol=1e-8)

    def test_turnover_ma20(self, feats: pd.DataFrame, raw: _Raw) -> None:
        helper = _rolling_mean(raw.turnover_rate, 20, min_periods=5)
        _assert_col(feats, "turnover_ma20", _helper_to_shift(helper))

    def test_turnover_self_z60(self, feats: pd.DataFrame, raw: _Raw) -> None:
        mean60 = _rolling_mean(raw.turnover_rate, 60, min_periods=20)
        std60 = _rolling_std(raw.turnover_rate, 60, min_periods=20, ddof=0)
        helper = (raw.turnover_rate - mean60) / (std60 + 1e-9)
        _assert_col(feats, "turnover_self_z60", _helper_to_shift(helper), atol=1e-8)

    def test_vola_60d(self, feats: pd.DataFrame, raw: _Raw) -> None:
        ret_1d = np.concatenate([[np.nan], np.log(raw.close[1:] / raw.close[:-1])])
        helper = _rolling_std(ret_1d, 60, min_periods=20, ddof=0)
        _assert_col(feats, "vola_60d", _helper_to_shift(helper), atol=1e-8)

    def test_vola_ratio_20_60(self, feats: pd.DataFrame, raw: _Raw) -> None:
        ret_1d = np.concatenate([[np.nan], np.log(raw.close[1:] / raw.close[:-1])])
        vola_20 = _rolling_std(ret_1d, 20, min_periods=5, ddof=0)
        vola_60 = _rolling_std(ret_1d, 60, min_periods=20, ddof=0)
        helper = vola_20 / (vola_60 + 1e-9)
        _assert_col(feats, "vola_ratio_20_60", _helper_to_shift(helper), atol=1e-8)


# =============================================================================
# Group 9: 价量匹配 (3)
# =============================================================================


class TestPriceVolumeMatch:
    def test_price_vol_match5(self, feats: pd.DataFrame, raw: _Raw) -> None:
        ret_5d = np.full(len(raw.close), np.nan)
        ret_5d[5:] = np.log(raw.close[5:] / raw.close[:-5])
        vol_ratio_5 = raw.volume / _rolling_mean(raw.volume, 5, min_periods=5)
        helper = ret_5d * np.tanh(vol_ratio_5 - 1)
        _assert_col(feats, "price_vol_match5", _helper_to_shift(helper), atol=1e-8)

    def test_inflow_5d_sum(self, feats: pd.DataFrame, raw: _Raw) -> None:
        inflow_5d = _rolling_sum(raw.net_mf, 5, min_periods=2)
        cmv_safe = np.where(raw.circ_mv == 0, np.nan, raw.circ_mv)
        helper = inflow_5d / cmv_safe
        _assert_col(feats, "inflow_5d_sum", _helper_to_shift(helper), atol=1e-8)

    def test_vol_ret_corr60(self, feats: pd.DataFrame, raw: _Raw) -> None:
        ret_1d = np.concatenate([[np.nan], np.log(raw.close[1:] / raw.close[:-1])])
        helper = _rolling_corr(ret_1d, raw.volume, 60, min_periods=20)
        _assert_col(feats, "vol_ret_corr60", _helper_to_shift(helper), atol=1e-8)
