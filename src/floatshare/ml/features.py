"""特征工程 — long-format DataFrame → 37 维特征.

## 时间索引契约 (反前视核心)

特征计算通过 `FeatureSource` 抽象统一声明时间可用性:

- `availability_cutoff_hour`: T 日当天几点后该数据可用 (24h 制)
- `shift_days`: 自动 = 1 if cutoff > DECISION_HOUR(22) else 0

pipeline 决策时点在 T 日 22:30 (stage 4). 所以:

- OHLCV / daily_basic / moneyflow (cutoff<=18) 虽然 T 日 22:30 可见, 但为
  保持与旧 v8 ckpt 兼容, `_DailyDerivedSource` 默认 cutoff=19, shift_days=1
  (feats[D] = helper@(D-1)). 未来 v10+ 可以把部分 source 改 cutoff<=22 放开.
- CCTV 新闻联播 (cutoff=20): shift_days=0, feats[D] = helper@D 直接用 D 日新闻.
- 财报 fina_indicator (ann_date=T, cutoff=25): shift_days=1 不够, 需要 PIT join
  用 `ann_date < trade_date` 处理 (未来加).

每个 `_feat_*` 的 docstring 包含 **"手动推导"** 段.

## 特征列表 (37 维, 10 组)

[价格派生 5]   ret_1d, ret_5d, ret_20d, range_pct, gap
[均线偏离 4]   ma5_dev, ma20_dev, ma60_dev, ma_short_long
[量能 3]      vol_z20, vol_ratio_5, vol_ret_corr20
[技术指标 4]   rsi12, macd_hist_norm, kdj_j, atr14_pct
[基本面 4]    pe_log, pb_log, turnover, mv_log
[资金流 2]    inflow_pct, big_ratio
[涨停史 6]    days_since_limit, n_limits_20d, n_limits_60d,
             max_ret_20d, consecutive_up, is_yi_zi
[规模波动 6]  circ_mv_log, circ_mv_z_self60, turnover_ma20,
             turnover_self_z60, vola_60d, vola_ratio_20_60
[价量匹配 3]  price_vol_match5, inflow_5d_sum, vol_ret_corr60
"""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, cast

import numpy as np
import pandas as pd

# 涨停判定阈值 — 只针对主板 (9.5% ~ 10%); 创业板/科创板 universe 已过滤
_LIMIT_UP_THRESHOLD = 0.095

# 每个 feature helper 类型: (g 原始 DataFrame, derived 已算列) → 新列 dict.
# dict value 实际都是 pd.Series, 但某些 pandas 链式操作返回类型是 Series | DataFrame |
# ndarray 的 union (pyright 的 stubs 偏严), 这里用 Any 放宽.
_DerivedCols = dict[str, Any]
_FeatureHelper = Callable[[pd.DataFrame, _DerivedCols], _DerivedCols]


# =============================================================================
# FeatureSource ABC — 统一声明时间可用性契约
# =============================================================================


@dataclass(frozen=True, slots=True)
class ColumnQualityRule:
    """单个 feature 列的数据质量契约 — stage 2 后验因子排查用.

    **NaN 处理** (懒检查 — 仅真出现 NaN 时才校验):
        nan_fill:            替换策略. **默认 None → 若运行时遇 NaN 且未配则 raise**
            - 'zero':   NaN → 0 (cross_sectional_zscore 之后也是 0)
            - 'prev':   ffill, 用时序前值 (适合 daily_basic 停牌日)
            - 'median': 当日截面中位数 (适合行业/市场级特征)
            - 'keep':   保留 NaN (让模型学 "缺失")
            - None:     该 feature 预期不该有 NaN; 出现即 AuditFailedError
        nan_alert_threshold: NaN 比例 (warm-up 期后) 超过此值 → raise
        warmup_days:         前 N 天 NaN 不告警 (rolling window 暖机)

    **值域** (滚动 252 天 p0.5/p99.5 优先, 硬阈值 fallback):
        expected_min/max:     None=仅依赖滚动阈值; 非 None=滚动样本不够时的 fallback
        out_of_range_action:  超限处理策略 (影响归一化前的数据清洗)
            - 'clip':        winsorize 到 [p0.5, p99.5] (默认, 业界标准)
                              防止极端值污染 cross_sectional_zscore 的 mean/std
            - 'alert':       不改值, 仅 raise (交给人工处理)
            - 'drop_stock':  超限的股当日所有 feature 置 NaN (保守, 丢数据)

    Design: frozen dataclass, 由 FeatureSource.quality_rules 声明.
    错误处理: 超限/缺策略 统一通过 AuditAlert + run_feature_audit 的 raise.
    """

    nan_fill: Literal["zero", "prev", "median", "keep"] | None = None
    nan_alert_threshold: float = 0.10
    warmup_days: int = 0

    expected_min: float | None = None
    expected_max: float | None = None
    out_of_range_action: Literal["clip", "alert", "drop_stock"] = "clip"


class FeatureSource(ABC):
    """一组 feature 的抽象, 声明时间可用性 + 质量契约 + 提供计算.

    核心契约: 每个 source 必须说明:
        1. "T 日 22:30 决策时能否看到 T 日数据" (availability_cutoff_hour)
        2. 每列的质量规则 (quality_rules) — NaN / 值域 / warmup

    DECISION_HOUR = 22 (pipeline stage 4 开跑时点).
    cutoff <= DECISION_HOUR  → 当日可见 → shift_days=0 (feats[D] = helper@D)
    cutoff >  DECISION_HOUR  → 当日不可见 → shift_days=1 (feats[D] = helper@(D-1))
    """

    DECISION_HOUR: ClassVar[int] = 22

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def output_cols(self) -> tuple[str, ...]: ...

    @property
    @abstractmethod
    def availability_cutoff_hour(self) -> int:
        """T 日几点后可用 (24h 制, 0-25 — 25 = 次日也不可用)."""

    @property
    def can_use_same_day(self) -> bool:
        return self.availability_cutoff_hour <= self.DECISION_HOUR

    @property
    def shift_days(self) -> int:
        return 0 if self.can_use_same_day else 1

    @property
    @abstractmethod
    def quality_rules(self) -> dict[str, ColumnQualityRule]:
        """每列的质量规则 — **必须**为每个 output_col 提供 rule.

        缺任何一维 → raise AuditFailedError (阻止 audit 运行).
        运行时 expected_min/max 被滚动 252 天 p0.5/p99.5 覆盖 (rule 的硬阈值作 fallback).
        """

    @abstractmethod
    def compute(self, panel: pd.DataFrame, derived: _DerivedCols) -> _DerivedCols:
        """输入 panel (单只股, index=trade_date) + 已算列, 输出 {col: series}."""


@dataclass(frozen=True, slots=True)
class _DailyDerivedSource(FeatureSource):
    """包装旧式 helper fn 为 FeatureSource (shift_days=1 保持原 out.shift(1) 等价).

    为**兼容 v8 ckpt 语义**锁定 cutoff=23 → shift_days=1.
    要改 cutoff=18 放开到 shift_days=0 需改 v10+ 训练.

    _rules: 每列 ColumnQualityRule 字典. **必须**覆盖所有 _cols, 否则 raise.
    """

    _name: str
    _cols: tuple[str, ...]
    _helper: _FeatureHelper
    _cutoff: int = 23
    _rules: dict[str, ColumnQualityRule] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self._name

    @property
    def output_cols(self) -> tuple[str, ...]:
        return self._cols

    @property
    def availability_cutoff_hour(self) -> int:
        return self._cutoff

    @property
    def quality_rules(self) -> dict[str, ColumnQualityRule]:
        # 缺 rule 的 feature fallback 到空 rule (nan_fill=None). audit 遇 NaN 时再 raise.
        return {col: self._rules.get(col, ColumnQualityRule()) for col in self._cols}

    def compute(self, panel: pd.DataFrame, derived: _DerivedCols) -> _DerivedCols:
        return self._helper(panel, derived)


class AuditConfigError(RuntimeError):
    """FeatureSource 配置错误 (缺 rule / 非法值). 在 import 时就抛, 阻止 audit."""


# =============================================================================
# Feature helpers (分组. 每组 docstring 含手动推导).
# 执行顺序 = FEATURE_GROUPS 顺序. 后面的 helper 能读前面 helper 填进 derived 的列.
# =============================================================================


def _feat_price_derived(
    g: pd.DataFrame,
    _: _DerivedCols,
) -> _DerivedCols:
    """价格派生 5 维.

    手动推导 (shift 前值 → shift 后 D 日含义):
        ret_1d[D]   = log(close[D] / close[D-1])
                    → shift 后 = log(close[D-1] / close[D-2])      (D-1 收盘后可见)
        ret_5d[D]   = log(close[D] / close[D-5])                   (shift 后 D-1 已知)
        ret_20d[D]  = log(close[D] / close[D-20])                  (shift 后 D-1 已知)
        range_pct[D]= (high[D] - low[D]) / close[D]                (当日高低价, shift 后 D-1 已知)
        gap[D]      = (open[D] - close[D-1]) / close[D-1]          (shift 后 D-1 开盘 vs D-2 收盘)

    所有 5 维仅用当日或更早的 OHLC, 末尾 shift(1) 即让 D 日特征 = D-1 值, 无前视.
    """
    close = cast(pd.Series, g["close"].astype("float64"))
    open_ = cast(pd.Series, g["open"].astype("float64"))
    high = cast(pd.Series, g["high"].astype("float64"))
    low = cast(pd.Series, g["low"].astype("float64"))
    return {
        "ret_1d": np.log(close / close.shift(1)),
        "ret_5d": np.log(close / close.shift(5)),
        "ret_20d": np.log(close / close.shift(20)),
        "range_pct": (high - low) / close,
        "gap": (open_ - close.shift(1)) / close.shift(1),
    }


def _feat_ma_deviation(
    g: pd.DataFrame,
    _: _DerivedCols,
) -> _DerivedCols:
    """均线偏离 4 维.

    手动推导:
        ma5[D]  = mean(close[D-4..D])                              (含当日, 5 天均线)
        ma20[D] = mean(close[D-19..D])
        ma60[D] = mean(close[D-59..D])
        ma5_dev[D]     = close[D] / ma5[D] - 1                     (shift 后 D-1 已知)
        ma20_dev[D]    = close[D] / ma20[D] - 1
        ma60_dev[D]    = close[D] / ma60[D] - 1
        ma_short_long  = ma5 / ma20 - 1                            (短期均线相对长期)

    min_periods=5/20/60 确保 rolling 窗口未满时输出 NaN, 不会用更少天数凑出伪均线.
    """
    close = cast(pd.Series, g["close"].astype("float64"))
    ma5 = close.rolling(5, min_periods=5).mean()
    ma20 = close.rolling(20, min_periods=20).mean()
    ma60 = close.rolling(60, min_periods=60).mean()
    return {
        "ma5_dev": close / ma5 - 1,
        "ma20_dev": close / ma20 - 1,
        "ma60_dev": close / ma60 - 1,
        "ma_short_long": ma5 / ma20 - 1,
    }


def _feat_volume(
    g: pd.DataFrame,
    derived: _DerivedCols,
) -> _DerivedCols:
    """量能 3 维.

    手动推导:
        vol_z20[D]       = (volume[D] - mean(volume[D-19..D])) / std(volume[D-19..D])
                         → shift 后 D-1 已知异常量能
        vol_ratio_5[D]   = volume[D] / mean(volume[D-4..D])
        vol_ret_corr20[D]= corr(ret_1d[D-19..D], volume[D-19..D])  (20 天价量相关)

    说明: ret_1d 已在 _feat_price_derived 计算好, 通过 derived 复用 (避免重复计算
    log 返回). 末尾 shift(1) 把窗口语义变成 "D 日特征 = D-1 日 20 天窗口".
    """
    volume = cast(pd.Series, g["volume"].astype("float64"))
    ret_1d = derived["ret_1d"]
    vol_ma20 = volume.rolling(20, min_periods=20).mean()
    vol_std20 = volume.rolling(20, min_periods=20).std(ddof=0)
    return {
        "vol_z20": (volume - vol_ma20) / vol_std20,
        "vol_ratio_5": volume / volume.rolling(5, min_periods=5).mean(),
        "vol_ret_corr20": ret_1d.rolling(20, min_periods=20).corr(volume),
    }


def _feat_technical(
    g: pd.DataFrame,
    _: _DerivedCols,
) -> _DerivedCols:
    """技术指标 4 维 (RSI / MACD hist / KDJ-J / ATR14).

    **基于前复权价** (close_qfq / high_qfq / low_qfq), 跟 tushare stk_factor 对齐:
        - 避免除权日价格跳空污染 EWM 累积 (MACD / RSI / KDJ 全受影响)
        - 跟业界 (tushare / 同花顺 / 东财) 的 RSI/MACD/KDJ 值匹配
        - load_panel 保证 最后一天 close_qfq == raw close (latest_adj_factor 归一化)
        - 其它 feature (ret_1d / range_pct / 涨停史) 仍用 raw close, 保留除权事件信号

    手动推导 (全部只用当日及之前 OHLC, shift 后 D 日 = D-1 值):
        rsi12[D]           = EMA 风格的 gain/loss 比 (用 close_qfq[D-11..D]) → [-0.5, 0.5]
                             (n=12 跟 tushare stk_factor.rsi_12 对齐, 方便第三方对拍)
        macd_hist_norm[D]  = (DIF - DEA) / close_qfq[D]                     (MACD 柱归一)
        kdj_j[D]           = 3K - 2D, K/D 都是 9-period RSV 平滑            → [-0.5, 0.5]
        atr14_pct[D]       = ATR(14) / close_qfq[D]                         (真实波幅占比)

    所有 rolling / ewm 都是 past-only (pandas 默认), 无 lookahead.
    末尾 shift(1) 把语义变成 D-1 值.
    """

    # 优先 qfq 列, 缺失 (合成 panel 或老数据) fallback 到 raw — 保证向后兼容
    def _maybe_qfq(col: str) -> pd.Series:
        qfq_col = f"{col}_qfq"
        if qfq_col in g.columns:
            return cast(pd.Series, g[qfq_col].astype("float64"))
        return cast(pd.Series, g[col].astype("float64"))

    close = _maybe_qfq("close")
    high = _maybe_qfq("high")
    low = _maybe_qfq("low")
    return {
        "rsi12": _rsi(close, 12) / 100 - 0.5,
        "macd_hist_norm": _macd_hist(close) / close,
        "kdj_j": _kdj_j(high, low, close) / 100 - 0.5,
        "atr14_pct": _atr(high, low, close, 14) / close,
    }


def _feat_fundamental(
    g: pd.DataFrame,
    _: _DerivedCols,
) -> _DerivedCols:
    """基本面 4 维 (PE / PB / turnover / market_cap).

    手动推导:
        pe_log[D]     = log1p(max(pe_ttm[D], 0))
        pb_log[D]     = log1p(max(pb[D], 0))
        turnover[D]   = turnover_rate[D]
        mv_log[D]     = log(max(total_mv[D], 1))

    daily_basic 表的 pe_ttm / pb / turnover_rate / total_mv 都是 **D 日收盘后**
    交易所计算公布的值. shift(1) 后 D 日特征 = D-1 日公告值, 是 D 日开盘前可见的.

    clip(lower=0) / clip(lower=1): 防负 PE (亏损公司 NaN/<0) 和 0 市值崩溃.
    行业指数 panel 里 daily_basic 全 NaN, cross_sectional_zscore 后变 0.
    """
    return {
        "pe_log": np.log1p(g["pe_ttm"].clip(lower=0)),
        "pb_log": np.log1p(g["pb"].clip(lower=0)),
        "turnover": g["turnover_rate"],
        "mv_log": np.log(g["total_mv"].clip(lower=1)),
    }


def _feat_money_flow(
    g: pd.DataFrame,
    _: _DerivedCols,
) -> _DerivedCols:
    """资金流 2 维 (主力净流入率 / 大单买盘占比).

    手动推导:
        inflow_pct[D] = net_mf_amount[D] / total_mv[D]
                      → moneyflow 表的 D 日净流入 / D 日市值, 收盘后可见
        big_ratio[D]  = (buy_lg + buy_elg) / (buy_lg + buy_elg + buy_sm + buy_md)
                      → 大单买盘占全部买盘比, shift 后 D-1 已知

    所有 fillna(0) 把缺失买盘视为 0 (小市值股未必有大单数据). replace(0, NaN)
    防 0 除 → NaN → cross_sectional_zscore 后变 0.
    """
    big = cast(pd.Series, g["buy_lg_amount"]).fillna(0) + cast(
        pd.Series, g["buy_elg_amount"]
    ).fillna(0)
    total_buy = (
        big
        + cast(pd.Series, g["buy_sm_amount"]).fillna(0)
        + cast(pd.Series, g["buy_md_amount"]).fillna(0)
    )
    return {
        "inflow_pct": g["net_mf_amount"] / g["total_mv"].replace(0, np.nan),
        "big_ratio": big / total_buy.replace(0, np.nan),
    }


def _feat_limit_up_history(
    g: pd.DataFrame,
    derived: _DerivedCols,
) -> _DerivedCols:
    """涨停板历史 6 维 — 抓涨停任务核心特征.

    手动推导:
        is_limit[D]          = (ret_1d[D] >= log(1.095))          (D 日是否涨停)
        days_since_limit[D]  = (D 日索引 - 最近一次 is_limit=1 的索引) / 60, cap 60
                             → 距上次涨停 k 天 (最多 60 天前), shift 后 D-1 的 k
        n_limits_20d[D]      = is_limit.rolling(20).sum() / 20    → 近 20 天涨停频率
        n_limits_60d[D]      = is_limit.rolling(60).sum() / 60    → 近 60 天涨停频率
        max_ret_20d[D]       = max(ret_1d[D-19..D])               → 20 天最大单日涨幅
        consecutive_up[D]    = 连续上涨天数 (ret_1d > 0), cap 10, 归一 → [0,1]
        is_yi_zi[D]          = (open[D] == high[D] == low[D] == close[D]) 且涨停
                             → D 日是否一字板 (T 字/破板不算), shift 后 = D-1 一字板

    关键: days_since_limit 用 ffill 过去信息填当前, **pandas ffill 默认不跨越边界**,
    是 past-only 填充. streak_count 用 groupby cumsum 也只沿时间正向累加.
    全部 rolling / ffill / groupby(..).cumsum() 都是因果的, 末尾 shift(1) 后 D 日
    特征 = D-1 涨停结构, 严格无前视.

    特殊: `(days_idx * is_limit).replace(0, NaN).ffill()` — 当 days_idx[0]=0 且
    is_limit[0]=1 时会误丢第一个涨停信号, 但第 0 行本来就无 lookback 意义, 可忽略.
    """
    close = cast(pd.Series, g["close"].astype("float64"))
    open_ = cast(pd.Series, g["open"].astype("float64"))
    high = cast(pd.Series, g["high"].astype("float64"))
    low = cast(pd.Series, g["low"].astype("float64"))
    ret_1d = derived["ret_1d"]

    is_limit = (ret_1d >= np.log(1 + _LIMIT_UP_THRESHOLD)).astype(np.float64)
    days_idx = pd.Series(np.arange(len(g)), index=g.index, dtype=np.float64)
    # BUGFIX (2026-04-21): 原用 (days_idx * is_limit).replace(0, nan).ffill(), 当 Day 0
    # 涨停时 0*1=0 被 replace 吞掉, 整条 days_since_limit 被污染到下次涨停.
    # 改用 where(is_limit, ...) 避免 0 做 sentinel.
    last_limit_idx = days_idx.where(is_limit.astype(bool), np.nan).ffill()
    days_since = (days_idx - last_limit_idx).fillna(60).clip(upper=60) / 60.0

    up = (ret_1d > 0).astype(int)
    streak_id = (up != up.shift()).cumsum()
    streak_count = up.groupby(streak_id).cumsum() * up

    yi_zi = (
        (open_ == high)
        & (high == low)
        & (low == close)
        & (ret_1d >= np.log(1 + _LIMIT_UP_THRESHOLD))
    ).astype(np.float64)

    return {
        "days_since_limit": days_since,
        "n_limits_20d": is_limit.rolling(20, min_periods=1).sum() / 20.0,
        "n_limits_60d": is_limit.rolling(60, min_periods=1).sum() / 60.0,
        "max_ret_20d": ret_1d.rolling(20, min_periods=1).max(),
        "consecutive_up": streak_count.clip(upper=10) / 10.0,
        "is_yi_zi": yi_zi,
    }


def _feat_scale_turnover_vola(
    g: pd.DataFrame,
    derived: _DerivedCols,
) -> _DerivedCols:
    """规模 / 换手 / 波动 多尺度 6 维 (2024 升级).

    手动推导:
        circ_mv_log[D]         = log(circ_mv[D])                  (log 流通市值)
        circ_mv_z_self60[D]    = z-score(circ_mv[D] vs 60 天均值)
        turnover_ma20[D]       = mean(turnover_rate[D-19..D])
        turnover_self_z60[D]   = z-score(turnover_rate[D] vs 60 天均值)
        vola_60d[D]            = std(ret_1d[D-59..D])
        vola_ratio_20_60[D]    = std(ret_1d[D-19..D]) / vola_60d[D]

    所有 self-z 都是 **相对自身历史窗口**, 不涉及未来. 行业指数 panel 没 circ_mv,
    fallback 给 NaN, cross_sectional_zscore 后变 0 (行业 token 的该维度信号 0).
    shift(1) 后 D 日特征 = D-1 值.
    """
    circ_mv = (
        cast(pd.Series, g["circ_mv"]).astype("float64")
        if "circ_mv" in g.columns
        else pd.Series(np.nan, index=g.index)
    )
    turnover_rate = cast(pd.Series, g["turnover_rate"]).astype("float64")
    ret_1d = derived["ret_1d"]

    cmv_mean60 = circ_mv.rolling(60, min_periods=20).mean()
    cmv_std60 = circ_mv.rolling(60, min_periods=20).std(ddof=0)
    tr_mean60 = turnover_rate.rolling(60, min_periods=20).mean()
    tr_std60 = turnover_rate.rolling(60, min_periods=20).std(ddof=0)
    vola_60d = ret_1d.rolling(60, min_periods=20).std(ddof=0)
    vola_20d = ret_1d.rolling(20, min_periods=5).std(ddof=0)

    return {
        "circ_mv_log": np.log(circ_mv.clip(lower=1)),
        "circ_mv_z_self60": (circ_mv - cmv_mean60) / (cmv_std60 + 1e-9),
        "turnover_ma20": turnover_rate.rolling(20, min_periods=5).mean(),
        "turnover_self_z60": (turnover_rate - tr_mean60) / (tr_std60 + 1e-9),
        "vola_60d": vola_60d,
        "vola_ratio_20_60": vola_20d / (vola_60d + 1e-9),
    }


def _feat_price_volume_match(
    g: pd.DataFrame,
    derived: _DerivedCols,
) -> _DerivedCols:
    """价量匹配 3 维 — 联合价量信号 (单独的量和价特征模型能学, 但短窗联合关系抢跑).

    手动推导:
        price_vol_match5[D]  = ret_5d[D] × tanh(vol_ratio_5[D] - 1)
                               tanh 把 [0, +∞) 映到 (-1, 1), 防极端放量爆值
                             → 涨且放量 → 正大; 涨且缩量 → 负 (背离);
                             → 跌且放量 → 负大 (放量下跌); 跌且缩量 → 正小.
        inflow_5d_sum[D]     = sum(net_mf_amount[D-4..D]) / circ_mv[D]
                             → 大单净流 5 日累计 / 流通市值 (主力建仓连续性)
                               单日 inflow_pct 噪音大, 累计更稳; circ_mv 比 total_mv 真实
        vol_ret_corr60[D]    = corr(ret_1d[D-59..D], volume[D-59..D])
                             → 60 日价量相关, 与 vol_ret_corr20 配长短两尺度

    依赖已计算: ret_5d, vol_ratio_5 (price_derived & volume 组). 末尾 shift(1) 后
    D 日特征 = D-1 的窗口相关/累计.
    """
    volume = cast(pd.Series, g["volume"].astype("float64"))
    ret_1d = derived["ret_1d"]
    ret_5d = derived["ret_5d"]
    vol_ratio_5 = derived["vol_ratio_5"]
    circ_mv = (
        cast(pd.Series, g["circ_mv"]).astype("float64") if "circ_mv" in g.columns else g["total_mv"]
    )

    inflow_5d = cast(pd.Series, g["net_mf_amount"]).rolling(5, min_periods=2).sum()

    return {
        "price_vol_match5": ret_5d * np.tanh(vol_ratio_5 - 1),
        "inflow_5d_sum": inflow_5d / circ_mv.replace(0, np.nan),
        "vol_ret_corr60": ret_1d.rolling(60, min_periods=20).corr(volume),
    }


# =============================================================================
# 注册表 — 维护 "执行顺序 + 列名声明".
# 加新特征: 写新的 _feat_* helper + 在 FEATURE_GROUPS 末尾加一行.
# 不要在这里用字符串魔法 name, FEATURE_COLS 直接从 tuple 推导.
# =============================================================================


# =============================================================================
# CCTV 新闻联播 source — shift_days=0 的第一个特征源
# =============================================================================


@dataclass(frozen=True, slots=True)
class CctvNewsConfig:
    """CCTV 新闻联播特征的 DB 来源配置.

    max_mention_date: 只读 trade_date <= 此值的 mentions. None = 不限制.
        用途: audit_causality 截短 panel 时需要同时截短 DB 侧, 避免 news feature
        看到"未来"的 mention (DB 是时点快照, 截短 panel 不自动截短 DB).
    """

    db_path: str = "data/floatshare.db"
    table: str = "cctv_news_mentions"
    max_mention_date: str | None = None


class _CctvNewsSource(FeatureSource):
    """新闻联播行业提及 flag — T/T-1 两个二元特征.

    时间语义:
        availability_cutoff_hour = 20 (联播 19:30 结束 + tushare 入库 + 本地 NLP)
        shift_days = 0 → feats[D, "news_mentioned_t"] = D 日新闻提及 flag

    数据来源:
        cctv_news_mentions(trade_date, l1_code, mentioned) — ingest 阶段填充.
        表不存在或空 → 全 0 (graceful 兼容, 未 ingest 前模型训练不炸).

    映射逻辑:
        panel 一次处理一只股 → 查该股 SW L1 行业 → 按 trade_date 从 mentions 表对齐.
        指数 token (token_id 如 '801770.SI') 直接映射到自身 l1_code.
    """

    def __init__(self, cfg: CctvNewsConfig | None = None) -> None:
        self._cfg = cfg or CctvNewsConfig()

    @property
    def name(self) -> str:
        return "cctv_news"

    @property
    def output_cols(self) -> tuple[str, ...]:
        return ("news_mentioned_t", "news_mentioned_t1")

    @property
    def availability_cutoff_hour(self) -> int:
        return 20

    @property
    def quality_rules(self) -> dict[str, ColumnQualityRule]:
        # 二元 flag, 仅 0/1. NaN 不应出现 (ingest 失败时也填 0).
        return {
            col: ColumnQualityRule(
                nan_fill="zero",
                expected_min=0.0,
                expected_max=1.0,
                nan_alert_threshold=0.0,  # 一有 NaN 就 raise
                warmup_days=1,  # news_mentioned_t1 首日 NaN → shift 后 fillna(0)
            )
            for col in self.output_cols
        }

    def compute(self, panel: pd.DataFrame, derived: _DerivedCols) -> _DerivedCols:
        del derived
        # compute_features 约定: _features_per_code 进来时 panel 保留 code 列.
        # 从 panel["code"] 拿 code, 再查 DB 获取该股 l1 行业 + 按日期对齐 mention.
        code = str(panel["code"].iloc[0]) if "code" in panel.columns else ""
        mention = _load_cctv_mentions_for_code(
            self._cfg.db_path,
            self._cfg.table,
            code,
            panel.index,
            max_mention_date=self._cfg.max_mention_date,
        )
        mention_f = mention.astype(np.float64)
        return {
            "news_mentioned_t": mention_f,
            "news_mentioned_t1": mention_f.shift(1).fillna(0.0),
        }


def _load_cctv_mentions_for_code(
    db_path: str,
    table: str,
    code: str,
    trade_dates: pd.Index,
    max_mention_date: str | None = None,
) -> pd.Series:
    """查该股所属 L1 行业在 trade_dates 每一天是否被新闻联播提及.

    指数 token ('801770.SI' 等): l1_code = token 自身.
    股票 token ('300308.SZ' 等): 从 industry 表查 l1_code.
    查不到 / 表不存在 / DB 打不开 → 返回全 0 series (兼容 graceful).

    max_mention_date: 若非 None, SQL 额外加 `AND trade_date <= max_mention_date`
        用于因果性测试 (截短 panel 时同步截短 DB 时点, 防 news 看到"未来").
    """
    n = len(trade_dates)
    zeros = pd.Series(np.zeros(n, dtype=np.float64), index=trade_dates)
    if not code:
        return zeros
    try:
        with sqlite3.connect(db_path) as conn:
            l1 = _resolve_l1_code(conn, code)
            if l1 is None:
                return zeros
            # 批量查 l1 在这批日期的 mention
            start = pd.Timestamp(trade_dates.min()).strftime("%Y-%m-%d")
            end = pd.Timestamp(trade_dates.max()).strftime("%Y-%m-%d")
            if max_mention_date is not None:
                end = min(end, max_mention_date)  # 截短到 max_mention_date
            if start > end:
                return zeros
            cur = conn.execute(
                f"SELECT trade_date, mentioned FROM {table} "
                f"WHERE l1_code = ? AND trade_date BETWEEN ? AND ?",
                (l1, start, end),
            )
            mention_map: dict[pd.Timestamp, int] = {
                pd.Timestamp(r[0]): int(r[1]) for r in cur.fetchall()
            }
    except sqlite3.Error:
        return zeros

    values = np.array(
        [mention_map.get(pd.Timestamp(d), 0) for d in trade_dates],
        dtype=np.float64,
    )
    return pd.Series(values, index=trade_dates)


def _resolve_l1_code(conn: sqlite3.Connection, code: str) -> str | None:
    """code → SW L1 行业代码. 指数 token 直接返回自身; 股票查 industry 表."""
    if code.endswith(".SI"):
        return code
    try:
        cur = conn.execute(
            "SELECT l1_code FROM industry WHERE code = ?",
            (code,),
        )
        row = cur.fetchone()
        return row[0] if row else None
    except sqlite3.Error:
        return None


# =============================================================================
# Feature sources registry — 执行顺序 = FEATURE_SOURCES 顺序
# =============================================================================


# 37 维 OHLCV/daily_basic/moneyflow 派生特征的 quality rules 数据.
# 范围: A 股主板经验值 (±10% 涨跌停 → log ±0.11). warmup = rolling window + 1 (shift).
# fill 策略: 默认 zero (cross_sectional_zscore 后也是 0 等价中性).
_RULE_RET_1D = ColumnQualityRule(
    nan_fill="zero", expected_min=-0.11, expected_max=0.11, warmup_days=2
)
_RULE_CORR = ColumnQualityRule(nan_fill="zero", expected_min=-1.0, expected_max=1.0, warmup_days=21)
_RULE_UNIT = ColumnQualityRule(nan_fill="zero", expected_min=0.0, expected_max=1.0, warmup_days=1)

_PRICE_RULES: dict[str, ColumnQualityRule] = {
    "ret_1d": _RULE_RET_1D,
    "ret_5d": ColumnQualityRule(
        nan_fill="zero", expected_min=-0.5, expected_max=0.5, warmup_days=6
    ),
    "ret_20d": ColumnQualityRule(
        nan_fill="zero", expected_min=-1.0, expected_max=1.0, warmup_days=21
    ),
    "range_pct": ColumnQualityRule(
        nan_fill="zero", expected_min=0.0, expected_max=0.25, warmup_days=1
    ),
    "gap": ColumnQualityRule(nan_fill="zero", expected_min=-0.11, expected_max=0.11, warmup_days=2),
}

_MA_RULES: dict[str, ColumnQualityRule] = {
    "ma5_dev": ColumnQualityRule(
        nan_fill="zero", expected_min=-0.5, expected_max=0.5, warmup_days=6
    ),
    "ma20_dev": ColumnQualityRule(
        nan_fill="zero", expected_min=-0.5, expected_max=0.5, warmup_days=21
    ),
    "ma60_dev": ColumnQualityRule(
        nan_fill="zero", expected_min=-0.5, expected_max=0.5, warmup_days=61
    ),
    "ma_short_long": ColumnQualityRule(
        nan_fill="zero", expected_min=-0.3, expected_max=0.3, warmup_days=21
    ),
}

_VOLUME_RULES: dict[str, ColumnQualityRule] = {
    "vol_z20": ColumnQualityRule(
        nan_fill="zero", expected_min=-5.0, expected_max=30.0, warmup_days=21
    ),
    "vol_ratio_5": ColumnQualityRule(
        nan_fill="zero", expected_min=0.0, expected_max=15.0, warmup_days=6
    ),
    "vol_ret_corr20": _RULE_CORR,
}

_TECHNICAL_RULES: dict[str, ColumnQualityRule] = {
    # rsi12: 原 RSI [0,100] → /100 - 0.5 = [-0.5, 0.5] (n=12 对齐 tushare)
    "rsi12": ColumnQualityRule(
        nan_fill="zero", expected_min=-0.5, expected_max=0.5, warmup_days=13
    ),
    "macd_hist_norm": ColumnQualityRule(
        nan_fill="zero", expected_min=-0.1, expected_max=0.1, warmup_days=27
    ),
    # kdj_j: 3K-2D, K/D ∈ [0,100] 但 J 可越界, /100 - 0.5 → 可到 ±1.5
    "kdj_j": ColumnQualityRule(
        nan_fill="zero", expected_min=-1.5, expected_max=1.5, warmup_days=10
    ),
    "atr14_pct": ColumnQualityRule(
        nan_fill="zero", expected_min=0.0, expected_max=0.25, warmup_days=15
    ),
}

_FUNDAMENTAL_RULES: dict[str, ColumnQualityRule] = {
    # pe_log = log1p(pe_ttm.clip(lower=0)); pe_ttm 亏损 NaN → log1p(NaN)=NaN, 停牌日 NaN 常见
    "pe_log": ColumnQualityRule(
        nan_fill="zero",
        expected_min=0.0,
        expected_max=10.0,
        warmup_days=1,
        nan_alert_threshold=0.35,  # A 股亏损公司 25-30% 常态 (pe_ttm <0 → NaN)
    ),
    "pb_log": ColumnQualityRule(
        nan_fill="zero",
        expected_min=0.0,
        expected_max=5.0,
        warmup_days=1,
        nan_alert_threshold=0.30,
    ),
    # turnover = turnover_rate %, 极端换手可达 50%+
    "turnover": ColumnQualityRule(
        nan_fill="zero", expected_min=0.0, expected_max=50.0, warmup_days=1
    ),
    # mv_log = log(total_mv 万元), 市值 1 亿 = log(1e4) ≈ 9.2, 万亿 = log(1e8) = 18.4
    "mv_log": ColumnQualityRule(
        nan_fill="zero", expected_min=9.0, expected_max=20.0, warmup_days=1
    ),
}

_MONEY_FLOW_RULES: dict[str, ColumnQualityRule] = {
    "inflow_pct": ColumnQualityRule(
        nan_fill="zero",
        expected_min=-0.1,
        expected_max=0.1,
        warmup_days=1,
        nan_alert_threshold=0.15,
    ),
    "big_ratio": ColumnQualityRule(
        nan_fill="zero",
        expected_min=0.0,
        expected_max=1.0,
        warmup_days=1,
        nan_alert_threshold=0.15,
    ),
}

_LIMIT_UP_RULES: dict[str, ColumnQualityRule] = {
    "days_since_limit": _RULE_UNIT,
    "n_limits_20d": _RULE_UNIT,
    "n_limits_60d": _RULE_UNIT,
    "max_ret_20d": ColumnQualityRule(
        nan_fill="zero", expected_min=-0.11, expected_max=0.15, warmup_days=2
    ),
    "consecutive_up": _RULE_UNIT,
    "is_yi_zi": _RULE_UNIT,
}

_SCALE_RULES: dict[str, ColumnQualityRule] = {
    # circ_mv_log: 流通市值 万元 → log
    "circ_mv_log": ColumnQualityRule(
        nan_fill="zero", expected_min=9.0, expected_max=20.0, warmup_days=1
    ),
    "circ_mv_z_self60": ColumnQualityRule(
        nan_fill="zero", expected_min=-5.0, expected_max=5.0, warmup_days=61
    ),
    "turnover_ma20": ColumnQualityRule(
        nan_fill="zero", expected_min=0.0, expected_max=30.0, warmup_days=6
    ),
    "turnover_self_z60": ColumnQualityRule(
        nan_fill="zero", expected_min=-5.0, expected_max=5.0, warmup_days=61
    ),
    # vola_60d: ret_1d std, 极端小 0.005, 妖股可达 0.08+
    "vola_60d": ColumnQualityRule(
        nan_fill="zero", expected_min=0.0, expected_max=0.1, warmup_days=61
    ),
    "vola_ratio_20_60": ColumnQualityRule(
        nan_fill="zero", expected_min=0.1, expected_max=5.0, warmup_days=61
    ),
}

_PV_MATCH_RULES: dict[str, ColumnQualityRule] = {
    "price_vol_match5": ColumnQualityRule(
        nan_fill="zero", expected_min=-0.5, expected_max=0.5, warmup_days=6
    ),
    "inflow_5d_sum": ColumnQualityRule(
        nan_fill="zero",
        expected_min=-0.5,
        expected_max=0.5,
        warmup_days=6,
        nan_alert_threshold=0.15,
    ),
    "vol_ret_corr60": ColumnQualityRule(
        nan_fill="zero",
        expected_min=-1.0,
        expected_max=1.0,
        warmup_days=61,
    ),
}


FEATURE_SOURCES: tuple[FeatureSource, ...] = (
    _DailyDerivedSource(
        "price_derived",
        ("ret_1d", "ret_5d", "ret_20d", "range_pct", "gap"),
        _feat_price_derived,
        _rules=_PRICE_RULES,
    ),
    _DailyDerivedSource(
        "ma_dev",
        ("ma5_dev", "ma20_dev", "ma60_dev", "ma_short_long"),
        _feat_ma_deviation,
        _rules=_MA_RULES,
    ),
    _DailyDerivedSource(
        "volume",
        ("vol_z20", "vol_ratio_5", "vol_ret_corr20"),
        _feat_volume,
        _rules=_VOLUME_RULES,
    ),
    _DailyDerivedSource(
        "technical",
        ("rsi12", "macd_hist_norm", "kdj_j", "atr14_pct"),
        _feat_technical,
        _rules=_TECHNICAL_RULES,
    ),
    _DailyDerivedSource(
        "fundamental",
        ("pe_log", "pb_log", "turnover", "mv_log"),
        _feat_fundamental,
        _rules=_FUNDAMENTAL_RULES,
    ),
    _DailyDerivedSource(
        "money_flow",
        ("inflow_pct", "big_ratio"),
        _feat_money_flow,
        _rules=_MONEY_FLOW_RULES,
    ),
    _DailyDerivedSource(
        "limit_up_history",
        (
            "days_since_limit",
            "n_limits_20d",
            "n_limits_60d",
            "max_ret_20d",
            "consecutive_up",
            "is_yi_zi",
        ),
        _feat_limit_up_history,
        _rules=_LIMIT_UP_RULES,
    ),
    _DailyDerivedSource(
        "scale_turnover_vola",
        (
            "circ_mv_log",
            "circ_mv_z_self60",
            "turnover_ma20",
            "turnover_self_z60",
            "vola_60d",
            "vola_ratio_20_60",
        ),
        _feat_scale_turnover_vola,
        _rules=_SCALE_RULES,
    ),
    _DailyDerivedSource(
        "price_volume_match",
        ("price_vol_match5", "inflow_5d_sum", "vol_ret_corr60"),
        _feat_price_volume_match,
        _rules=_PV_MATCH_RULES,
    ),
    _CctvNewsSource(),  # shift_days=0 — 第一个 T 日可见 source
)

# 兼容旧 API: FEATURE_GROUPS 保留为 (helper_fn, cols) 元组列表, 指向 _DailyDerivedSource
# 的 _helper + _cols. CCTV source 是 ABC 子类不是 helper fn, 不在此列表.
FEATURE_GROUPS: tuple[tuple[_FeatureHelper, tuple[str, ...]], ...] = tuple(
    (s._helper, s.output_cols) for s in FEATURE_SOURCES if isinstance(s, _DailyDerivedSource)
)

FEATURE_COLS: tuple[str, ...] = tuple(c for s in FEATURE_SOURCES for c in s.output_cols)
N_FEATURES = len(FEATURE_COLS)
assert N_FEATURES == 39, f"特征维度应为 39 (37 旧 + 2 新闻), 实际 {N_FEATURES}"


# =============================================================================
# Public API
# =============================================================================


def compute_features(
    panel: pd.DataFrame,
    sources: tuple[FeatureSource, ...] | None = None,
) -> pd.DataFrame:
    """对 (code, trade_date, ...) panel 计算所有 sources 声明的特征.

    Args:
        panel:   load_panel 返回, 含 OHLCV + daily_basic + moneyflow 原始列
        sources: FeatureSource 列表. None → FEATURE_SOURCES (模块默认).
                 audit_causality 等会传入带 CctvNewsConfig.max_mention_date 的变体.

    Returns:
        DataFrame 含列 code + trade_date + FEATURE_COLS. Per-source shift_days 保证
        每个 feature 对应 "T 日 22:30 决策时可见" 的信息.
    """
    srcs = sources if sources is not None else FEATURE_SOURCES
    if panel.empty:
        return pd.DataFrame(columns=("code", "trade_date", *FEATURE_COLS))
    panel = panel.sort_values(["code", "trade_date"])
    # 手动 groupby (不用 apply + include_groups=False) — 保留 code 列给 _CctvNewsSource
    results: list[pd.DataFrame] = []
    for code, group in panel.groupby("code", sort=False):
        result = _features_per_code(group, srcs)
        result = result.reset_index()
        result["code"] = code
        results.append(result)
    out = pd.concat(results, ignore_index=True)
    return cast(pd.DataFrame, out[["code", "trade_date", *FEATURE_COLS]])


def _features_per_code(
    g: pd.DataFrame,
    sources: tuple[FeatureSource, ...] = FEATURE_SOURCES,
) -> pd.DataFrame:
    """单只股的时序特征 — 按 sources 遍历, per-source 自带 shift_days.

    不再末尾统一 shift(1):
        - shift_days=1 的 source: compute 结果 shift(1) (等价旧行为)
        - shift_days=0 的 source: compute 结果直接用 (T 日可见特征)

    g 含 `code` 列 (未 drop) — _DailyDerivedSource helpers 只读 OHLCV 等不受影响,
    _CctvNewsSource 从 `g["code"].iloc[0]` 拿 code 查 DB.
    """
    g = g.set_index("trade_date").sort_index()
    derived: _DerivedCols = {}
    out_cols: dict[str, pd.Series] = {}
    for source in sources:
        new_cols = source.compute(g, derived)
        _check_coverage(new_cols, source.output_cols)
        derived.update(new_cols)
        # Per-source shift: shift_days=1 → shift(1); shift_days=0 → 不动
        if source.shift_days > 0:
            for col in source.output_cols:
                out_cols[col] = new_cols[col].shift(source.shift_days)
        else:
            for col in source.output_cols:
                out_cols[col] = new_cols[col]
    return pd.DataFrame({c: out_cols[c] for c in FEATURE_COLS}, index=g.index)


def _check_coverage(produced: dict[str, pd.Series], declared: tuple[str, ...]) -> None:
    """helper 实际产出的列必须与 FEATURE_SOURCES 声明一致 (fail-fast 防漏改)."""
    missing = set(declared) - set(produced)
    extra = set(produced) - set(declared)
    if missing or extra:
        raise AssertionError(
            f"feature source 产出与声明不匹配: missing={missing}, extra={extra}",
        )


# =============================================================================
# Low-level 指标 (pure pandas, 不依赖 ta 库)
# =============================================================================


def _rsi(close: pd.Series, n: int = 12) -> pd.Series:
    """RSI — Wilder 平滑 (EWMA alpha=1/n), 默认 n=12 跟 tushare rsi_12 拉齐. past-only.

    BUGFIX (2026-04-21): 原 `fillna(50)` 让纯上涨 (loss=0, gain>0) 落在 50 (中性),
    业界标准应为 100 (极端超买). 现在区分:
        loss=0 & gain>0  →  RSI = 100 (纯涨 → 超买)
        loss=0 & gain=0  →  RSI = 50  (warmup 初期, 无信号)
        loss>0            →  正常公式
    """
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.where(loss > 0, np.nan)
    rsi = cast(pd.Series, 100 - 100 / (1 + rs))
    # loss=0 且 gain>0 → 纯涨 → 100
    pure_up = (loss == 0) & (gain > 0)
    rsi = rsi.where(~pure_up, 100.0)
    # 剩余 NaN (loss=0 & gain=0, 即 warmup 首值) → 50
    return rsi.fillna(50)


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD 柱 = 2 × (DIF - DEA). 全 EWMA past-only."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    return (dif - dea) * 2


def _kdj_j(
    high: pd.Series, low: pd.Series, close: pd.Series, n: int = 9, m1: int = 3, m2: int = 3
) -> pd.Series:
    """KDJ J 线 = 3K - 2D. RSV 用 n 日滚动 high/low (past-only), K/D 用 EWMA 平滑."""
    low_min = low.rolling(n).min()
    high_max = high.rolling(n).max()
    rsv = (close - low_min) / (high_max - low_min).replace(0, np.nan) * 100
    k = rsv.ewm(alpha=1 / m1, adjust=False).mean()
    d = k.ewm(alpha=1 / m2, adjust=False).mean()
    return 3 * k - 2 * d


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """ATR — True Range 的 Wilder EWMA. 用当日 H/L 和 D-1 close, 无前视."""
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(
        axis=1
    )
    return cast(pd.Series, tr.ewm(alpha=1 / n, adjust=False).mean())
