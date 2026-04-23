"""Feature 质量 audit — 后验因子排查 (pipeline stage 2, T 日 22:00, 30min 预算).

跑当日 feats 的三道检查:

    1. quality_rules 检查:
        - NaN 比例 > threshold (warm-up 期外)      → alert
        - 值越界 expected_min/max                   → alert / clip
        - NaN 替换策略 (zero/prev/median)           → 实际执行替换
    2. 手算 spot-check:
        - 随机 n_samples 只股 × 关键 feature        → 重算 ground truth
        - 匹配 atol=1e-10 (EWM 类 1e-6)             → 不匹配 → error
    3. 输出 AuditReport:
        - alerts list + passed list + summary
        - 写 logs/audit-YYYY-MM-DD.json
        - has_errors() 时通过 observability.notify 告警

设计约束: 真 DB 驱动 (不是 mock). 输入 feats (compute_features 输出), 输出 report.
所在层: ml 层 (跟 FEATURE_SOURCES / ColumnQualityRule 同层, 避免 application→ml 反向依赖).
Caller: application/cli 层的 pipeline orchestrator 在 22:00 调用此模块.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from floatshare.ml.audit_manual import (
    LIMIT_UP_LOG,
    manual_atr,
    manual_consecutive_up,
    manual_days_since_limit,
    manual_is_limit,
    manual_kdj_j,
    manual_macd_hist,
    manual_rsi,
    rolling_corr_np,
    rolling_max_np,
    rolling_mean_np,
    rolling_std_np,
    rolling_sum_np,
)
from floatshare.ml.features import (
    FEATURE_COLS,
    FEATURE_SOURCES,
    CctvNewsConfig,
    ColumnQualityRule,
    FeatureSource,
    _CctvNewsSource,
    compute_features,
)
from floatshare.observability import logger

# EWM-based features 的数值误差略大 (递推累积), 用更宽 atol
_EWM_FEATURES: frozenset[str] = frozenset(
    {
        "rsi12",
        "macd_hist_norm",
        "kdj_j",
        "atr14_pct",
    }
)
# CCTV 新闻 feature 不做 spot-check (值来自同一 DB 表, 自己验自己无意义)
_SKIP_SPOT_CHECK: frozenset[str] = frozenset({"news_mentioned_t", "news_mentioned_t1"})

# 滚动阈值默认参数
_ROLLING_WINDOW_DAYS = 252
_ROLLING_LOW_Q = 0.005
_ROLLING_HIGH_Q = 0.995
_ROLLING_MIN_SAMPLES = 1000  # 样本少于此值 → fallback 到 rule 硬阈值


class AuditFailedError(RuntimeError):
    """Stage 2 后验因子排查发现 error — feats 进不了训练/推理.

    携带 AuditReport 对象供调用方查看详情.
    """

    def __init__(self, report: AuditReport) -> None:
        super().__init__(report.summary())
        self.report = report


@dataclass(frozen=True, slots=True)
class AuditAlert:
    """一条 audit 告警 (全部视作 error, 不再区分 severity)."""

    feature: str
    source: str
    issue: str  # 'nan_rate_high' / 'below_min' / 'above_max' / 'manual_mismatch' / ...
    value: float | None = None  # 实际观测值
    threshold: float | None = None  # 规则阈值 (可能是滚动 p99.5)
    affected_codes: int = 0  # 多少只股触发
    details: str = ""


@dataclass(slots=True)
class AuditReport:
    trade_date: str
    n_codes: int
    n_features_checked: int
    alerts: list[AuditAlert] = field(default_factory=list)
    passed_features: list[str] = field(default_factory=list)
    generated_at: str = ""

    def has_errors(self) -> bool:
        return len(self.alerts) > 0

    def summary(self) -> str:
        return (
            f"AuditReport T={self.trade_date} codes={self.n_codes} "
            f"features={self.n_features_checked} "
            f"passed={len(self.passed_features)} "
            f"errors={len(self.alerts)}"
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "trade_date": self.trade_date,
                "n_codes": self.n_codes,
                "n_features_checked": self.n_features_checked,
                "generated_at": self.generated_at,
                "has_errors": self.has_errors(),
                "alerts": [asdict(a) for a in self.alerts],
                "passed_features": self.passed_features,
            },
            ensure_ascii=False,
            indent=2,
        )


# =============================================================================
# 入口: run_feature_audit(feats, trade_date, panel=None)
# =============================================================================


def run_feature_audit(
    feats: pd.DataFrame,
    trade_date: str,
    panel: pd.DataFrame | None = None,
    spot_check_samples: int | None = None,
    apply_nan_fill: bool = False,
    audit_log_dir: str | Path = "logs/audit",
    raise_on_error: bool = True,
) -> tuple[AuditReport, pd.DataFrame]:
    """主入口. 走一遍质量检查, 若发现任何 alert 默认 raise AuditFailedError.

    Args:
        feats: compute_features 输出, 含 code + trade_date + FEATURE_COLS
        trade_date: 要审计的交易日 YYYY-MM-DD
        panel: 原始 panel (若提供, 跑 spot-check 手算对拍)
        spot_check_samples: 抽几只股; None = 全量对拍
        apply_nan_fill: True 则按 rule.nan_fill 策略替换 NaN (返回 feats_processed)
        audit_log_dir: AuditReport JSON 写到这里
        raise_on_error: True (default) → 有 alert 就 raise AuditFailedError. False → 仅返回 report.

    Returns:
        (AuditReport, feats_processed). feats_processed = 若 apply_nan_fill 则已替换.

    Raises:
        AuditFailedError: raise_on_error=True 且 report.has_errors().
    """
    today = feats[feats["trade_date"] == pd.Timestamp(trade_date)].copy()
    if today.empty:
        logger.warning(f"audit: feats 里找不到 trade_date={trade_date}")
        empty_report = AuditReport(
            trade_date=trade_date,
            n_codes=0,
            n_features_checked=0,
            generated_at=datetime.now().isoformat(),
        )
        return empty_report, feats

    report = AuditReport(
        trade_date=trade_date,
        n_codes=len(today),
        n_features_checked=0,
        generated_at=datetime.now().isoformat(),
    )

    # Step 1: quality rules (NaN + 值域, 滚动 252 天阈值)
    _audit_quality_rules(feats, today, trade_date, report)

    # Step 2: 手算 spot-check (全量默认)
    if panel is not None:
        _audit_spot_check_manual_calc(panel, today, trade_date, spot_check_samples, report)

    # Step 3: 如果要求, 执行 NaN 替换
    feats_out = feats.copy() if apply_nan_fill else feats
    if apply_nan_fill:
        feats_out = _apply_nan_fill_rules(feats_out, trade_date)

    # Step 4: 写 log + 决定是否 raise
    _write_audit_log(report, audit_log_dir)
    logger.info(report.summary())
    if raise_on_error and report.has_errors():
        raise AuditFailedError(report)
    return report, feats_out


# =============================================================================
# Step 1: quality_rules 检查
# =============================================================================


def _audit_quality_rules(
    feats: pd.DataFrame,
    today: pd.DataFrame,
    trade_date: str,
    report: AuditReport,
) -> None:
    """遍历每维 feature 做 NaN + 值域检查. 阈值优先用滚动 252 天 p0.5/p99.5, 样本不够用 rule 硬阈值."""
    rolling = _compute_rolling_thresholds(feats, trade_date)
    for source in FEATURE_SOURCES:
        for col, rule in source.quality_rules.items():
            report.n_features_checked += 1
            values = today[col].to_numpy(dtype=np.float64)
            nan_alerts = _check_nan(col, source.name, values, rule)
            range_alerts = _check_value_range(col, source.name, values, rule, rolling.get(col))
            alerts = nan_alerts + range_alerts
            if alerts:
                report.alerts.extend(alerts)
            else:
                report.passed_features.append(col)


def _compute_rolling_thresholds(
    feats: pd.DataFrame,
    trade_date: str,
    window: int = _ROLLING_WINDOW_DAYS,
    low_q: float = _ROLLING_LOW_Q,
    high_q: float = _ROLLING_HIGH_Q,
) -> dict[str, tuple[float, float]]:
    """实时算每维最近 `window` 交易日的 p{low_q}/p{high_q} 作为当日阈值.

    样本 < _ROLLING_MIN_SAMPLES 时不返回该 col, caller fallback 到 rule 硬阈值.
    """
    T = pd.Timestamp(trade_date)
    # 近似: 取 T 之前所有天, 挑最后 `window` 个 trade_date
    hist = feats[feats["trade_date"] <= T]
    last_dates = sorted(hist["trade_date"].unique())[-window:]
    hist = hist[hist["trade_date"].isin(last_dates)]
    thresholds: dict[str, tuple[float, float]] = {}
    for col in FEATURE_COLS:
        if col not in hist.columns:
            continue
        vals = hist[col].dropna().to_numpy(dtype=np.float64)
        if len(vals) < _ROLLING_MIN_SAMPLES:
            continue
        thresholds[col] = (float(np.quantile(vals, low_q)), float(np.quantile(vals, high_q)))
    return thresholds


def _check_nan(
    col: str,
    src: str,
    values: np.ndarray,
    rule: ColumnQualityRule,
) -> list[AuditAlert]:
    """懒检查: 只有真出现 NaN 才报错.

    - 有 NaN 且 rule.nan_fill=None → raise-worthy alert (开发者需跑 ablation 选策略)
    - 有 NaN 且 nan_rate > threshold → raise-worthy alert (超容忍度)
    - 有 NaN 但 rate OK 且有 fill 策略 → 通过 (step 3 apply_nan_fill 会填)
    """
    if len(values) == 0:
        return []
    nan_mask = np.isnan(values)
    n_nan = int(nan_mask.sum())
    if n_nan == 0:
        return []
    nan_rate = n_nan / len(values)
    if rule.nan_fill is None:
        return [
            AuditAlert(
                feature=col,
                source=src,
                issue="nan_strategy_missing",
                value=nan_rate,
                threshold=None,
                affected_codes=n_nan,
                details=(
                    f"出现 NaN {n_nan}/{len(values)} 但未配 nan_fill 策略. "
                    f"建议跑 `floatshare-nan-ablation --feature {col}` 对比 "
                    f"zero/prev/median/keep 四种策略的小模型 val AUC, 最佳策略回填 rule."
                ),
            )
        ]
    if nan_rate > rule.nan_alert_threshold:
        return [
            AuditAlert(
                feature=col,
                source=src,
                issue="nan_rate_high",
                value=nan_rate,
                threshold=rule.nan_alert_threshold,
                affected_codes=n_nan,
                details=f"NaN 比例 {nan_rate:.2%} 超过阈值 {rule.nan_alert_threshold:.2%}",
            )
        ]
    return []


def _check_value_range(
    col: str,
    src: str,
    values: np.ndarray,
    rule: ColumnQualityRule,
    rolling_range: tuple[float, float] | None,
) -> list[AuditAlert]:
    """值域检查 — 优先滚动阈值, fallback 到 rule 硬阈值.

    action 语义:
        - 'clip' (default): 市场极端值通过 winsorize 处理, 不产生 alert (日志 info 级别)
        - 'alert':          超限 → alert → raise AuditFailedError (要求人工介入)
        - 'drop_stock':     超限 → alert (告知 drop 了几只股)
    """
    non_nan = values[~np.isnan(values)]
    if len(non_nan) == 0:
        return []
    if rolling_range is not None:
        lo, hi = rolling_range
        src_label = f"{src}/rolling252"
    elif rule.expected_min is not None and rule.expected_max is not None:
        lo, hi = rule.expected_min, rule.expected_max
        src_label = f"{src}/fixed"
    else:
        return []  # 无阈值可比

    n_below = int((non_nan < lo).sum())
    n_above = int((non_nan > hi).sum())
    if n_below == 0 and n_above == 0:
        return []

    # clip 策略: 静默 winsorize, 不产生 alert (但打 info log 保留可见性)
    if rule.out_of_range_action == "clip":
        if n_below or n_above:
            logger.info(
                f"  winsorize {col} ({src_label}): {n_below} 只股 < {lo:.4f}, "
                f"{n_above} 只股 > {hi:.4f} (静默 clip, 防极端污染 zscore)",
            )
        return []

    alerts: list[AuditAlert] = []
    if n_below:
        alerts.append(
            AuditAlert(
                feature=col,
                source=src_label,
                issue="below_min",
                value=float(non_nan.min()),
                threshold=lo,
                affected_codes=n_below,
                details=f"{n_below} 只股低于 {lo:.4f}, min={float(non_nan.min()):.4f}",
            )
        )
    if n_above:
        alerts.append(
            AuditAlert(
                feature=col,
                source=src_label,
                issue="above_max",
                value=float(non_nan.max()),
                threshold=hi,
                affected_codes=n_above,
                details=f"{n_above} 只股高于 {hi:.4f}, max={float(non_nan.max()):.4f}",
            )
        )
    return alerts


# =============================================================================
# Step 2: 手算 spot-check
# =============================================================================


def _audit_spot_check_manual_calc(
    panel: pd.DataFrame,
    today: pd.DataFrame,
    trade_date: str,
    n_samples: int | None,
    report: AuditReport,
) -> None:
    """随机 n 只股, 对全 37 维 OHLCV/daily_basic 派生特征手算对拍 (news 跳过).

    feats[T] (shift_days=1) = helper@(T-1). 用 audit_manual 独立纯 numpy 实现重算,
    对拍 atol=1e-8 (EWM 特征 1e-5, 递推累积误差).

    不匹配 → AuditAlert, 指向 features.py 代码 bug 或数据污染.
    """
    if n_samples is None or n_samples >= len(today):
        codes = today["code"].tolist()  # 全量
    else:
        codes = today["code"].sample(n_samples, random_state=0).tolist()
    for code in codes:
        stock_panel = panel[panel["code"] == code].sort_values("trade_date").reset_index(drop=True)
        matching = stock_panel.index[stock_panel["trade_date"] == pd.Timestamp(trade_date)].tolist()
        if not matching:
            continue
        t_idx = matching[0]
        if t_idx < 61:  # 需 60 天 warmup 覆盖 ma60 / vola_60d / self_z60 等
            continue

        expected = _compute_expected_features(stock_panel, t_idx)
        actual_row = today[today["code"] == code]
        if actual_row.empty:
            continue
        actual = actual_row.iloc[0]
        for feat, exp in expected.items():
            if feat in _SKIP_SPOT_CHECK:
                continue
            act = float(actual[feat])
            if np.isnan(exp) or np.isnan(act):
                continue
            atol = 1e-5 if feat in _EWM_FEATURES else 1e-8
            if abs(act - exp) > atol:
                report.alerts.append(
                    AuditAlert(
                        feature=feat,
                        source="spot_check",
                        issue="manual_mismatch",
                        value=act,
                        threshold=exp,
                        affected_codes=1,
                        details=(
                            f"code={code}: feats={act:.8f} vs manual={exp:.8f} "
                            f"diff={act - exp:+.2e} (atol={atol:.0e})"
                        ),
                    )
                )


def _compute_expected_features(
    stock_panel: pd.DataFrame,
    t_idx: int,
) -> dict[str, float]:
    """重算 37 维 OHLCV/daily_basic 派生特征在 feats[T] = helper@(T-1) 位置的期望值.

    全部用 audit_manual 独立纯 numpy 实现, 不用 pandas rolling/ewm.

    返回 {feature_name: expected_value} (NaN 值也返回, caller 自己跳过).
    """

    # 截 0..t_idx 的 numpy 数组 (helper 用到 T-1 = t_idx-1 位置, 需要 0..t_idx 历史)
    def _col(name: str, dtype: Any = np.float64) -> np.ndarray:
        return stock_panel[name].to_numpy(dtype=dtype)[: t_idx + 1]

    close = _col("close")
    open_ = _col("open")
    high = _col("high")
    low = _col("low")
    volume = _col("volume")
    pe_ttm = _col("pe_ttm")
    pb = _col("pb")
    turnover_rate = _col("turnover_rate")
    total_mv = _col("total_mv")
    circ_mv = _col("circ_mv") if "circ_mv" in stock_panel.columns else total_mv
    net_mf = _col("net_mf_amount")
    buy_sm = _col("buy_sm_amount")
    buy_md = _col("buy_md_amount")
    buy_lg = _col("buy_lg_amount")
    buy_elg = _col("buy_elg_amount")

    # feats[T] = helper@(T-1). panel index T = t_idx, helper index = t_idx - 1
    s = t_idx - 1

    # 价格派生
    ret_1d = np.concatenate([[np.nan], np.log(close[1:] / close[:-1])])
    ret_5d = np.full(len(close), np.nan)
    ret_5d[5:] = np.log(close[5:] / close[:-5])
    ret_20d = np.full(len(close), np.nan)
    ret_20d[20:] = np.log(close[20:] / close[:-20])
    prev_close = np.concatenate([[np.nan], close[:-1]])

    # 均线
    ma5 = rolling_mean_np(close, 5, min_periods=5)
    ma20 = rolling_mean_np(close, 20, min_periods=20)
    ma60 = rolling_mean_np(close, 60, min_periods=60)

    # 量能
    vol_ma20 = rolling_mean_np(volume, 20, min_periods=20)
    vol_std20 = rolling_std_np(volume, 20, min_periods=20, ddof=0)
    vol_ma5 = rolling_mean_np(volume, 5, min_periods=5)
    vol_ratio_5_arr = volume / np.where(vol_ma5 == 0, np.nan, vol_ma5)

    # 技术指标 (EWM)
    rsi = manual_rsi(close, 12)
    macd_hist = manual_macd_hist(close)
    kdj = manual_kdj_j(high, low, close)
    atr = manual_atr(high, low, close, 14)

    # 基本面
    pe_log = np.log1p(np.maximum(pe_ttm, 0))
    pb_log = np.log1p(np.maximum(pb, 0))
    mv_log = np.log(np.maximum(total_mv, 1))

    # 资金流
    total_mv_safe = np.where(total_mv == 0, np.nan, total_mv)
    big = buy_lg + buy_elg
    total_buy = big + buy_sm + buy_md
    total_buy_safe = np.where(total_buy == 0, np.nan, total_buy)

    # 涨停史
    is_limit = manual_is_limit(close)
    days_since = manual_days_since_limit(is_limit)
    n_limits_20 = rolling_sum_np(is_limit, 20, min_periods=1) / 20.0
    n_limits_60 = rolling_sum_np(is_limit, 60, min_periods=1) / 60.0
    max_ret_20 = rolling_max_np(ret_1d, 20, min_periods=1)
    consec_up = manual_consecutive_up(close)
    yi_zi_bool = (open_ == high) & (high == low) & (low == close)
    limit_bool = np.where(np.isnan(ret_1d), False, ret_1d >= LIMIT_UP_LOG)
    yi_zi = (yi_zi_bool & limit_bool).astype(np.float64)

    # 规模 / 波动
    cmv_mean60 = rolling_mean_np(circ_mv, 60, min_periods=20)
    cmv_std60 = rolling_std_np(circ_mv, 60, min_periods=20, ddof=0)
    tr_mean60 = rolling_mean_np(turnover_rate, 60, min_periods=20)
    tr_std60 = rolling_std_np(turnover_rate, 60, min_periods=20, ddof=0)
    vola_60 = rolling_std_np(ret_1d, 60, min_periods=20, ddof=0)
    vola_20 = rolling_std_np(ret_1d, 20, min_periods=5, ddof=0)
    tm_ma20 = rolling_mean_np(turnover_rate, 20, min_periods=5)

    # 价量匹配
    inflow_5d = rolling_sum_np(net_mf, 5, min_periods=2)
    cmv_safe = np.where(circ_mv == 0, np.nan, circ_mv)

    # 在 helper 位置 s 返回所有 37 维 (news 两维由调用方跳过)
    return {
        # 价格派生
        "ret_1d": float(ret_1d[s]),
        "ret_5d": float(ret_5d[s]),
        "ret_20d": float(ret_20d[s]),
        "range_pct": float((high[s] - low[s]) / close[s]),
        "gap": float((open_[s] - prev_close[s]) / prev_close[s]) if prev_close[s] != 0 else np.nan,
        # 均线
        "ma5_dev": float(close[s] / ma5[s] - 1) if not np.isnan(ma5[s]) else np.nan,
        "ma20_dev": float(close[s] / ma20[s] - 1) if not np.isnan(ma20[s]) else np.nan,
        "ma60_dev": float(close[s] / ma60[s] - 1) if not np.isnan(ma60[s]) else np.nan,
        "ma_short_long": float(ma5[s] / ma20[s] - 1)
        if not np.isnan(ma20[s]) and not np.isnan(ma5[s])
        else np.nan,
        # 量能
        "vol_z20": float((volume[s] - vol_ma20[s]) / vol_std20[s])
        if not np.isnan(vol_std20[s]) and vol_std20[s] != 0
        else np.nan,
        "vol_ratio_5": float(vol_ratio_5_arr[s]),
        "vol_ret_corr20": float(rolling_corr_np(ret_1d, volume, 20, min_periods=20)[s]),
        # 技术
        "rsi12": float(rsi[s] / 100 - 0.5),
        "macd_hist_norm": float(macd_hist[s] / close[s]) if close[s] != 0 else np.nan,
        "kdj_j": float(kdj[s] / 100 - 0.5),
        "atr14_pct": float(atr[s] / close[s]) if close[s] != 0 else np.nan,
        # 基本面
        "pe_log": float(pe_log[s]),
        "pb_log": float(pb_log[s]),
        "turnover": float(turnover_rate[s]),
        "mv_log": float(mv_log[s]),
        # 资金流
        "inflow_pct": float(net_mf[s] / total_mv_safe[s])
        if not np.isnan(total_mv_safe[s])
        else np.nan,
        "big_ratio": float(big[s] / total_buy_safe[s])
        if not np.isnan(total_buy_safe[s])
        else np.nan,
        # 涨停史
        "days_since_limit": float(days_since[s]),
        "n_limits_20d": float(n_limits_20[s]),
        "n_limits_60d": float(n_limits_60[s]),
        "max_ret_20d": float(max_ret_20[s]),
        "consecutive_up": float(consec_up[s]),
        "is_yi_zi": float(yi_zi[s]),
        # 规模 / 波动
        "circ_mv_log": float(np.log(max(circ_mv[s], 1))),
        "circ_mv_z_self60": float((circ_mv[s] - cmv_mean60[s]) / (cmv_std60[s] + 1e-9))
        if not np.isnan(cmv_mean60[s])
        else np.nan,
        "turnover_ma20": float(tm_ma20[s]),
        "turnover_self_z60": float((turnover_rate[s] - tr_mean60[s]) / (tr_std60[s] + 1e-9))
        if not np.isnan(tr_mean60[s])
        else np.nan,
        "vola_60d": float(vola_60[s]),
        "vola_ratio_20_60": float(vola_20[s] / (vola_60[s] + 1e-9))
        if not np.isnan(vola_60[s]) and not np.isnan(vola_20[s])
        else np.nan,
        # 价量匹配
        "price_vol_match5": float(ret_5d[s] * np.tanh(vol_ratio_5_arr[s] - 1))
        if not np.isnan(ret_5d[s]) and not np.isnan(vol_ratio_5_arr[s])
        else np.nan,
        "inflow_5d_sum": float(inflow_5d[s] / cmv_safe[s])
        if not np.isnan(cmv_safe[s]) and not np.isnan(inflow_5d[s])
        else np.nan,
        "vol_ret_corr60": float(rolling_corr_np(ret_1d, volume, 60, min_periods=20)[s]),
    }


# =============================================================================
# Step 2b: 因果性检测 (非后验判断 — 真 DB 上跑)
# =============================================================================


def audit_causality(
    panel: pd.DataFrame,
    hold_out_days: int = 20,
    tolerance_atol: float = 1e-10,
) -> list[AuditAlert]:
    """因果性测试: 截短 panel 跑 compute_features, common prefix 必须一致.

    算法:
        1. feats_full = compute_features(panel)
        2. feats_trunc = compute_features(panel[:-hold_out_days])
        3. 对 common 时间段 (panel[:len-hold_out]), 验证每个 feature 相等
        4. 不等 → look-ahead bias (feature 用了未来数据)

    Args:
        panel: 原始 panel (含 OHLCV + daily_basic + ...)
        hold_out_days: 截掉的天数 (越大 → 测试越严格)
        tolerance_atol: 容忍的浮点误差

    Returns: list[AuditAlert]. 空 list = 通过 (无 look-ahead).
    """
    if panel.empty or len(panel) < hold_out_days + 60:
        return []
    # 按 code 截短 (对每只股去掉最后 hold_out_days 天). 手动 groupby 避开 pandas 2.2+
    # include_groups 废弃问题.
    panel_sorted = panel.sort_values(["code", "trade_date"])
    trunc_parts: list[pd.DataFrame] = []
    for _code, g in panel_sorted.groupby("code", sort=False):
        if len(g) > hold_out_days:
            trunc_parts.append(g.iloc[:-hold_out_days])
        else:
            trunc_parts.append(g)
    trunc = pd.concat(trunc_parts, ignore_index=True)

    # 对截短版同时截短 DB 时点 (news feature 需要, 否则 DB 全量会让 news 看到"未来")
    trunc_max_date = trunc["trade_date"].max().strftime("%Y-%m-%d")
    trunc_sources = _build_truncated_sources(trunc_max_date)

    feats_full = compute_features(panel_sorted)
    feats_trunc = compute_features(trunc, sources=trunc_sources)

    # common prefix = trunc 中的所有行 (code, trade_date)
    merge_keys = ["code", "trade_date"]
    merged = feats_full.merge(
        feats_trunc,
        on=merge_keys,
        suffixes=("_full", "_trunc"),
    )
    alerts: list[AuditAlert] = []
    for source in FEATURE_SOURCES:
        for col in source.output_cols:
            col_full = f"{col}_full"
            col_trunc = f"{col}_trunc"
            if col_full not in merged.columns or col_trunc not in merged.columns:
                continue
            diff = (merged[col_full] - merged[col_trunc]).abs()
            bad = diff > tolerance_atol
            # NaN != NaN 会被 diff 变 NaN, 被 > atol 评估为 False → 自动忽略
            if bad.sum() > 0:
                max_diff = float(diff[bad].max())
                alerts.append(
                    AuditAlert(
                        feature=col,
                        source=f"causality/{source.name}",
                        issue="look_ahead_bias",
                        value=max_diff,
                        threshold=tolerance_atol,
                        affected_codes=int(bad.sum()),
                        details=(
                            f"{source.name}.{col}: 截短 {hold_out_days} 天后 {int(bad.sum())} 个 cell "
                            f"不一致, max_diff={max_diff:.2e}"
                        ),
                    )
                )
    return alerts


def _build_truncated_sources(max_mention_date: str) -> tuple[FeatureSource, ...]:
    """对 FEATURE_SOURCES 创建截短变体 — _CctvNewsSource 用 max_mention_date 过滤 DB.

    只替换 _CctvNewsSource 实例, 其它 source (不查 DB) 原样复用.
    """
    new_sources: list[FeatureSource] = []
    for s in FEATURE_SOURCES:
        if isinstance(s, _CctvNewsSource):
            # 从原 cfg 继承 db_path/table, 加 max_mention_date
            trunc_cfg = CctvNewsConfig(
                db_path=s._cfg.db_path,
                table=s._cfg.table,
                max_mention_date=max_mention_date,
            )
            new_sources.append(_CctvNewsSource(trunc_cfg))
        else:
            new_sources.append(s)
    return tuple(new_sources)


# =============================================================================
# Step 3: NaN fill + 越界 clip (范围检测失败后处理)
# =============================================================================


def _apply_nan_fill_rules(feats: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    """按每列 rule 补救当日 feats — NaN 填充 + 越界 winsorize.

    **NaN 填充** (nan_fill 策略):
        zero   : NaN → 0 (cross_sectional_zscore 后也是 0)
        prev   : NaN → 该 code 前一日该 feature 值 (stock-level ffill)
        median : NaN → 当日截面中位数
        keep   : 不改 (模型自己学缺失)

    **越界处理** (out_of_range_action):
        clip        : winsorize 到 [lo, hi] (默认, 业界标准防极端污染 zscore mean/std)
        alert       : 不改值, 仅报告
        drop_stock  : 超限股的该 feature 设 NaN (保守, 与 NaN 策略联动)

    阈值 lo/hi: 优先用滚动 252 天 p0.5/p99.5, 样本不够用 rule.expected_min/max.
    只处理 trade_date 当日行, 历史数据不改.
    """
    mask = feats["trade_date"] == pd.Timestamp(trade_date)
    today_idx = feats.index[mask]
    if len(today_idx) == 0:
        return feats

    rolling = _compute_rolling_thresholds(feats, trade_date)
    for source in FEATURE_SOURCES:
        for col, rule in source.quality_rules.items():
            _fill_nan_col(feats, col, rule, today_idx)
            _handle_range_col(feats, col, rule, rolling.get(col), today_idx)
    return feats


def _fill_nan_col(
    feats: pd.DataFrame,
    col: str,
    rule: ColumnQualityRule,
    today_idx: pd.Index,
) -> None:
    nan_today = feats.loc[today_idx, col].isna()
    if not nan_today.any():
        return
    if rule.nan_fill == "zero":
        feats.loc[today_idx[nan_today], col] = 0.0
    elif rule.nan_fill == "median":
        median = feats.loc[today_idx, col].median()
        feats.loc[today_idx[nan_today], col] = 0.0 if pd.isna(median) else median
    elif rule.nan_fill == "prev":
        # ffill per code — 对每只 NaN 的股, 取其该 feature 前一日值
        for ix in today_idx[nan_today]:
            code = feats.at[ix, "code"]
            hist = feats[
                (feats["code"] == code) & (feats["trade_date"] < feats.at[ix, "trade_date"])
            ]
            if not hist.empty:
                prev_val = hist.iloc[-1][col]
                if pd.notna(prev_val):
                    feats.at[ix, col] = prev_val
    # keep: 不做事


def _handle_range_col(
    feats: pd.DataFrame,
    col: str,
    rule: ColumnQualityRule,
    rolling_range: tuple[float, float] | None,
    today_idx: pd.Index,
) -> None:
    """按 out_of_range_action 处理超限: clip / drop_stock / alert (不改).

    优先使用滚动 252 天阈值 (rolling_range), fallback 到 rule.expected_min/max.
    """
    # 决定 lo/hi
    if rolling_range is not None:
        lo, hi = rolling_range
    elif rule.expected_min is not None and rule.expected_max is not None:
        lo, hi = rule.expected_min, rule.expected_max
    else:
        return  # 无阈值可用

    if rule.out_of_range_action == "clip":
        feats.loc[today_idx, col] = feats.loc[today_idx, col].clip(lower=lo, upper=hi)
    elif rule.out_of_range_action == "drop_stock":
        today_vals = feats.loc[today_idx, col]
        out_mask = (today_vals < lo) | (today_vals > hi)
        if out_mask.any():
            feats.loc[today_idx[out_mask], col] = np.nan
    # 'alert': 不改值 (仅在 _check_value_range 阶段报告)


# =============================================================================
# Step 4: 持久化 report
# =============================================================================


def _write_audit_log(report: AuditReport, log_dir: str | Path) -> Path:
    d = Path(log_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"audit-{report.trade_date}.json"
    p.write_text(report.to_json(), encoding="utf-8")
    return p
