"""第三方对拍 — tushare stk_factor 的 MACD/KDJ/RSI 跟我们 compute_features 对比.

为什么要做:
    - audit_manual.py 是我们自己写的 numpy 实现, 跟 features.py 是 **同源代码翻译**
    - spot-check "0 mismatch" 只说明两份自家实现一致, 不代表跟业界标准一致
    - tushare stk_factor 是**独立第三方实现**, 用 Wilder 标准公式
    - 对拍 tushare 能抓到:
        a. features.py 的公式错误 (我们和 audit_manual 都写错的情况)
        b. 跨实现的初值/NaN 处理差异
        c. 业界标准 vs 我们实现的系统性偏差

覆盖 feature (对齐后):
    - rsi12          ← tushare.rsi_12  (周期 12, 两边对齐)
    - kdj_j          ← tushare.kdj_j   (9/3/3 参数)
    - macd_hist_norm ← tushare.macd    (=(dif-dea)*2, 反归一化需 × close)

跳过:
    - rsi_6, rsi_24    — 我们不用这俩周期
    - MACD DIF/DEA     — 我们只保留 hist, tushare 提供 3 个字段都能对
    - ATR              — tushare stk_factor 没有 ATR, 可选 stk_factor_pro

时效:
    tushare stk_factor 当日收盘后 ~18:00 可拉 T 日数据.
    注意 shift_days=1: 我们 feats[T, 'rsi12'] 对应 tushare[T-1, 'rsi_12'].

用法:
    from floatshare.infrastructure.data_sources.tushare import TushareSource
    from floatshare.ml.audit_tushare import run_tushare_spot_check
    src = TushareSource(token=...)
    alerts = run_tushare_spot_check(feats, "2026-04-20", src, sample_codes=20)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from floatshare.observability import logger

if TYPE_CHECKING:
    from floatshare.infrastructure.data_sources.tushare import TushareSource

# 跨实现的累积浮点误差 (基于 2026-04-20 × 460 股 真 DB 校准):
#   RSI12: 0% 差异 (前复权 + n=12 完美对齐 tushare)
#   KDJ J: 0% 差异 (前复权修复除权日污染)
#   MACD:  max|diff|=0.0005 (pandas EWM vs tushare EWM 数值精度差 + tushare 4 位小数 rounding)
_ATOL_RSI = 0.5  # RSI 12 值域 [0, 100], 0.5 = 0.5%
_ATOL_KDJ = 1.0  # KDJ J 值域可超 [0, 100] (可到 ±150), 宽一点
_ATOL_MACD_REL = 0.05  # macd 相对误差 5% (macd 数量级小, 2% 会被 tushare rounding 误触发)
_ATOL_MACD_ABS = 5e-4  # macd 绝对阈值: tushare 只存 4 位小数, 0.0005 差异是 rounding 噪声


@dataclass(frozen=True, slots=True)
class TushareMismatch:
    """tushare 对拍不一致的一条记录."""

    feature: str  # 'rsi12' / 'kdj_j' / 'macd_hist_norm'
    code: str
    trade_date: str  # 我们 feats 的 T (tushare 实际对的是 T-1)
    our_value: float
    tushare_value: float
    diff: float
    atol: float


def run_tushare_spot_check(
    feats: pd.DataFrame,
    trade_date: str,
    tushare_source: TushareSource,
    sample_codes: int | None = None,
    atol_rsi: float = _ATOL_RSI,
    atol_kdj: float = _ATOL_KDJ,
    atol_macd_rel: float = _ATOL_MACD_REL,
) -> list[TushareMismatch]:
    """从 tushare stk_factor 拉 T-1 日数据, 对拍我们 feats[T] 的 rsi12/kdj_j/macd_hist_norm.

    因 feats 用 shift_days=1, 我们的 feats[T, col] = helper@(T-1),
    所以应 match tushare 在 **T-1** 的值.

    Args:
        feats: compute_features 输出 (必须含 code, trade_date, close, rsi12, kdj_j, macd_hist_norm)
        trade_date: 我们 feats 里的 T (YYYY-MM-DD)
        tushare_source: 已初始化的 TushareSource
        sample_codes: 抽样股数 (None=全量, 但会压 tushare 限速; 建议 20-50)

    Returns:
        list[TushareMismatch]. 空 list = 全匹配 (误差 < atol).
    """
    helper_date_ts = pd.Timestamp(trade_date) - pd.Timedelta(days=1)
    # tushare.cctv_news uses YYYYMMDD str; stk_factor by trade_date also YYYYMMDD
    # helper_date 可能落周末, 取最近交易日
    helper_date = _last_trading_day(helper_date_ts.date())
    logger.info(
        f"tushare spot check: feats[T={trade_date}] vs tushare[T-1={helper_date}]",
    )

    # 拉 tushare stk_factor (全市场, 按日期单参 API)
    tushare_df = tushare_source.pro.stk_factor(trade_date=helper_date.strftime("%Y%m%d"))
    if tushare_df.empty:
        logger.warning(f"tushare stk_factor T-1={helper_date} 返回空, 跳过对拍")
        return []
    tushare_df = tushare_df.rename(columns={"ts_code": "code"})
    tushare_map = tushare_df.set_index("code")

    today = feats[feats["trade_date"] == pd.Timestamp(trade_date)]
    if sample_codes is not None and sample_codes < len(today):
        today = today.sample(sample_codes, random_state=0)

    mismatches: list[TushareMismatch] = []
    for row in today.itertuples(index=False):
        code = str(row.code)
        if code not in tushare_map.index:
            continue
        ts_row = tushare_map.loc[code]

        # RSI 12: 我们 feats = rsi/100 - 0.5 → 反归一化 = (feats + 0.5) * 100
        our_rsi = (float(row.rsi12) + 0.5) * 100
        ts_rsi = float(ts_row.get("rsi_12", np.nan))
        if not np.isnan(ts_rsi) and abs(our_rsi - ts_rsi) > atol_rsi:
            mismatches.append(
                TushareMismatch(
                    feature="rsi12",
                    code=code,
                    trade_date=trade_date,
                    our_value=our_rsi,
                    tushare_value=ts_rsi,
                    diff=our_rsi - ts_rsi,
                    atol=atol_rsi,
                )
            )

        # KDJ J: 我们 feats = j/100 - 0.5 → 反归一化
        our_kdj = (float(row.kdj_j) + 0.5) * 100
        ts_kdj = float(ts_row.get("kdj_j", np.nan))
        if not np.isnan(ts_kdj) and abs(our_kdj - ts_kdj) > atol_kdj:
            mismatches.append(
                TushareMismatch(
                    feature="kdj_j",
                    code=code,
                    trade_date=trade_date,
                    our_value=our_kdj,
                    tushare_value=ts_kdj,
                    diff=our_kdj - ts_kdj,
                    atol=atol_kdj,
                )
            )

        # MACD hist: 我们 feats = macd_hist / close_qfq. 反归一化需 × close_qfq[T-1].
        # tushare 技术指标已是 qfq 版, tushare.close 是 raw close, 所以应用 ts_row.close_qfq
        # (如果 tushare 接口返回 close_qfq 列, 否则 fallback 到 close)
        ts_close_qfq = float(ts_row.get("close_qfq", ts_row.get("close", np.nan)))
        if np.isnan(ts_close_qfq) or ts_close_qfq == 0:
            continue
        our_macd_hist = float(row.macd_hist_norm) * ts_close_qfq
        ts_macd = float(ts_row.get("macd", np.nan))  # tushare.macd = (dif-dea)*2
        if not np.isnan(ts_macd):
            abs_diff = abs(our_macd_hist - ts_macd)
            # 绝对 OR 相对 任一在容忍内就 pass (防 MACD 小值被相对误差误触发)
            if abs_diff > _ATOL_MACD_ABS and abs_diff / max(abs(ts_macd), 1e-6) > atol_macd_rel:
                mismatches.append(
                    TushareMismatch(
                        feature="macd_hist_norm",
                        code=code,
                        trade_date=trade_date,
                        our_value=our_macd_hist,
                        tushare_value=ts_macd,
                        diff=our_macd_hist - ts_macd,
                        atol=max(_ATOL_MACD_ABS, atol_macd_rel * abs(ts_macd)),
                    )
                )
    return mismatches


def _last_trading_day(d: datetime | pd.Timestamp) -> datetime:
    """简化: 周末退到周五. 真交易日历需查 trade_calendar."""
    if isinstance(d, pd.Timestamp):
        d = d.to_pydatetime()
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d -= timedelta(days=1)
    return d
