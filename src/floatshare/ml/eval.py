"""PPO 模型评估 — 加载 ckpt, 跑 test set, 输出 signal CSV + 打印指标.

输出:
    data/ml/signals/daily.csv  — (date, token_id, weight)
    控制台: sharpe / cum_return / max_drawdown / turnover
"""

from __future__ import annotations

import argparse
import dataclasses as _dc
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from floatshare.ml.config import DataConfig, PPOConfig, TrainConfig
from floatshare.ml.data.dataset import build_cube
from floatshare.ml.data.loader import load_market_returns
from floatshare.ml.evaluation.env import run_deterministic_rollout
from floatshare.ml.model.agent import load_ckpt
from floatshare.ml.rl.env import MarketEnv
from floatshare.observability import logger

if TYPE_CHECKING:
    from floatshare.ml.config import ModelConfig
    from floatshare.ml.model.agent import ActorCritic


# === 结果 dataclass — 替代原 dict 返回, 跨函数/模块属性访问而非字符串键 ===========


@dataclass(frozen=True, slots=True)
class SignalMetrics:
    """evaluate_signals 输出 — portfolio K-day alpha 指标."""

    sharpe: float
    cum_return: float
    mean_per_step: float
    max_drawdown: float
    turnover_avg: float
    n_steps: int
    n_signals: int


@dataclass(frozen=True, slots=True)
class SuspendedCase:
    """某日选股在次日实际停牌的一条记录."""

    trade_date: str
    next_date: str
    code: str
    weight: float


@dataclass(frozen=True, slots=True)
class SuspensionReport:
    """count_recommend_then_suspended 输出 — T→T+1 停牌盲点命中率."""

    total: int
    suspended_next_day: int
    rate: float
    rate_weighted: float
    cases: list[SuspendedCase] = field(default_factory=list)


def predict_signals(
    env: MarketEnv,
    model: ActorCritic,
    model_cfg: ModelConfig,
    cube_dates: np.ndarray,
    token_ids: list[str],
    device: torch.device,
) -> pd.DataFrame:
    """对 env 全程跑 inference, 输出 long-format DataFrame (date, token_id, weight)."""
    rows: list[dict] = []

    def _on_step(t: int, _state, weights_full: np.ndarray, _reward: float) -> None:
        cur_date = cube_dates[t]
        for i, tid in enumerate(token_ids):
            if weights_full[i] > 1e-6:
                rows.append(
                    {
                        "trade_date": pd.Timestamp(cur_date),
                        "token_id": tid,
                        "weight": float(weights_full[i]),
                    }
                )

    run_deterministic_rollout(env, model, device, model_cfg, _on_step)
    return pd.DataFrame(rows)


def evaluate_signals(
    signals: pd.DataFrame,
    cube_dates: np.ndarray,
    log_returns: np.ndarray,
    token_ids: list[str],
    market_returns: np.ndarray,
    K: int = 5,
) -> SignalMetrics:
    """按 trade_date 聚合 signals 算 portfolio K-day alpha → sharpe/drawdown/turnover."""
    if signals.empty:
        return _empty_signal_metrics(len(signals))

    daily_alpha, turnovers = _compute_daily_alpha_and_turnover(
        signals,
        cube_dates,
        log_returns,
        token_ids,
        market_returns,
        K,
    )
    rs = np.array(daily_alpha, dtype=np.float64)
    if len(rs) < 2:
        return _empty_signal_metrics(len(signals))

    cum = np.cumsum(rs)
    peak = np.maximum.accumulate(cum)
    return SignalMetrics(
        sharpe=float(rs.mean() / (rs.std() + 1e-8) * np.sqrt(252 / K)),
        cum_return=float(rs.sum()),
        mean_per_step=float(rs.mean()),
        max_drawdown=float((cum - peak).min()),
        turnover_avg=float(np.mean(turnovers)) if turnovers else 0.0,
        n_steps=len(rs),
        n_signals=len(signals),
    )


def _empty_signal_metrics(n_signals: int) -> SignalMetrics:
    return SignalMetrics(
        sharpe=0.0,
        cum_return=0.0,
        mean_per_step=0.0,
        max_drawdown=0.0,
        turnover_avg=0.0,
        n_steps=0,
        n_signals=n_signals,
    )


def count_recommend_then_suspended(
    signals: pd.DataFrame,
    cube_dates: np.ndarray,
    cube_traded: np.ndarray,
    token_ids: list[str],
    log_path: Path | None = None,
) -> SuspensionReport:
    """回测盲点测量: T 日选中 → T+1 实际停牌 的 case 占比.

    生产 T+1 07:00 推荐阶段无法前瞻拿停牌公告 (见 audit_mask.py 设计说明).
    用历史回测的"事后 ground truth" 反推该盲点的发生概率, 决定是否值得补前瞻 mask.

    Args:
        signals: predict_signals 输出 (trade_date, token_id, weight > 0)
        cube_dates: cube.dates — cube 时间轴, 用于定位 T/T+1
        cube_traded: cube.traded — (n_days, n_tokens) bool, True = 该日有成交
        token_ids: cube.tokens 的 code 列表 (和 cube_traded 第二维对齐)
        log_path: 非空则把每条 case 落 CSV (trade_date, next_date, code, weight)

    Returns:
        SuspensionReport (含 cases list, 属性访问 report.rate / report.cases 等).
    """
    if signals.empty:
        return SuspensionReport(total=0, suspended_next_day=0, rate=0.0, rate_weighted=0.0)

    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(cube_dates)}
    code_to_idx = {c: i for i, c in enumerate(token_ids)}

    cases: list[SuspendedCase] = []
    total = 0
    weight_total = 0.0
    weight_suspended = 0.0
    dates_arr = signals["trade_date"].to_numpy()
    codes_arr = signals["token_id"].to_numpy()
    weights_arr = signals["weight"].to_numpy(dtype=np.float64)
    for k in range(len(signals)):
        di = date_to_idx.get(pd.Timestamp(dates_arr[k]))
        ci = code_to_idx.get(str(codes_arr[k]))
        if di is None or ci is None or di + 1 >= len(cube_dates):
            continue
        total += 1
        w = float(weights_arr[k])
        weight_total += w
        if not cube_traded[di + 1, ci]:
            weight_suspended += w
            cases.append(
                SuspendedCase(
                    trade_date=pd.Timestamp(cube_dates[di]).date().isoformat(),
                    next_date=pd.Timestamp(cube_dates[di + 1]).date().isoformat(),
                    code=str(codes_arr[k]),
                    weight=w,
                )
            )

    rate = len(cases) / total if total else 0.0
    rate_weighted = weight_suspended / weight_total if weight_total else 0.0

    if cases:
        logger.warning(
            f"⚠️  {len(cases)}/{total} 次买入选股在次日实际停牌 "
            f"(unweighted={rate:.3%}, weighted={rate_weighted:.3%})"
        )
        for c in cases[:10]:
            logger.info(f"  {c.trade_date} → {c.next_date}  {c.code}  w={c.weight:.4f}")
        if len(cases) > 10:
            logger.info(f"  ... 还有 {len(cases) - 10} 个")
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([_dc.asdict(c) for c in cases]).to_csv(log_path, index=False)
            logger.info(f"  cases CSV → {log_path}")
    else:
        logger.info(f"✓ {total} 次买入选股全部在次日有成交, 无停牌命中")

    return SuspensionReport(
        total=total,
        suspended_next_day=len(cases),
        rate=rate,
        rate_weighted=rate_weighted,
        cases=cases,
    )


def _compute_daily_alpha_and_turnover(
    signals: pd.DataFrame,
    cube_dates: np.ndarray,
    log_returns: np.ndarray,
    token_ids: list[str],
    market_returns: np.ndarray,
    K: int,
) -> tuple[list[float], list[float]]:
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(cube_dates)}
    code_to_idx = {c: i for i, c in enumerate(token_ids)}

    daily_alpha: list[float] = []
    turnovers: list[float] = []
    prev_w: dict[str, float] | None = None

    for d, g in signals.groupby("trade_date"):
        di = date_to_idx.get(pd.Timestamp(d))
        if di is None or di + K >= len(log_returns):
            continue
        K_ret = log_returns[di : di + K].sum(axis=0)
        mkt_K_ret = (
            float(market_returns[di : di + K].sum()) if di + K <= len(market_returns) else 0.0
        )

        idx = g["token_id"].map(code_to_idx).dropna().astype(int).to_numpy()
        w = g.loc[g["token_id"].isin(code_to_idx), "weight"].to_numpy(dtype=np.float32)
        daily_alpha.append(float((w * (K_ret[idx] - mkt_K_ret)).sum()))

        cur_w = dict(zip(g["token_id"], g["weight"], strict=False))
        if prev_w is not None:
            keys = set(cur_w) | set(prev_w)
            turnovers.append(sum(abs(cur_w.get(k, 0) - prev_w.get(k, 0)) for k in keys))
        prev_w = cur_w

    return daily_alpha, turnovers


def build_parser() -> argparse.ArgumentParser:
    """PPO eval CLI argparse — 放在 ml 层, 被 cli/run_eval_ppo.py 调用."""
    p = argparse.ArgumentParser(description="FloatShare PPO 评估 + signal 输出")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--phase", type=int, choices=(1, 2), default=1)
    from floatshare.ml.config import DataConfig

    _cfg = DataConfig()
    p.add_argument("--start", default=_cfg.test_start)
    p.add_argument("--end", default=_cfg.test_end)
    p.add_argument("--device", default="mps", choices=("mps", "cuda", "cpu"))
    p.add_argument("--out", default="data/ml/signals/daily.csv")
    p.add_argument("--universe-mode", default="top_mv")
    p.add_argument("--top-mv-n", type=int, default=300)
    return p


def run_eval(args) -> None:
    """PPO eval 主流程 — 不做 metrics bootstrap (由 cli 层包 cli_metrics_run).

    分层理由: ml 层不可 import application, 所以 bootstrap 放外层入口.
    """
    from floatshare.observability.metrics import Metric, record_counter, record_kpi, scope

    device = torch.device(args.device)
    model = load_ckpt(args.ckpt).to(device)
    model_cfg = model.cfg

    data_cfg = DataConfig(
        test_start=args.start,
        test_end=args.end,
        universe_mode=args.universe_mode,
        top_mv_n=args.top_mv_n,
    )
    _ = TrainConfig(device=args.device)

    cube = build_cube(data_cfg, args.start, args.end, phase=model_cfg.phase)
    market = load_market_returns(data_cfg.db_path, args.start, args.end).to_numpy()

    env = MarketEnv(cube, PPOConfig(), model_cfg, market_returns=market)
    token_ids = [t.token_id for t in cube.tokens]

    logger.info(f"加载 ckpt: {args.ckpt}, phase={model_cfg.phase}, params={model.n_params():,}")
    logger.info(f"test cube: {cube.n_days} days × {cube.n_tokens} tokens")

    signals = predict_signals(env, model, model_cfg, cube.dates, token_ids, device)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.nan_to_num(
            np.log(cube.prices[1:] / cube.prices[:-1]),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    metrics = evaluate_signals(
        signals,
        cube.dates,
        log_ret,
        token_ids,
        market,
        K=PPOConfig().reward_horizon,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(out_path, index=False)
    logger.info(f"signal CSV → {out_path}  ({len(signals)} rows)")
    logger.info(
        f"指标: sharpe={metrics.sharpe:+.3f}  "
        f"cum_return={metrics.cum_return:+.4f}  "
        f"mean_per_step={metrics.mean_per_step:+.5f}  "
        f"max_dd={metrics.max_drawdown:+.4f}  "
        f"turnover={metrics.turnover_avg:.3f}  "
        f"n_steps={metrics.n_steps}",
    )

    # T+1 停牌盲点分析 — 替代前瞻性 suspend_d, 事后测概率
    suspended_log = out_path.with_name(out_path.stem + "_suspended_next_day.csv")
    suspension = count_recommend_then_suspended(
        signals, cube.dates, cube.traded, token_ids, log_path=suspended_log
    )

    # === 自动 record_kpi — SignalMetrics 每字段一条 KPI (Cookbook 8.9: fields 即名字) ===
    ckpt_id = Path(args.ckpt).stem  # 'v9-ckpt123' 风格
    subject = f"{ckpt_id}@{args.start}..{args.end}"
    for f in _dc.fields(metrics):
        record_kpi(Metric.Domain.BACKTEST, subject, f.name, float(getattr(metrics, f.name)))

    # 派生 KPI: excess_ratio = (strategy_cum - bench_cum) / |bench_cum|
    bench_cum = float(market.sum()) if len(market) else 0.0
    excess_ratio = (metrics.cum_return - bench_cum) / abs(bench_cum) if bench_cum else 0.0
    record_kpi(
        Metric.Domain.BACKTEST,
        subject,
        Metric.Kpi.EXCESS_RATIO,
        excess_ratio,
        bench_cum=bench_cum,
    )

    # Suspension 盲点 KPI (前瞻性 mask 缺失的影响度量)
    record_kpi(
        Metric.Domain.BACKTEST,
        subject,
        Metric.Kpi.SUSPENDED_BLINDSPOT_RATE,
        suspension.rate,
        weighted=suspension.rate_weighted,
        total_picks=suspension.total,
    )

    # 每日 log-return → counter (web 看板画逐日 pnl 曲线)
    backtest_scope = scope(Metric.Domain.BACKTEST, ckpt_id)
    for i, d in enumerate(cube.dates[:-1]):
        if i < len(log_ret):
            record_counter(
                backtest_scope,
                Metric.Counter.DAILY_LOG_RETURN,
                float(log_ret[i].sum()),
                trade_date=pd.Timestamp(d).date().isoformat(),
            )


if __name__ == "__main__":
    # 独立运行时退化: 走本模块的 run_eval, 无 metrics sink
    run_eval(build_parser().parse_args())
