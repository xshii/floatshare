"""ML 层 pipeline stage fn — 需要依赖 ml/data + ml/features + ml/audit*.

不能放 application/pipeline/ (分层契约禁止 application → ml).
Caller (cli/run_pipeline.py) 组装 StagePolicy tuple 时一起引入.

每个 stage fn 签名兼容 application.pipeline.stages.StageContext:
    (StageContext) → dict[str, Any]
"""

from __future__ import annotations

import sqlite3
from datetime import date
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from floatshare.domain.pipeline import StageContext


def stage_s2b_prep_features(ctx: StageContext) -> dict[str, Any]:
    """universe + load_panel + compute_features. 把 feats/panel/codes 塞 ctx.shared."""
    from floatshare.domain.enums import ExchangeSuffix
    from floatshare.ml.data.loader import load_panel
    from floatshare.ml.features import compute_features

    with sqlite3.connect(str(ctx.db_path)) as conn:
        rows = conn.execute(
            f"""
            SELECT DISTINCT r.code
            FROM raw_daily r
            LEFT JOIN stock_lifecycle l ON l.code = r.code
            WHERE r.trade_date LIKE ?
              AND r.code NOT LIKE '%{ExchangeSuffix.BJ}'
              AND (l.name IS NULL OR (l.name NOT LIKE 'ST%' AND l.name NOT LIKE '*ST%'))
            """,
            (f"{ctx.trade_date}%",),
        ).fetchall()
    codes = [r[0] for r in rows]
    # 窗口: T-365 日历日 (约 240 交易日, MACD 26 EMA + 安全余量)
    t = date.fromisoformat(ctx.trade_date)
    start = (t.replace(year=t.year - 1) if t.month > 2 else date(t.year - 1, 1, 1)).isoformat()
    panel = load_panel(str(ctx.db_path), codes, start, ctx.trade_date)
    feats = compute_features(panel)
    ctx.shared["panel"] = panel
    ctx.shared["feats"] = feats
    ctx.shared["codes"] = codes
    return {
        "n_codes": len(codes),
        "panel_rows": len(panel),
        "feats_rows": len(feats),
    }


def stage_s3a_tushare_check(ctx: StageContext, sample_codes: int = 50) -> dict[str, Any]:
    """RSI12/KDJ_J/MACD 对拍 tushare stk_factor — 需要 ctx.shared['feats']."""
    from floatshare.infrastructure.data_sources.tushare import TushareSource
    from floatshare.ml.audit_tushare import run_tushare_spot_check

    feats = ctx.shared.get("feats")
    if feats is None:
        raise RuntimeError("stage_s2b 缺前置 feats (应由 stage_s2b_prep_features 先跑)")

    ts = TushareSource()
    mm = run_tushare_spot_check(feats, ctx.trade_date, ts, sample_codes=sample_codes)
    return {
        "sample_size": sample_codes,
        "n_mismatches": len(mm),
        "features": sorted({m.feature for m in mm}) if mm else [],
    }


def stage_s3b_feature_audit(ctx: StageContext) -> dict[str, Any]:
    """rolling 252 winsorize + 39 特征质量 — 需要 ctx.shared['feats']."""
    from floatshare.ml.audit import run_feature_audit

    feats = ctx.shared.get("feats")
    if feats is None:
        raise RuntimeError("stage_s3 缺前置 feats")

    report, _ = run_feature_audit(feats, ctx.trade_date, panel=None, raise_on_error=False)
    return {
        "n_alerts": len(report.alerts),
        "n_features_checked": report.n_features_checked,
        "has_errors": report.has_errors(),
    }


# === S4 train (evening 跨夜) =================================================


def stage_s4_train(
    ctx: StageContext,
    epochs: int = 3,
    val_window_days: int = 60,
) -> dict[str, Any]:
    """S4: warm-start 增量训练 N epochs.

    时间预估 (Mac mps, 默认 epochs=3):
        - 冷缓存首跑: 40-60 min
        - 暖缓存 (pickle cache 命中): 30-45 min
        - CPU fallback: 2-3 h

    Val 窗口设计: train 到 trade_date - val_window_days, val 取最近 N 天,
    避免 train/val 重叠 (floatshare-train-pop 默认 val=2024 会嵌在 train 里 → 数据泄露).

    生产 v9 长训练 (30+ epochs 完整冷启动) 独立跑, 不走 pipeline.
    Pipeline 里只做 warm-start 增量. 失败 → RuntimeError, runner 按 SOFT 记 FAIL.
    """
    import subprocess
    from datetime import date, timedelta

    from floatshare.ml.pipeline_stages_internal import find_best_ckpt

    best = find_best_ckpt()
    t = date.fromisoformat(ctx.trade_date)
    val_end = t
    val_start = t - timedelta(days=val_window_days)
    train_end = val_start - timedelta(days=1)  # train_end < val_start, 防边界漏

    cmd = [
        "floatshare-train-pop",
        "--epochs",
        str(epochs),
        "--end",
        train_end.isoformat(),
        "--val-start",
        val_start.isoformat(),
        "--val-end",
        val_end.isoformat(),
    ]
    if best is not None:
        cmd.extend(["--resume-from", str(best)])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600 * 10, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"floatshare-train-pop 失败 rc={r.returncode}: {r.stderr[-500:]}")

    new_best = find_best_ckpt()
    if new_best is None:
        raise RuntimeError("S4 训练完毕但未找到 ckpt — 检查 data/ml/ckpts/")
    return {
        "epochs": epochs,
        "resume_from": str(best) if best else "(cold start)",
        "best_ckpt": str(new_best),
        "ckpt_size_mb": round(new_best.stat().st_size / 1_000_000, 2),
        "train_end": train_end.isoformat(),
        "val_window": f"{val_start.isoformat()}..{val_end.isoformat()}",
    }


# === S5 recommend (morning) ==================================================


def stage_s5_recommend(ctx: StageContext, top_k: int = 10) -> dict[str, Any]:
    """S5: 用 best ckpt 预测今日 top-K 候选, 写 recommend-YYYY-MM-DD.csv + notify.

    TODO: 现在是 scaffold — 完整实现需要:
        1. build_cube(trade_date, trade_date) 单日 cube
        2. load_ckpt(best) + model.forward → P(hit)
        3. 应用 universe 过滤 (BJ 已在 universe.py 排除) + tradable mask
        4. np.argpartition top-K + 按 P(hit) 降序
        5. 写 CSV: code / p_hit / expected_return / rank
        6. notify 推送 top-K 清单 + best ckpt id

    缺失条件: 目前无 v9 ckpt, 无 daily-recommend 简化入口. 等 v9 训练就位后填.
    """
    raise NotImplementedError(
        "S5 stage 尚未实装 — 需要先跑 v9 训练产出 ckpt + 实现 daily-recommend 逻辑. "
        "orchestrator 骨架已就绪, 填充时替换本函数即可."
    )
