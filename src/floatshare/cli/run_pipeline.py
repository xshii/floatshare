"""CLI 入口 — `floatshare-pipeline`. launchd 定时触发.

用法:
    floatshare-pipeline                                  # 默认 evening phase, today
    floatshare-pipeline --trade-date 2026-04-22
    floatshare-pipeline --phase morning                  # S5 recommend
    floatshare-pipeline --skip-preflight                 # 调试用

Stage 编号约定:
    数字 = 执行阶段; 字母 (A/B/C) = 同阶段内可并行
    S1 + S1C news 并行 → {S2A db_integrity, S2B prep_features} → {S3A tushare_check, S3B feature_audit} → S4 train
    S5 recommend 独立 morning phase (依赖前夜 S4 产出 ckpt)

CLI 层负责组装 application + ml 两层的 stage fn — 绕开 application → ml 分层禁令.
"""

from __future__ import annotations

import argparse
from datetime import date


def _evening_stages():
    """Evening phase 7 个 stage: S1 + S1C → S2 并行组 → S3 并行组 → S4.

    数据层 S1/S2 FAST (阻断式), 新闻/模型/审计层 S1C/S3/S4 SOFT (记 FAIL 但继续).
    S1C 拉不到新闻 → news 特征全 0, 不阻断后续训练.
    """
    from floatshare.application.pipeline.runner import StagePolicy
    from floatshare.application.pipeline.stages import (
        stage_s1_sync,
        stage_s1c_news_ingest,
        stage_s2a_db_integrity,
    )
    from floatshare.domain.enums import FailPolicy, PipelineStage
    from floatshare.ml.pipeline_stages import (
        stage_s2b_prep_features,
        stage_s3a_tushare_check,
        stage_s3b_feature_audit,
        stage_s4_train,
    )

    return (
        StagePolicy(PipelineStage.S1_SYNC, FailPolicy.FAST, stage_s1_sync),
        StagePolicy(PipelineStage.S1C_NEWS_INGEST, FailPolicy.SOFT, stage_s1c_news_ingest),
        StagePolicy(PipelineStage.S2A_DB_INTEGRITY, FailPolicy.FAST, stage_s2a_db_integrity),
        StagePolicy(PipelineStage.S2B_PREP_FEATURES, FailPolicy.FAST, stage_s2b_prep_features),
        StagePolicy(PipelineStage.S3A_TUSHARE_CHECK, FailPolicy.SOFT, stage_s3a_tushare_check),
        StagePolicy(PipelineStage.S3B_FEATURE_AUDIT, FailPolicy.SOFT, stage_s3b_feature_audit),
        StagePolicy(PipelineStage.S4_TRAIN, FailPolicy.SOFT, stage_s4_train),
    )


def _morning_stages():
    """Morning phase 仅 S5 recommend — 用前夜 S4 训好的 ckpt 预测今日 top-K."""
    from floatshare.application.pipeline.runner import StagePolicy
    from floatshare.domain.enums import FailPolicy, PipelineStage
    from floatshare.ml.pipeline_stages import stage_s5_recommend

    return (StagePolicy(PipelineStage.S5_RECOMMEND, FailPolicy.SOFT, stage_s5_recommend),)


_PHASE_STAGES = {
    "evening": _evening_stages,
    "morning": _morning_stages,
}


def main() -> None:
    from floatshare.application.bootstrap import cli_metrics_run
    from floatshare.application.pipeline.runner import run_pipeline

    p = argparse.ArgumentParser(description="FloatShare pipeline orchestrator")
    p.add_argument("--trade-date", default=date.today().isoformat(), help="T 日 YYYY-MM-DD")
    p.add_argument(
        "--skip-preflight",
        action="store_true",
        help="跳过 env/DB 预检 (调试用, 生产不要加)",
    )
    p.add_argument(
        "--phase",
        default="evening",
        choices=tuple(_PHASE_STAGES.keys()),
        help="evening=S1~S4, morning=S5",
    )
    args = p.parse_args()

    stages = _PHASE_STAGES[args.phase]()
    with cli_metrics_run():
        summary = run_pipeline(args.trade_date, stages, skip_preflight=args.skip_preflight)

    if summary.aborted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
