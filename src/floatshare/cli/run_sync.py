"""命令行入口 — `floatshare-sync` 批量同步数据到本地 SQLite。

默认增量模式: 每个数据类型查询本地最新日期，仅拉取之后的数据。
用 --force 覆盖（重新拉指定窗口的全部数据）。

用法示例:
    # 增量同步指定股票（首次会拉全量，后续只追加增量）
    floatshare-sync --codes 600000.SH 000001.SZ --start 2020-01-01 --end 2026-04-17

    # 强制重拉
    floatshare-sync --codes 600000.SH --start 2024-01-01 --force

    # 拉券商金股
    floatshare-sync --include broker_picks --months 202401 202402

    # 全 A 股 limit
    floatshare-sync --all-stocks --limit 100 --start 2024-01-01
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING

# 触发集中 register (cli + web 都从这里读, 单一真相来源)
from floatshare.application import feature_registry  # noqa: F401
from floatshare.application.sync_progress import (
    SyncStage,
    SyncStatus,
    empty_progress,
)
from floatshare.domain.enums import DataKind, DataSourceKind
from floatshare.domain.records import (
    AdjFactor,
    Balancesheet,
    BrokerPicks,
    Cashflow,
    ChipDist,
    ChipPerf,
    CnCpi,
    CnPpi,
    ConceptBoard,
    ConceptMember,
    DailyBasic,
    Dividend,
    EarningsForecast,
    FinaIndicator,
    FxDaily,
    Income,
    IndexDaily,
    Industry,
    MarginDetail,
    Moneyflow,
    MoneyflowHsgt,
    RawDaily,
    RecordSchema,
    Shibor,
    StkHolderNumber,
    StockLifecycle,
    TopInst,
    TopList,
)
from floatshare.observability import logger, notify

if TYPE_CHECKING:
    from collections.abc import Callable
    from enum import StrEnum

    import pandas as pd

    from floatshare.infrastructure.data_sources.tushare import TushareSource
    from floatshare.infrastructure.storage.database import DatabaseStorage


# 排序即 sync 推荐顺序: lifecycle 先（用于后续窗口收窄）
_SYNC_ORDER: tuple[StrEnum, ...] = tuple(DataKind.all())


# 默认主流指数 (用 --include index_weight + --indexes 覆盖)
_DEFAULT_INDEXES: tuple[str, ...] = (
    "000300.SH",  # 沪深 300
    "000905.SH",  # 中证 500
    "000016.SH",  # 上证 50
    "000688.SH",  # 科创 50
    "399006.SZ",  # 创业板指
)

# 默认外汇币对 (用 --fx 覆盖)
# 注: Tushare FXCM 源只有 CNH (离岸人民币)，没有 CNY (在岸人民币，外汇管制)
_DEFAULT_FX_CODES: tuple[str, ...] = (
    "USDCNH.FXCM",  # 美元离岸人民币 (做 CNH 对冲 / 汇率因子)
    "EURUSD.FXCM",  # 欧元美元 (美元指数主要分量)
    "USDJPY.FXCM",  # 美元日元 (避险情绪)
    "XAUUSD.FXCM",  # 黄金美元 (商品/避险)
)

# 默认指数日线 — 风格轮动 + 行业轮动两类研究的基础数据
# 用 --index-daily-codes 覆盖
_DEFAULT_STYLE_INDICES: tuple[str, ...] = (
    "000300.SH",  # 沪深 300         (大盘核心)
    "000905.SH",  # 中证 500         (中盘)
    "000852.SH",  # 中证 1000        (小盘)
    "000016.SH",  # 上证 50          (大盘蓝筹)
    "399006.SZ",  # 创业板指         (成长)
    "000688.SH",  # 科创 50          (新经济)
    "000922.CSI",  # 中证红利         (高股息防御)
    "000919.CSI",  # 300 价值
    "000918.CSI",  # 300 成长
)
# 申万 31 个一级行业 (SW2021)
_DEFAULT_SW_L1_INDICES: tuple[str, ...] = (
    "801010.SI",  # 农林牧渔
    "801030.SI",  # 基础化工
    "801040.SI",  # 钢铁
    "801050.SI",  # 有色金属
    "801080.SI",  # 电子
    "801110.SI",  # 家用电器
    "801120.SI",  # 食品饮料
    "801130.SI",  # 纺织服饰
    "801140.SI",  # 轻工制造
    "801150.SI",  # 医药生物
    "801160.SI",  # 公用事业
    "801170.SI",  # 交通运输
    "801180.SI",  # 房地产
    "801200.SI",  # 商贸零售
    "801210.SI",  # 社会服务
    "801230.SI",  # 综合
    "801710.SI",  # 建筑材料
    "801720.SI",  # 建筑装饰
    "801730.SI",  # 电力设备
    "801740.SI",  # 国防军工
    "801750.SI",  # 计算机
    "801760.SI",  # 传媒
    "801770.SI",  # 通信
    "801780.SI",  # 银行
    "801790.SI",  # 非银金融
    "801880.SI",  # 汽车
    "801890.SI",  # 机械设备
    "801950.SI",  # 煤炭
    "801960.SI",  # 石油石化
    "801970.SI",  # 环保
    "801980.SI",  # 美容护理
)
_DEFAULT_INDEX_DAILY_CODES: tuple[str, ...] = (
    *_DEFAULT_STYLE_INDICES,
    *_DEFAULT_SW_L1_INDICES,
)


# === per-code job spec 表 — clean dispatch，避免 lambda 重复 ===============


@dataclass(frozen=True, slots=True)
class _DailyJob:
    """per-code date-range sync 配置 (覆盖 9/16 数据类型)。

    table 是派生属性 — 直接来自 save_record.TABLE，避免重复书写 + 漂移风险。
    """

    fetch_attr: str  # ts.<attr>
    save_record: type[RecordSchema]  # db.save(save_record, df) — 强类型
    date_col: str = "trade_date"
    one_shot: bool = False  # True = fetch(code) 一次返全 (财务类)
    stale_days: int | None = None  # one_shot 过期阈值

    @property
    def table(self) -> str:
        return self.save_record.TABLE


_DAILY_JOBS: dict[StrEnum, _DailyJob] = {
    DataKind.DAILY.RAW_DAILY: _DailyJob("get_raw_daily", RawDaily),
    DataKind.DAILY.DAILY_BASIC: _DailyJob("get_daily_basic", DailyBasic),
    DataKind.DAILY.CHIP_PERF: _DailyJob("get_chip_perf", ChipPerf),
    DataKind.DAILY.MONEYFLOW: _DailyJob("get_moneyflow", Moneyflow),
    DataKind.DAILY.MARGIN_DETAIL: _DailyJob("get_margin_detail", MarginDetail),
    DataKind.INTRADAY_HEAVY.CHIP_DIST: _DailyJob("get_chip_dist", ChipDist),
    DataKind.FUNDAMENTAL.FORECAST: _DailyJob(
        "get_earnings_forecast", EarningsForecast, date_col="report_date"
    ),
    DataKind.FUNDAMENTAL.INCOME: _DailyJob(
        "get_income", Income, date_col="end_date", one_shot=True, stale_days=100
    ),
    DataKind.FUNDAMENTAL.BALANCESHEET: _DailyJob(
        "get_balancesheet", Balancesheet, date_col="end_date", one_shot=True, stale_days=100
    ),
    DataKind.FUNDAMENTAL.CASHFLOW: _DailyJob(
        "get_cashflow", Cashflow, date_col="end_date", one_shot=True, stale_days=100
    ),
    DataKind.FUNDAMENTAL.FINA_INDICATOR: _DailyJob(
        "get_fina_indicator", FinaIndicator, date_col="end_date", one_shot=True, stale_days=100
    ),
    DataKind.FUNDAMENTAL.HOLDER_NUMBER: _DailyJob(
        "get_holder_number", StkHolderNumber, date_col="end_date", one_shot=True, stale_days=90
    ),
    DataKind.EVENT.DIVIDEND: _DailyJob(
        "get_dividend", Dividend, date_col="end_date", one_shot=True, stale_days=100
    ),
}


@dataclass(frozen=True, slots=True)
class _ByDateJob:
    """全市场按交易日同步: 1 次 API 拿当天 5500 票, 替代 5500 次 per-code。

    适用 Tushare 接口: daily / daily_basic / moneyflow / adj_factor /
    margin_detail / cyq_perf — 都接受 trade_date='YYYYMMDD' 单参数返当日全市场。
    """

    by_date_attr: str
    save_record: type[RecordSchema]

    @property
    def table(self) -> str:
        return self.save_record.TABLE


# 走 by-date 模式的 6 个高频日表 (覆盖 daily-sync 90%+ 的 API 调用量)
#
# adj_factor 走 smart 路径: 每天 by-date 拉新行 + 查除权事件 + 仅对受影响
# codes 全量 refresh (~80% 的日子无事件, 平均一天 < 50 codes 触发修正)
#
# 不在的:
#   chip_dist  — Tushare cyq_chips 不支持 trade_date 全市场, per-code only
#   index_daily — 走 _DEFAULT_INDEX_DAILY_CODES 列表 per-index, 量已经很小
_BY_DATE_JOBS: dict[StrEnum, _ByDateJob] = {
    DataKind.DAILY.RAW_DAILY: _ByDateJob("get_raw_daily_by_date", RawDaily),
    DataKind.DAILY.DAILY_BASIC: _ByDateJob("get_daily_basic_by_date", DailyBasic),
    DataKind.DAILY.MONEYFLOW: _ByDateJob("get_moneyflow_by_date", Moneyflow),
    DataKind.DAILY.MARGIN_DETAIL: _ByDateJob("get_margin_detail_by_date", MarginDetail),
    DataKind.DAILY.CHIP_PERF: _ByDateJob("get_chip_perf_by_date", ChipPerf),
    DataKind.DAILY.ADJ_FACTOR: _ByDateJob("get_adj_factor_by_date", AdjFactor),
}


def main() -> None:
    from floatshare.application.bootstrap import cli_metrics_run

    args = _build_parser().parse_args()
    db, ts = _bootstrap()
    with cli_metrics_run(existing_db=db):
        _run_sync(args, db, ts)


def _run_sync(args, db, ts) -> None:
    """sync 主流程 — 与 main 分离让 with cli_metrics_run() 包得干净."""
    from floatshare.observability.metrics import record_kpi

    codes = _resolve_codes(args, ts, db)
    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None

    selected: set[StrEnum] = (
        {DataKind.from_value(s) for s in args.include} if args.include else set(_SYNC_ORDER)
    )
    mode = "force" if args.force else "incremental"
    logger.info(
        f"开始同步 [{mode}] — codes={len(codes)} types={sorted(s.value for s in selected)} "
        f"window=[{start}, {end}]"
    )

    errors: list[str] = []
    progress = empty_progress(
        pid=os.getpid(),
        selected=sorted(s.value for s in selected),
        codes_total=len(codes),
    )
    progress.write()

    def section(label: str, fn: Callable[[], None]) -> None:
        """每个独立 section 用这个跑，失败累计但不中断 (P1: 推送汇总用)。"""
        try:
            fn()
        except Exception as exc:
            logger.error(f"[{label}] failed: {exc}")
            errors.append(f"{label}: {exc}")
            progress.errors = len(errors)
            progress.write()

    # === Batch 级别 (无 code 循环) ==========================================

    if DataKind.REFERENCE.LIFECYCLE in selected:
        section("lifecycle", lambda: _sync_lifecycle(ts, db))
    if DataKind.REFERENCE.INDUSTRY in selected:
        section("industry", lambda: _sync_industry(ts, db))
    if DataKind.REFERENCE.CONCEPT in selected:
        section("concept", lambda: _sync_concepts(ts, db))
    if DataKind.MARKET.MONEYFLOW_HSGT in selected:
        section("moneyflow_hsgt", lambda: _sync_moneyflow_hsgt(ts, db, start, end, args.force))

    _MACRO_JOBS: tuple[tuple[StrEnum, type, str], ...] = (
        (DataKind.MARKET.CN_CPI, CnCpi, "get_cn_cpi"),
        (DataKind.MARKET.CN_PPI, CnPpi, "get_cn_ppi"),
        (DataKind.MARKET.SHIBOR, Shibor, "get_shibor"),
    )
    for kind, record_cls, fetch_attr in _MACRO_JOBS:
        if kind in selected:
            section(
                kind.value,
                partial(
                    _sync_macro,
                    ts,
                    db,
                    record_cls,
                    fetch_attr,
                    start,
                    end,
                    args.force,
                ),
            )

    if DataKind.MARKET.FX_DAILY in selected:
        for fx_code in args.fx or _DEFAULT_FX_CODES:
            section(
                f"fx_daily/{fx_code}",
                partial(
                    _sync_fx_daily,
                    ts,
                    db,
                    fx_code,
                    start,
                    end,
                    args.force,
                ),
            )

    if DataKind.REFERENCE.INDEX_WEIGHT in selected:
        for index_code in args.indexes or _DEFAULT_INDEXES:
            section(
                f"index_weight/{index_code}",
                partial(
                    _sync_index_weight,
                    ts,
                    db,
                    index_code,
                    start,
                    end,
                    args.force,
                ),
            )

    if DataKind.MARKET.INDEX_DAILY in selected:
        for index_code in args.index_daily_codes or _DEFAULT_INDEX_DAILY_CODES:
            section(
                f"index_daily/{index_code}",
                partial(
                    _sync_index_daily,
                    ts,
                    db,
                    index_code,
                    start,
                    end,
                    args.force,
                ),
            )

    # === by-date 全市场维度 (大头, daily-sync 走这条) ========================
    # 必须在 per-code 之前: 处理过的 kind 从 selected 剔除, 避免重复

    if args.by_date and start and end:
        progress.stage = SyncStage.PER_DAY.value
        progress.write()
        try:
            handled = _sync_universe_by_date(ts, db, start, end, args.force, selected)
        except Exception as exc:
            logger.error(f"[by-date] 路径失败: {exc}")
            errors.append(f"by-date: {exc}")
            handled = set()
        selected -= handled
        if handled:
            logger.info(f"[by-date] 完成 {len(handled)} 个表, 剩余 {len(selected)} 类型走 per-code")

    # === per-code dispatch (大部分数据走这条) ================================

    progress.stage = SyncStage.PER_CODE.value
    progress.write()
    for i, code in enumerate(codes, start=1):
        progress.current_code = code
        for kind in _SYNC_ORDER:
            if kind not in selected:
                continue
            if kind in _DAILY_JOBS:
                _run_safely(
                    _per_code_runner(ts, db, _DAILY_JOBS[kind], start, end, args.force),
                    code,
                    kind.value,
                )
            elif kind == DataKind.DAILY.ADJ_FACTOR:
                _run_safely(
                    lambda c: _sync_adj_factor(ts, db, c, start, end),
                    code,
                    "adj_factor",
                )
        progress.codes_done = i
        # 每 5 个 code 刷一次盘 (~每分钟一次)，省 IO
        if i % 5 == 0 or i == len(codes):
            progress.write()

    # === per-month (event 类) ================================================

    if DataKind.EVENT.BROKER_PICKS in selected and args.months:
        for month in args.months:
            section(
                f"broker_picks/{month}",
                partial(
                    _sync_broker_picks,
                    ts,
                    db,
                    month,
                    args.force,
                ),
            )

    # === per-day batch (龙虎榜) ==============================================

    progress.stage = SyncStage.PER_DAY.value
    progress.write()
    if start and end:
        if DataKind.EVENT.TOP_LIST in selected:
            section(
                "top_list",
                partial(
                    _sync_per_day_batch,
                    ts,
                    db,
                    start,
                    end,
                    args.force,
                    fetch_attr="get_top_list",
                    record_cls=TopList,
                    label="top_list",
                ),
            )
        if DataKind.EVENT.TOP_INST in selected:
            section(
                "top_inst",
                partial(
                    _sync_per_day_batch,
                    ts,
                    db,
                    start,
                    end,
                    args.force,
                    fetch_attr="get_top_inst",
                    record_cls=TopInst,
                    label="top_inst",
                ),
            )

    progress.status = (SyncStatus.FAILED if errors else SyncStatus.DONE).value
    progress.stage = SyncStage.DONE.value
    progress.finished_at = datetime.now().isoformat(timespec="seconds")
    progress.write()

    # 刷新 web 用的 table_counts snapshot — 避免 web 现算等 25s
    try:
        from floatshare.application.db_snapshot import refresh_counts_snapshot

        refresh_counts_snapshot(db)
        logger.info("[sync] table-counts.json 刷新")
    except Exception as exc:
        logger.warning(f"[sync] snapshot 刷新失败 (非致命): {exc}")

    logger.info("同步完成 ✓")
    _notify_summary(codes, selected, errors)

    # KPI 落盘 — SyncKpis dataclass 字段 = KPI 名 (单一真相来源, 无魔法字符串)
    import dataclasses as _dc

    from floatshare.domain.records import SyncKpis
    from floatshare.observability.metrics import Metric

    kpis = SyncKpis(tables_synced=len(selected), error_count=len(errors), codes_total=len(codes))
    today = date.today().isoformat()
    for f in _dc.fields(kpis):
        record_kpi(Metric.Domain.SYNC, today, f.name, getattr(kpis, f.name))


def _notify_summary(
    codes: list[str],
    selected: set[StrEnum],
    errors: list[str],
) -> None:
    """推 Bark — 成功简报或失败摘要。"""
    sel_str = ",".join(sorted(s.value for s in selected))
    if errors:
        body = "\n".join(errors[:5])
        if len(errors) > 5:
            body += f"\n...还有 {len(errors) - 5} 个"
        notify(f"⚠️ Sync 部分失败 {len(errors)}", body)
    else:
        notify("✓ Sync 完成", f"codes={len(codes)} types=[{sel_str}]")


def _per_code_runner(
    ts: TushareSource,
    db: DatabaseStorage,
    job: _DailyJob,
    start: date | None,
    end: date | None,
    force: bool,
) -> Callable[[str], None]:
    """构造单参数 (code) 闭包，喂给 _run_safely。"""
    fetch_fn = getattr(ts, job.fetch_attr)
    record_cls = job.save_record

    def base_save(df: pd.DataFrame) -> int:
        return db.save(record_cls, df)

    # raw_daily 特殊：成功后更新 watermark
    if job.table == RawDaily.TABLE:

        def save_fn(df: pd.DataFrame) -> int:
            n = base_save(df)
            if not df.empty:
                db.update_watermark(
                    df["code"].iloc[0],
                    df["trade_date"].max().date(),
                    DataSourceKind.PAID_REMOTE.TUSHARE.value,
                )
            return n
    else:
        save_fn = base_save

    def runner(code: str) -> None:
        _execute_per_code_job(db, job, code, start, end, force, fetch_fn, save_fn)

    return runner


def _execute_per_code_job(
    db: DatabaseStorage,
    job: _DailyJob,
    code: str,
    start: date | None,
    end: date | None,
    force: bool,
    fetch_fn: Callable,
    save_fn: Callable[[pd.DataFrame], int],
) -> None:
    """执行单个 per-code job — 区分 incremental 和 one-shot 模式。"""
    if job.one_shot:
        # 财务类: 过期检查 + 一次性拉
        if not force and job.stale_days is not None:
            last = db.latest_date(job.table, code, date_col=job.date_col)
            if last is not None and (date.today() - last).days < job.stale_days:
                logger.debug(f"[{job.table}] {code} fresh (last {last}), skip")
                return
        df = fetch_fn(code)
        if df.empty:
            return
        n = save_fn(df)
        logger.info(f"[{job.table}] {code} {n} rows (one-shot)")
    else:
        _sync_daily_table(
            db,
            code,
            start,
            end,
            force,
            table=job.table,
            fetch=fetch_fn,
            save=save_fn,
            date_col=job.date_col,
        )


def _sync_macro(
    ts: TushareSource,
    db: DatabaseStorage,
    record_cls: type,
    fetch_attr: str,
    start: date | None,
    end: date | None,
    force: bool,
) -> None:
    """市场级宏观数据 (CPI/PPI/SHIBOR) — 无 code 循环，单表覆盖。

    date_col 列名由 record_cls.PK[0] 推断 (month 或 date)。
    """
    date_col = record_cls.PK[0]  # type: ignore[attr-defined]
    table = record_cls.TABLE  # type: ignore[attr-defined]
    if not force and start is None:
        if date_col == "month":
            # cn_cpi / cn_ppi 的 month 列是 "YYYYMM" 字符串，
            # 不能走 pd.to_datetime（会被当 YYMMDD 解析）
            latest = _latest_yyyymm(db, table)
            if latest is not None:
                yr, mo = int(latest[:4]), int(latest[4:])
                yr2, mo2 = (yr, mo + 1) if mo < 12 else (yr + 1, 1)
                start = date(yr2, mo2, 1)
        else:
            last = db.latest_date(table, date_col=date_col)
            if last is not None:
                start = last + timedelta(days=1)
    fetch_fn = getattr(ts, fetch_attr)
    df = fetch_fn(start, end)
    if df.empty:
        return
    n = db.save(record_cls, df)
    logger.info(f"[{table}] +{n} rows")


def _latest_yyyymm(db: DatabaseStorage, table: str) -> str | None:
    """读 cn_cpi / cn_ppi 最新 month（'YYYYMM' 字符串）。"""
    from sqlalchemy import text

    try:
        with db.engine.connect() as conn:
            row = conn.execute(text(f"SELECT MAX(month) FROM {table}")).fetchone()
    except Exception:
        return None
    return row[0] if row and row[0] else None


def _sync_fx_daily(
    ts: TushareSource,
    db: DatabaseStorage,
    fx_code: str,
    start: date | None,
    end: date | None,
    force: bool,
) -> None:
    """外汇日行情 — 按币对拉取。"""
    if not force:
        last = db.latest_date(FxDaily.TABLE, code=fx_code)
        if last is not None and start is None:
            start = last + timedelta(days=1)
    df = ts.get_fx_daily(fx_code, start, end)
    if df.empty:
        return
    n = db.save(FxDaily, df)
    logger.info(f"[fx_daily] {fx_code} +{n} rows")


def _sync_per_day_batch(
    ts: TushareSource,
    db: DatabaseStorage,
    start: date,
    end: date,
    force: bool,
    fetch_attr: str,
    record_cls: type,
    label: str,
) -> None:
    """按交易日批量拉取 (top_list / top_inst)：逐日 fetch(trade_date=d)，已有数据可跳。"""
    trading_days = ts.get_trade_calendar(start, end)
    fetch_fn = getattr(ts, fetch_attr)
    table = record_cls.TABLE  # type: ignore[attr-defined]  # 所有 dataclass 都带 TABLE
    total = 0
    for d in trading_days:
        if not force and db.has_rows(table, trade_date=d.isoformat()):
            continue
        try:
            df = fetch_fn(d)
        except Exception as exc:
            logger.error(f"[{label}] {d} failed: {exc}")
            continue
        if df.empty:
            continue
        n = db.save(record_cls, df)
        total += n
    logger.info(f"[{label}] +{total} rows over {len(trading_days)} days")


def _sync_moneyflow_hsgt(
    ts: TushareSource,
    db: DatabaseStorage,
    start: date | None,
    end: date | None,
    force: bool,
) -> None:
    """北向资金 (市场级别) — 增量同步。

    Tushare 返回上限 300 行/次，按年分片避免截断。
    """
    if not force:
        last = db.latest_date(MoneyflowHsgt.TABLE)
        if last is not None and start is None:
            start = last + timedelta(days=1)

    eff_start = start or date(2010, 1, 1)
    eff_end = end or date.today()
    if eff_start > eff_end:
        logger.debug(f"[moneyflow_hsgt] up-to-date (start {eff_start} > end {eff_end})")
        return

    total = 0
    for year in range(eff_start.year, eff_end.year + 1):
        y_start = max(eff_start, date(year, 1, 1))
        y_end = min(eff_end, date(year, 12, 31))
        df = ts.get_moneyflow_hsgt(y_start, y_end)
        if not df.empty:
            total += db.save(MoneyflowHsgt, df)
    if total:
        logger.info(f"[moneyflow_hsgt] +{total} rows ({eff_start} ~ {eff_end})")


def _sync_index_weight(
    ts: TushareSource,
    db: DatabaseStorage,
    index_code: str,
    start: date | None,
    end: date | None,
    force: bool,  # noqa: ARG001  保留接口一致性
) -> None:
    """指数成分股权重 — Tushare 内部已处理增量，直接覆盖。"""
    from floatshare.domain.records import IndexWeight

    df = ts.get_index_weight(index_code, start, end)
    if df.empty:
        return
    n = db.save(IndexWeight, df)
    logger.info(f"[index_weight] {index_code} +{n} rows")


def _sync_index_daily(
    ts: TushareSource,
    db: DatabaseStorage,
    index_code: str,
    start: date | None,
    end: date | None,
    force: bool,
) -> None:
    """指数日线 OHLCV — 风格轮动核心数据。watermark 增量。"""
    if not force:
        last = db.latest_date(IndexDaily.TABLE, code=index_code)
        if last is not None and start is None:
            start = last + timedelta(days=1)
    df = ts.get_index_daily(index_code, start, end)
    if df.empty:
        return
    n = db.save(IndexDaily, df)
    logger.info(f"[index_daily] {index_code} +{n} rows")


# --- 工厂 + 解析 ----------------------------------------------------------


def _bootstrap() -> tuple[DatabaseStorage, TushareSource]:
    from floatshare.infrastructure.data_sources.tushare import TushareSource
    from floatshare.infrastructure.storage.database import DatabaseStorage

    db = DatabaseStorage()
    db.init_tables()
    ts = TushareSource()
    if not ts.token:
        raise SystemExit("缺少 TUSHARE_TOKEN — 请检查 .env")
    return db, ts


def _resolve_codes(args, ts: TushareSource, db: DatabaseStorage) -> list[str]:
    if args.codes:
        return list(args.codes)
    if args.all_stocks:
        df = ts.get_stock_list()
        codes = df["code"].tolist()
        if args.limit:
            codes = codes[: args.limit]
        db.save_stock_list(df)
        return codes
    return []


def _run_safely(fn: Callable, arg: str, label: str) -> None:
    try:
        fn(arg)
    except Exception as exc:
        logger.error(f"[{label}] {arg} failed: {exc}")


# --- 增量同步引擎 --------------------------------------------------------


def _parse_iso(s: str | None) -> date | None:
    """解析 'YYYY-MM-DD' 字符串为 date，None/空透传。"""
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def _effective_window(
    db: DatabaseStorage,
    code: str,
    requested_start: date | None,
    requested_end: date | None,
) -> tuple[date | None, date | None] | None:
    """根据 stock_lifecycle 收窄请求窗口。

    返回 (eff_start, eff_end)；若该股从未上市或退市后请求 → None（应跳过整个同步）。
    若 lifecycle 表无该 code 记录 → 不收窄，原样返回。
    """
    lc = db.get_lifecycle(code)
    if lc is None:
        return requested_start, requested_end

    list_date = _parse_iso(lc.list_date)
    delist_date = _parse_iso(lc.delist_date)

    eff_start: date | None = (
        max(requested_start, list_date)
        if (requested_start and list_date)
        else (requested_start or list_date)
    )
    eff_end: date | None
    if delist_date:
        eff_end = min(requested_end, delist_date) if requested_end else delist_date
    else:
        eff_end = requested_end

    if eff_start and eff_end and eff_start > eff_end:
        return None  # 窗口无效（如退市后才请求）
    return eff_start, eff_end


def _missing_ranges(
    db: DatabaseStorage,
    table: str,
    code: str,
    requested_start: date | None,
    requested_end: date | None,
    date_col: str = "trade_date",
) -> list[tuple[date | None, date | None]]:
    """计算需要拉取的日期区间列表，避开已有数据。

    返回最多 2 个区间:
    - 历史回填: [requested_start, local_min - 1]（当 requested_start 早于本地最早日期时）
    - 前向追加: [local_max + 1, requested_end]（当 requested_end 晚于本地最新日期时）

    本地无数据时返回单个区间 [requested_start, requested_end]。
    """
    today = date.today()
    local_min, local_max = db.date_range(table, code, date_col)

    # 本地无数据 → 拉全窗口
    if local_min is None or local_max is None:
        return [(requested_start, requested_end)]

    ranges: list[tuple[date | None, date | None]] = []

    # 历史回填: 用户要的起点早于本地最早日期
    if requested_start is not None and requested_start < local_min:
        ranges.append((requested_start, local_min - timedelta(days=1)))

    # 前向追加: 用户要的终点晚于本地最新日期，且不在未来
    next_day = local_max + timedelta(days=1)
    fwd_end = requested_end if requested_end is not None else today
    if next_day <= fwd_end and next_day <= today:
        ranges.append((next_day, requested_end))

    return ranges


def _sync_daily_table(
    db: DatabaseStorage,
    code: str,
    start: date | None,
    end: date | None,
    force: bool,
    *,
    table: str,
    fetch: Callable,
    save: Callable,
    date_col: str = "trade_date",
    after_save: Callable | None = None,
) -> None:
    """通用：同步一个 (code, date_range) 表，自动识别缺失区间。

    流程: lifecycle 收窄请求窗口 → _missing_ranges 计算缺失区间 → 逐段拉取
    """
    if not force:
        narrowed = _effective_window(db, code, start, end)
        if narrowed is None:
            logger.debug(f"[{table}] {code} skipped (outside lifecycle window)")
            return
        start, end = narrowed

    if force:
        ranges: list[tuple[date | None, date | None]] = [(start, end)]
    else:
        ranges = _missing_ranges(db, table, code, start, end, date_col)

    if not ranges:
        logger.debug(f"[{table}] {code} up-to-date (no missing ranges)")
        return

    total = 0
    for fetch_start, fetch_end in ranges:
        df = fetch(code, fetch_start, fetch_end)
        if df.empty:
            continue
        n = save(df)
        total += n
        logger.info(f"[{table}] {code} +{n} rows [{fetch_start}, {fetch_end}]")
        if after_save:
            after_save(df)

    if total == 0:
        logger.debug(f"[{table}] {code} fetched but no new rows")


def _sync_adj_factor(
    ts: TushareSource,
    db: DatabaseStorage,
    code: str,
    start: date | None,
    end: date | None,
) -> None:
    """复权因子永远全量拉取 — 因子可能追溯修正，增量会漏。"""
    df = ts.get_adj_factor(code, start, end)
    if df.empty:
        return
    n = db.save(AdjFactor, df)
    logger.info(f"[adj_factor] {code} {n} rows (full refresh)")


def _sync_broker_picks(
    ts: TushareSource,
    db: DatabaseStorage,
    month: str,
    force: bool,
) -> None:
    """券商金股按月，本月已有就跳过。"""
    if not force and db.has_rows("broker_picks", month=month):
        logger.debug(f"[broker_picks] {month} already exists, skip")
        return
    df = ts.get_broker_picks(month)
    if df.empty:
        return
    n = db.save(BrokerPicks, df)
    logger.info(f"[broker_picks] {month} {n} rows")


def _sync_lifecycle(ts: TushareSource, db: DatabaseStorage) -> None:
    """同步全 A 股生命周期表（L+D+P 三状态合并）。"""
    df = ts.get_lifecycle()
    if df.empty:
        return
    n = db.save(StockLifecycle, df)
    logger.info(f"[lifecycle] +{n} rows (L+D+P merged)")


def _sync_universe_by_date(
    ts: TushareSource,
    db: DatabaseStorage,
    start: date,
    end: date,
    force: bool,
    selected: set[StrEnum],
) -> set[StrEnum]:
    """按交易日 × 全市场维度同步 (1 次 API/天/类型, 替代 N×K per-code)。

    adj_factor 走 smart 子路径: 每天拉新行 + 检测除权事件 + 仅对受影响
    codes 全量 refresh (因子有追溯修正)。

    返回处理过的 DataKind 集合 — 调用方应从 selected 中剔除, 避免后续
    per-code 路径重复拉取。
    """
    today = date.today()
    handled: set[StrEnum] = set()

    # 决定每个表是否需要拉 + 起点 (各表 watermark 独立)
    plan: list[tuple[StrEnum, _ByDateJob, date]] = []
    for kind, job in _BY_DATE_JOBS.items():
        if kind not in selected:
            continue
        s = start
        if not force:
            last = db.latest_date(job.table)  # 全表最新 trade_date
            if last is not None and s <= last:
                s = last + timedelta(days=1)
        if s > end:
            logger.info(f"[{job.table}] up-to-date (last >= {end}), skip")
            handled.add(kind)
            continue
        plan.append((kind, job, s))
        handled.add(kind)

    if not plan:
        return handled

    # 一次性拿交易日历, 全 plan 共享
    try:
        all_days = ts.get_trade_calendar(start, end)
    except Exception as exc:
        logger.error(f"[by-date] 拿交易日历失败: {exc}")
        return set()
    all_days = [d for d in all_days if d <= today]

    for kind, job, job_start in plan:
        days = [d for d in all_days if d >= job_start]
        if not days:
            continue
        fetch_fn = getattr(ts, job.by_date_attr)
        record_cls = job.save_record
        total = 0
        affected_codes_total = 0
        for d in days:
            try:
                df = fetch_fn(d)
            except Exception as exc:
                logger.error(f"[{job.table}] {d} failed: {exc}")
                continue
            if not df.empty:
                try:
                    total += db.save(record_cls, df)
                except Exception as exc:
                    logger.error(f"[{job.table}] {d} save failed: {exc}")
            # adj_factor 特殊: 当天有除权 → 受影响 codes 必须 full-refresh 历史
            if kind == DataKind.DAILY.ADJ_FACTOR:
                affected_codes_total += _refresh_affected_adj_factor(ts, db, d)
        suffix = (
            f", refreshed {affected_codes_total} codes (ex-events)"
            if kind == DataKind.DAILY.ADJ_FACTOR
            else ""
        )
        logger.info(
            f"[{job.table}] +{total} rows over {len(days)} days "
            f"({len(days)} API calls, by-date mode{suffix})"
        )
    return handled


def _refresh_affected_adj_factor(
    ts: TushareSource,
    db: DatabaseStorage,
    day: date,
) -> int:
    """检查某天的除权事件, 对受影响 codes 全量重拉 adj_factor 历史。

    返回 refresh 的 code 数 (无事件返 0, 无额外 API)。

    设计: 80% 的日子全市场无除权 → 1 次 dividend 探测 API 即返回。
    剩 20% 平均触发 < 50 codes, 每码 1 次 adj_factor 全量 refresh。
    """
    try:
        ex = ts.get_dividend_by_ex_date(day)
    except Exception as exc:
        logger.warning(f"[adj_factor smart] {day} 探测除权事件失败: {exc}")
        return 0
    if ex.empty or "code" not in ex.columns:
        return 0
    affected = sorted(ex["code"].dropna().unique().tolist())
    for code in affected:
        try:
            df = ts.get_adj_factor(code)  # 全量, 让追溯修正生效
            if not df.empty:
                db.save(AdjFactor, df)
        except Exception as exc:
            logger.error(f"[adj_factor smart] {day} {code} refresh 失败: {exc}")
    logger.info(f"[adj_factor smart] {day} 除权 {len(affected)} 只 → 全量重拉历史")
    return len(affected)


def _sync_industry(ts: TushareSource, db: DatabaseStorage) -> None:
    """同步申万 L1/L2/L3 行业分类映射 (code → industry)."""
    df = ts.get_industry()
    if df.empty:
        return
    n = db.save(Industry, df)
    logger.info(f"[industry] +{n} rows (SW L1/L2/L3)")


def _sync_concepts(ts: TushareSource, db: DatabaseStorage) -> None:
    """概念板块清单 + 全部成分股映射 (一次全量, 增量意义不大)。

    数据流: ts.get_concept_boards() → 写 concept_board
    然后对每个 board_code: ts.get_concept_members(c) → 写 concept_member
    板块约 ~700 (同花顺) / ~300 (Tushare ts), 总耗时 1-3 分钟。
    """
    boards = ts.get_concept_boards()
    if boards.empty:
        logger.warning("[concept] 拉到 0 个板块, 跳过")
        return
    n_boards = db.save(ConceptBoard, boards)
    logger.info(f"[concept] 板块 {n_boards} 个, 开始拉成分股 ...")

    n_members = 0
    for i, code in enumerate(boards["board_code"].tolist(), start=1):
        try:
            members = ts.get_concept_members(code)
            if not members.empty:
                n_members += db.save(ConceptMember, members)
        except Exception as exc:
            logger.warning(f"[concept] {code} 成分股拉取失败: {exc}")
        if i % 50 == 0:
            logger.info(f"[concept] 进度 {i}/{len(boards)}, 累计成分 {n_members} 行")
    logger.info(f"[concept] 完成: {n_boards} 板块 / {n_members} 成分映射")


# --- argparse ------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FloatShare 数据同步工具（增量模式）")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--codes", nargs="+", help="股票代码列表")
    src.add_argument("--all-stocks", action="store_true", help="拉取全 A 股")
    p.add_argument("--limit", type=int, help="--all-stocks 模式截取前 N 只")
    p.add_argument("--start", help="起始日期 YYYY-MM-DD")
    p.add_argument("--end", help="结束日期 YYYY-MM-DD")
    choices = [k.value for k in DataKind.all()]
    p.add_argument(
        "--include",
        nargs="+",
        choices=choices,
        help=f"数据类型（默认全部）: {choices}",
    )
    p.add_argument("--months", nargs="+", help="券商金股月份 YYYYMM")
    p.add_argument(
        "--indexes",
        nargs="+",
        help=f"指数代码列表 (--include index_weight 用)，默认 {_DEFAULT_INDEXES}",
    )
    p.add_argument(
        "--fx",
        nargs="+",
        help=f"外汇币对 (--include fx_daily 用)，默认 {_DEFAULT_FX_CODES}",
    )
    p.add_argument(
        "--index-daily-codes",
        nargs="+",
        dest="index_daily_codes",
        help=f"指数日线代码 (--include index_daily 用)，默认 {_DEFAULT_INDEX_DAILY_CODES}",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="忽略本地 watermark，重拉指定窗口的全部数据",
    )
    p.add_argument(
        "--by-date",
        action="store_true",
        dest="by_date",
        help=(
            "全市场按交易日维度拉取 (1 API/天/类型, 替代 5500x per-code), "
            "适用 daily-sync 短窗口场景, 提速 ~1000x"
        ),
    )
    return p


if __name__ == "__main__":
    main()
