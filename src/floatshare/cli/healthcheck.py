"""命令行入口 — `floatshare-healthcheck` 各数据源 API 健康度检查。

每个 source 跑一组 probe（小窗口/小代码量），记录:
  - status (OK/FAIL)
  - latency (ms)
  - rows returned
  - error message

结果同时写入 SQLite (healthcheck_log 表) 和打印到终端。

用法:
    floatshare-healthcheck                  # 全部源
    floatshare-healthcheck --source tushare
    floatshare-healthcheck --source akshare --code 600519.SH
    floatshare-healthcheck --output json    # 机器可读输出
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from enum import StrEnum

from floatshare.domain.enums import (
    DataSourceKind,
    HealthStatus,
    OutputFormat,
)
from floatshare.observability import logger, notify

# Probe 窗口大小 — 测试用，不取实际业务值
_PROBE_DAYS_DAILY = 10  # daily 类 (raw_daily, chip_perf, index 等) 拉最近 10 天
_PROBE_DAYS_FORECAST = 90  # forecast 类拉近 90 天 (季报间隔)


@dataclass(frozen=True, slots=True)
class _Probe:
    """单个 API 探测点。"""

    method: str
    fn: Callable  # 无参 closure，直接调用
    description: str = ""


@dataclass(frozen=True, slots=True)
class _Result:
    """单个 probe 的执行结果。"""

    source: str
    method: str
    status: HealthStatus
    latency_ms: int
    rows_returned: int | None
    error_message: str | None


# === Probe 工厂 — 每个源一组 =================================================


def _tushare_probes(code: str) -> list[_Probe]:
    from floatshare.infrastructure.data_sources.tushare import TushareSource

    src = TushareSource()
    end = date.today()
    start = end - timedelta(days=_PROBE_DAYS_DAILY)
    forecast_start = end - timedelta(days=_PROBE_DAYS_FORECAST)
    last_month = (end.replace(day=1) - timedelta(days=1)).strftime("%Y%m")
    return [
        # 参考类
        _Probe("get_stock_list", src.get_stock_list, "全 A 股列表"),
        _Probe("get_lifecycle", src.get_lifecycle, "L+D+P 生命周期"),
        _Probe("get_trade_calendar", lambda: src.get_trade_calendar(start, end), "交易日历"),
        _Probe(
            "get_index_weight",
            lambda: src.get_index_weight("000300.SH", start, end),
            "沪深 300 成分权重",
        ),
        # 行情类
        _Probe("get_raw_daily", lambda: src.get_raw_daily(code, start, end), "未复权日线"),
        _Probe("get_adj_factor", lambda: src.get_adj_factor(code, start, end), "复权因子"),
        _Probe(
            "get_index_daily", lambda: src.get_index_daily("000300.SH", start, end), "沪深 300 日线"
        ),
        _Probe("get_daily_basic", lambda: src.get_daily_basic(code, start, end), "PE/PB/股息率"),
        # 资金流类
        _Probe("get_moneyflow", lambda: src.get_moneyflow(code, start, end), "个股资金流"),
        _Probe("get_moneyflow_hsgt", lambda: src.get_moneyflow_hsgt(start, end), "北向资金"),
        # 筹码类
        _Probe("get_chip_perf", lambda: src.get_chip_perf(code, start, end), "筹码胜率"),
        _Probe("get_chip_dist", lambda: src.get_chip_dist(code, start, end), "筹码分布"),
        # 两融
        _Probe(
            "get_margin_detail", lambda: src.get_margin_detail(code, start, end), "个股两融明细"
        ),
        # 财务类
        _Probe(
            "get_earnings_forecast",
            lambda: src.get_earnings_forecast(code, forecast_start, end),
            "盈利预测",
        ),
        _Probe("get_income", lambda: src.get_income(code), "完整利润表"),
        _Probe("get_balancesheet", lambda: src.get_balancesheet(code), "资产负债表"),
        _Probe("get_cashflow", lambda: src.get_cashflow(code), "现金流量表"),
        _Probe("get_fina_indicator", lambda: src.get_fina_indicator(code), "财务衍生指标"),
        _Probe("get_holder_number", lambda: src.get_holder_number(code), "股东户数"),
        # 事件
        _Probe(
            "get_broker_picks", lambda: src.get_broker_picks(last_month), f"券商金股 {last_month}"
        ),
        _Probe("get_dividend", lambda: src.get_dividend(code), "分红送股"),
        _Probe("get_top_list", lambda: src.get_top_list(end - timedelta(days=1)), "龙虎榜个股"),
        _Probe("get_top_inst", lambda: src.get_top_inst(end - timedelta(days=1)), "龙虎榜席位"),
        # 宏观
        _Probe("get_cn_cpi", lambda: src.get_cn_cpi(start, end), "居民消费价格指数"),
        _Probe("get_cn_ppi", lambda: src.get_cn_ppi(start, end), "工业生产者出厂价格"),
        _Probe("get_shibor", lambda: src.get_shibor(start, end), "SHIBOR 利率"),
        _Probe(
            "get_fx_daily", lambda: src.get_fx_daily("USDCNY.FXCM", start, end), "美元人民币汇率"
        ),
    ]


def _akshare_probes(code: str) -> list[_Probe]:
    from floatshare.infrastructure.data_sources.akshare import AKShareSource

    src = AKShareSource()
    end = date.today()
    start = end - timedelta(days=_PROBE_DAYS_DAILY)
    return [
        _Probe("get_stock_list", src.get_stock_list, "全 A 股列表"),
        _Probe("get_raw_daily", lambda: src.get_raw_daily(code, start, end), "未复权日线"),
        _Probe("get_index_daily", lambda: src.get_index_daily("000300", start, end), "沪深 300"),
        _Probe("get_trade_calendar", lambda: src.get_trade_calendar(start, end), "交易日历"),
    ]


def _eastmoney_probes(code: str) -> list[_Probe]:
    from floatshare.infrastructure.data_sources.eastmoney import EastMoneySource

    src = EastMoneySource()
    end = date.today()
    start = end - timedelta(days=_PROBE_DAYS_DAILY)
    return [
        _Probe("get_stock_list", src.get_stock_list, "股票列表"),
        _Probe("get_daily", lambda: src.get_daily(code, start, end), "日线 (复权)"),
    ]


def _localdb_probes(code: str) -> list[_Probe]:
    from floatshare.application.db_integrity import (
        raise_if_any_duplicates,
        raise_if_cross_table_missing,
        raise_if_row_count_unstable,
    )
    from floatshare.infrastructure.data_sources.local_db import LocalDbSource

    src = LocalDbSource()
    db_path = str(src.db.db_path)
    return [
        _Probe("get_daily", lambda: src.get_daily(code), "本地日线 smoke"),
        _Probe("get_stock_list", src.get_stock_list, "本地股票列表"),
        _Probe(
            "trade_date_duplicates",
            lambda: raise_if_any_duplicates(db_path),
            "7 张日频主表 (code, trade_date) 唯一性",
        ),
        _Probe(
            "cross_table_alignment",
            lambda: raise_if_cross_table_missing(db_path),
            "raw_daily vs daily_basic/moneyflow 对齐",
        ),
        _Probe(
            "row_count_stability",
            lambda: raise_if_row_count_unstable(db_path),
            "近 30 交易日股数跳变 (±5%)",
        ),
    ]


def _cached_probes(code: str) -> list[_Probe]:
    from floatshare.infrastructure.data_sources.cached import CachedSource

    src = CachedSource()
    return [
        _Probe("get_daily", lambda: src.get_daily(code), "缓存日线"),
        _Probe("get_stock_list", src.get_stock_list, "缓存股票列表"),
    ]


_PROBE_BUILDERS: dict[StrEnum, Callable[[str], list[_Probe]]] = {
    DataSourceKind.PAID_REMOTE.TUSHARE: _tushare_probes,
    DataSourceKind.FREE_REMOTE.AKSHARE: _akshare_probes,
    DataSourceKind.FREE_REMOTE.EASTMONEY: _eastmoney_probes,
    DataSourceKind.LOCAL_PERSIST.LOCALDB: _localdb_probes,
    DataSourceKind.LOCAL_CACHE.CACHED: _cached_probes,
}


# === 执行引擎 ================================================================


def _run_probe(source: str, probe: _Probe) -> _Result:
    """执行单个 probe，捕获异常 + 测延迟。"""
    t0 = time.monotonic()
    try:
        ret = probe.fn()
        latency_ms = int((time.monotonic() - t0) * 1000)
        rows = _count_rows(ret)
        return _Result(source, probe.method, HealthStatus.OK, latency_ms, rows, None)
    except Exception as exc:
        latency_ms = int((time.monotonic() - t0) * 1000)
        return _Result(
            source,
            probe.method,
            HealthStatus.FAIL,
            latency_ms,
            None,
            str(exc)[:200],
        )


def _count_rows(ret: object) -> int | None:
    """从返回值推断行数 — DataFrame / list / dict / scalar。"""
    try:
        return len(ret)  # type: ignore[arg-type]
    except TypeError:
        return None


# === 报告 ====================================================================


_STATUS_ICON = {HealthStatus.OK: "✓", HealthStatus.FAIL: "✗"}


def _print_table(results: list[_Result]) -> None:
    """终端表格输出，按 source 分组。"""
    by_source: dict[str, list[_Result]] = {}
    for r in results:
        by_source.setdefault(r.source, []).append(r)

    for source, rs in by_source.items():
        ok = sum(1 for r in rs if r.status == HealthStatus.OK)
        total = len(rs)
        header = f"  {source}  —  {ok}/{total} OK"
        print()
        print("─" * 80)
        print(header)
        print("─" * 80)
        for r in rs:
            icon = _STATUS_ICON.get(r.status, "?")
            rows_str = f"{r.rows_returned:>6} rows" if r.rows_returned is not None else "       "
            line = f"  {icon}  {r.method:<24}  {r.status.value:<4}  {r.latency_ms:>5}ms  {rows_str}"
            if r.error_message:
                line += f"  — {r.error_message[:60]}"
            print(line)

    # 总览
    print()
    total_ok = sum(1 for r in results if r.status == HealthStatus.OK)
    print(f"📊 总计: {len(results)} probes, {total_ok} OK, {len(results) - total_ok} FAIL")


def _print_json(results: list[_Result]) -> None:
    print(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2))


# === CLI ====================================================================


def main() -> None:
    from floatshare.application.bootstrap import cli_metrics_run
    from floatshare.observability.metrics import Metric, record_counter, scope

    args = _build_parser().parse_args()
    sources: list[StrEnum] = _resolve_sources(args)

    results: list[_Result] = []
    with cli_metrics_run():
        for source in sources:
            logger.info(f"开始检查 {source.value} …")
            for probe in _PROBE_BUILDERS[source](args.code):
                r = _run_probe(source.value, probe)
                results.append(r)
                # 每个 probe 结果 → 两条 counter (latency + status int)
                probe_scope = scope(Metric.Domain.HEALTHCHECK, source.value, probe.method)
                record_counter(
                    probe_scope,
                    Metric.Counter.LATENCY_MS,
                    float(r.latency_ms),
                    status=r.status.value,
                    rows=r.rows_returned,
                )
                record_counter(
                    probe_scope,
                    Metric.Counter.OK,
                    1.0 if r.status == HealthStatus.OK else 0.0,
                    error=r.error_message,
                )

    if OutputFormat(args.output) == OutputFormat.JSON:
        _print_json(results)
    else:
        _print_table(results)

    fails = [r for r in results if r.status == HealthStatus.FAIL]
    if fails:
        _notify_failures(fails, total=len(results))
        raise SystemExit(1)


def _notify_failures(fails: list[_Result], total: int) -> None:
    """推送告警 — 列出前 5 个失败的 source/method/error 摘要。"""
    body_lines = [f"{r.source}/{r.method}: {(r.error_message or '?')[:80]}" for r in fails[:5]]
    if len(fails) > 5:
        body_lines.append(f"...还有 {len(fails) - 5} 个")
    notify(f"⚠️ Healthcheck 失败 {len(fails)}/{total}", "\n".join(body_lines))


def _resolve_sources(args) -> list[StrEnum]:
    """从 --source 解析最终要检查的源列表。"""
    if args.source == "all":
        return DataSourceKind.all()
    return [DataSourceKind.from_value(args.source)]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FloatShare 数据源健康度检查")
    p.add_argument(
        "--source",
        choices=[*(s.value for s in DataSourceKind.all()), "all"],
        default="all",
        help="单个源或 all (默认 all)",
    )
    p.add_argument(
        "--code",
        default="600000.SH",
        help="测试用股票代码（默认 600000.SH）",
    )
    p.add_argument(
        "--output",
        choices=[f.value for f in OutputFormat],
        default=OutputFormat.TABLE.value,
        help="输出格式",
    )
    return p


if __name__ == "__main__":
    main()
