"""Tushare 适配器 — token 通过构造参数或环境变量注入，失效时自动刷新。"""

from __future__ import annotations

import contextlib
import functools
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import requests

from floatshare.domain.enums import AdjustType, ListStatus, TimeFrame
from floatshare.domain.schema import normalize_ohlcv
from floatshare.interfaces.data_source import DataSourceError
from floatshare.observability import logger

if TYPE_CHECKING:
    from typing import Any

_MAX_REFRESH_RETRIES = 2

# Token 持久化路径 — 启动时优先读，refresh 后写入；不污染 .env
_TOKEN_CACHE_PATH = Path.home() / ".floatshare" / "tushare_token"


# === Spec-driven fetch — 所有 fetch 方法共用模板 ==============================


@dataclass(frozen=True, slots=True)
class _FetchSpec:
    """Tushare API 的返回处理配置 (调用参数交给具体方法构造)。"""

    api: str  # tushare pro 方法名
    rename: dict[str, str] | None = None  # 列重命名 (默认无)
    date_col: str | None = "trade_date"  # 主日期列 (None 表示无日期列)
    sort_cols: tuple[str, ...] = ("trade_date",)  # 排序键
    select: tuple[str, ...] | None = None  # 投影特定列
    is_ohlcv: bool = False  # 是否走 OHLCV schema 标准化


_RENAME_OHLCV = {"ts_code": "code", "vol": "volume"}
_RENAME_CODE = {"ts_code": "code"}

# _SPECS key 常量 — 避免 "income_bulk" 这类魔鬼字符串在 _SPECS + get_*_bulk 两处各写一遍
_SPEC_INCOME_BULK = "income_bulk"
_SPEC_BALANCESHEET_BULK = "balancesheet_bulk"
_SPEC_CASHFLOW_BULK = "cashflow_bulk"
_SPEC_FINA_INDICATOR_BULK = "fina_indicator_bulk"
_SPEC_HOLDER_NUMBER_BULK = "holder_number_bulk"
_SPEC_DIVIDEND_BULK = "dividend_bulk"

# tushare API 方法名常量 — 同一个 API 在 per-code 和 bulk spec 被引用 2 次, 抽常量避免漂移
_API_HOLDER_NUMBER = "stk_holdernumber"  # 无 _vip 后缀, 本体支持 ann_date/start_date/end_date
_API_DIVIDEND = "dividend"  # 同上, 本体支持 ann_date 参数

_SPECS: dict[str, _FetchSpec] = {
    # 行情
    "raw_daily": _FetchSpec("daily", _RENAME_OHLCV, is_ohlcv=True),
    "adj_factor": _FetchSpec(
        "adj_factor", _RENAME_CODE, select=("code", "trade_date", "adj_factor")
    ),
    "index_daily": _FetchSpec("index_daily", _RENAME_OHLCV, is_ohlcv=True),
    # 申万行业指数走专用接口 (`.SI` 后缀)，字段大致同 index_daily
    "sw_daily": _FetchSpec("sw_daily", _RENAME_OHLCV, is_ohlcv=True),
    "daily_basic": _FetchSpec("daily_basic", _RENAME_CODE),
    # 筹码
    "chip_perf": _FetchSpec("cyq_perf", _RENAME_CODE),
    "chip_dist": _FetchSpec("cyq_chips", _RENAME_CODE, sort_cols=("trade_date", "price")),
    # 资金流
    "moneyflow": _FetchSpec("moneyflow", _RENAME_CODE),
    "moneyflow_hsgt": _FetchSpec("moneyflow_hsgt"),  # 市场级别，无 ts_code 字段
    # 财务
    "income": _FetchSpec("income", _RENAME_CODE, date_col="end_date", sort_cols=("end_date",)),
    "balancesheet": _FetchSpec(
        "balancesheet", _RENAME_CODE, date_col="end_date", sort_cols=("end_date",)
    ),
    "cashflow": _FetchSpec("cashflow", _RENAME_CODE, date_col="end_date", sort_cols=("end_date",)),
    "forecast": _FetchSpec(
        "report_rc", _RENAME_CODE, date_col="report_date", sort_cols=("report_date",)
    ),
    # 参考
    "index_weight": _FetchSpec("index_weight", sort_cols=("trade_date", "con_code")),
    # Wave A — 扩展基本面/两融/事件
    "fina_indicator": _FetchSpec(
        "fina_indicator", _RENAME_CODE, date_col="end_date", sort_cols=("end_date",)
    ),
    "holder_number": _FetchSpec(
        _API_HOLDER_NUMBER, _RENAME_CODE, date_col="end_date", sort_cols=("end_date",)
    ),
    "dividend": _FetchSpec(
        _API_DIVIDEND, _RENAME_CODE, date_col="end_date", sort_cols=("end_date", "div_proc")
    ),
    "margin_detail": _FetchSpec("margin_detail", _RENAME_CODE),
    "top_list": _FetchSpec("top_list", _RENAME_CODE, sort_cols=("trade_date", "code")),
    "top_inst": _FetchSpec("top_inst", _RENAME_CODE, sort_cols=("trade_date", "code", "exalter")),
    # Wave B — 宏观数据
    # CPI/PPI 的 month 列是 YYYYMM 字符串，跳过 date_col 自动解析
    "cn_cpi": _FetchSpec("cn_cpi", date_col=None, sort_cols=("month",)),
    "cn_ppi": _FetchSpec("cn_ppi", date_col=None, sort_cols=("month",)),
    "shibor": _FetchSpec(
        "shibor",
        rename={
            "on": "overnight",
            "1w": "w1",
            "2w": "w2",
            "1m": "m1",
            "3m": "m3",
            "6m": "m6",
            "9m": "m9",
            "1y": "y1",
        },
        date_col="date",
        sort_cols=("date",),
    ),
    "fx_daily": _FetchSpec("fx_daily", _RENAME_CODE, sort_cols=("trade_date", "code")),
    # Wave C — 新闻联播 (T 日 19:30 后由 tushare 入库, ~20:15-20:45 可拉)
    "cctv_news": _FetchSpec("cctv_news", date_col="date", sort_cols=("date",)),
    # === Bulk VIP 接口 — 按 ann_date 一次拉全市场 (财务表加速) ===
    # tushare VIP 专有, 需 ≥5000 积分, 免除 per-code 循环的 5000+ API 调用
    _SPEC_INCOME_BULK: _FetchSpec(
        "income_vip", _RENAME_CODE, date_col="end_date", sort_cols=("end_date",)
    ),
    _SPEC_BALANCESHEET_BULK: _FetchSpec(
        "balancesheet_vip", _RENAME_CODE, date_col="end_date", sort_cols=("end_date",)
    ),
    _SPEC_CASHFLOW_BULK: _FetchSpec(
        "cashflow_vip", _RENAME_CODE, date_col="end_date", sort_cols=("end_date",)
    ),
    _SPEC_FINA_INDICATOR_BULK: _FetchSpec(
        "fina_indicator_vip", _RENAME_CODE, date_col="end_date", sort_cols=("end_date",)
    ),
    # stk_holdernumber / dividend 本体就支持 ann_date, 跟 per-code spec 共享 API 常量
    _SPEC_HOLDER_NUMBER_BULK: _FetchSpec(
        _API_HOLDER_NUMBER, _RENAME_CODE, date_col="end_date", sort_cols=("end_date",)
    ),
    _SPEC_DIVIDEND_BULK: _FetchSpec(
        _API_DIVIDEND, _RENAME_CODE, date_col="end_date", sort_cols=("end_date", "div_proc")
    ),
}


def _auto_refresh(method: Callable[..., Any]) -> Callable[..., Any]:
    """装饰器：tushare API 调用失败且判定为 token 失效时，自动刷新重试。"""

    @functools.wraps(method)
    def wrapper(self: TushareSource, *args: Any, **kwargs: Any) -> Any:
        last_exc: Exception | None = None
        for attempt in range(_MAX_REFRESH_RETRIES + 1):
            try:
                return method(self, *args, **kwargs)
            except Exception as exc:
                if not _is_token_error(exc):
                    raise
                last_exc = exc
                if attempt >= _MAX_REFRESH_RETRIES:
                    break
                logger.warning(f"Tushare token 可能失效 (第 {attempt + 1} 次)，尝试刷新 …")
                try:
                    self.token = self._refresh_token()
                    self._reset_pro()
                except DataSourceError:
                    raise DataSourceError(
                        f"Tushare token 失效且刷新失败，原始错误: {last_exc}"
                    ) from last_exc
        raise DataSourceError(
            f"Tushare token 刷新 {_MAX_REFRESH_RETRIES} 次后仍失败: {last_exc}。"
            f"提示: 刷新服务可能返回了无效 token，建议检查 TUSHARE_TOKEN_REFRESH_KEY 配置，"
            f"或直接到 https://tushare.pro/user/token 获取有效 token 更新 .env"
        )

    return wrapper


class TushareSource:
    """Tushare Pro 数据源，token 失效时自动刷新（最多重试 2 次）。"""

    def __init__(
        self,
        token: str | None = None,
        refresh_url: str | None = None,
        refresh_key: str | None = None,
    ) -> None:
        # Token 优先级: 显式参数 > 本地 cache (上次 refresh 的) > 环境变量
        self.token: str | None = token or _load_cached_token() or os.getenv("TUSHARE_TOKEN")
        self._refresh_url: str = refresh_url or os.environ.get(
            "TUSHARE_TOKEN_REFRESH_URL",
            "https://token.jingjingtech.com/api/v1/gtst",
        )
        self._refresh_key: str | None = refresh_key or os.getenv("TUSHARE_TOKEN_REFRESH_KEY")
        self._pro: Any = None

    def _refresh_token(self) -> str:
        """调用刷新接口获取新 token，失败时抛出 DataSourceError。"""
        if not self._refresh_key:
            raise DataSourceError("无法刷新 token：缺少 TUSHARE_TOKEN_REFRESH_KEY 环境变量")
        logger.info("正在刷新 Tushare token …")
        try:
            resp = requests.get(
                self._refresh_url,
                params={"k": self._refresh_key},
                timeout=10,
            )
            resp.raise_for_status()
            raw = resp.text.strip()
        except requests.RequestException as exc:
            raise DataSourceError(f"Tushare token 刷新请求失败: {exc}") from exc

        new_token = _extract_token(raw)
        if not new_token:
            raise DataSourceError(f"Tushare token 刷新响应无法解析: {raw[:100]}")

        # 持久化 + 同步进程内 env，避免下次启动重复刷新
        _save_cached_token(new_token)
        os.environ["TUSHARE_TOKEN"] = new_token

        logger.info(f"Tushare token 刷新成功，已写入 {_TOKEN_CACHE_PATH}")
        return new_token

    def _reset_pro(self) -> None:
        """用当前 token 重建 pro_api 连接。

        注意: 不能用 `ts.set_token` + 无参 `pro_api()` — Tushare 的 get_token()
        优先读环境变量 TUSHARE_TOKEN，导致 set_token 写入的新 token 被旧的
        环境变量覆盖。直接 `pro_api(token=...)` 绕过这条读取链。
        """
        try:
            import tushare as ts  # pyright: ignore[reportMissingImports]
        except ImportError as exc:  # pragma: no cover
            raise DataSourceError("请先安装 tushare: pip install tushare") from exc
        self._pro = ts.pro_api(token=self.token or "")

    @property
    def pro(self) -> Any:
        if self._pro is None:
            try:
                import tushare as ts  # pyright: ignore[reportMissingImports]  # 可选依赖
            except ImportError as exc:  # pragma: no cover
                raise DataSourceError("请先安装 tushare: pip install tushare") from exc
            if not self.token:
                raise DataSourceError("Tushare 缺少 token，请设置 TUSHARE_TOKEN 环境变量")
            self._pro = ts.pro_api(token=self.token)
        return self._pro

    # === Spec-driven fetch ==================================================

    def _fetch(self, spec_key: str, **api_kwargs: Any) -> pd.DataFrame:
        """通用 fetch 模板 — api_kwargs 直接透传给 Tushare API。

        步骤: 调用 → rename → 解析日期 → 投影 → 排序 → (可选)OHLCV normalize
        """
        spec = _SPECS[spec_key]
        df = getattr(self.pro, spec.api)(**api_kwargs)
        if df.empty:
            return df
        if spec.rename:
            df = df.rename(columns=spec.rename)
        if spec.date_col and spec.date_col in df.columns:
            df[spec.date_col] = pd.to_datetime(df[spec.date_col])
        if spec.select:
            df = df[list(spec.select)]
        if spec.sort_cols and all(c in df.columns for c in spec.sort_cols):
            df = df.sort_values(list(spec.sort_cols)).reset_index(drop=True)
        return normalize_ohlcv(df) if spec.is_ohlcv else df

    @staticmethod
    def _date_range_kwargs(code: str, start: date | None, end: date | None) -> dict[str, Any]:
        """构造 Tushare 标准 (ts_code, start_date, end_date) kwargs。"""
        return {
            "ts_code": code,
            "start_date": _fmt_date(start),
            "end_date": _fmt_date(end),
        }

    # --- 公开接口 — 按 Protocol 分组 -----------------------------------------

    @_auto_refresh
    def get_stock_list(self) -> pd.DataFrame:
        df = self.pro.stock_basic(
            exchange="",
            list_status="L",
            fields="ts_code,symbol,name,area,industry,market,list_date",
        )
        return df.rename(columns={"ts_code": "code", "symbol": "ticker"})

    @_auto_refresh
    def get_industry(self) -> pd.DataFrame:
        """申万行业分类映射 (code → L1/L2/L3)。

        注意: Tushare 的 parent_code 存的是父级 `industry_code` (6 位数字代码),
        **不是** `index_code` (SW2021 的 .SI 指数代码)，需要用 industry_code 做 join。
        """
        from collections import namedtuple

        l1 = self.pro.index_classify(level="L1", src="SW2021")
        l2 = self.pro.index_classify(level="L2", src="SW2021")
        l3 = self.pro.index_classify(level="L3", src="SW2021")
        if l3.empty:
            return pd.DataFrame()

        L1Node = namedtuple("L1Node", ["code", "name"])
        L2Node = namedtuple("L2Node", ["code", "name", "l1_icode"])
        MISSING_L1 = L1Node(None, None)
        MISSING_L2 = L2Node(None, None, None)

        l1_by_icode = {
            r.industry_code: L1Node(r.index_code, r.industry_name) for r in l1.itertuples()
        }
        l2_by_icode = {
            r.industry_code: L2Node(r.index_code, r.industry_name, r.parent_code)
            for r in l2.itertuples()
        }

        frames: list[pd.DataFrame] = []
        for row in l3.itertuples():
            members = self.pro.index_member_all(l3_code=row.index_code)
            if members.empty:
                continue
            l2_node = l2_by_icode.get(row.parent_code, MISSING_L2)
            l1_node = (
                l1_by_icode.get(l2_node.l1_icode, MISSING_L1) if l2_node.l1_icode else MISSING_L1
            )

            sub = pd.DataFrame({"code": members["ts_code"]})
            sub["l3_code"] = row.index_code
            sub["l3_name"] = row.industry_name
            sub["l2_code"] = l2_node.code
            sub["l2_name"] = l2_node.name
            sub["l1_code"] = l1_node.code
            sub["l1_name"] = l1_node.name
            frames.append(sub)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).drop_duplicates(
            subset=["code"],
            keep="last",
        )

    @_auto_refresh
    def get_lifecycle(self) -> pd.DataFrame:
        """全 A 股生命周期表（上市/退市/暂停三状态合并）。"""
        fields = "ts_code,name,market,industry,list_date,delist_date"
        frames = []
        for status in ListStatus:
            df = self.pro.stock_basic(
                exchange="",
                list_status=status.value,
                fields=fields,
            )
            if df.empty:
                continue
            df = df.rename(columns={"ts_code": "code"})
            df["list_status"] = status.value
            df["list_date"] = _normalize_date_col(df.get("list_date"))
            df["delist_date"] = _normalize_date_col(df.get("delist_date"))
            frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # --- by-trade-date 全市场接口 (1 次 API 拿当天 5500 票) ------------------
    # daily-sync 走这一组, 比 per-code 提速 ~5500x
    # Tushare 这些 API 都接受 trade_date='YYYYMMDD' 单参数返当天全市场 snapshot

    @_auto_refresh
    def get_raw_daily_by_date(self, trade_date: date) -> pd.DataFrame:
        return self._fetch("raw_daily", trade_date=_fmt_date(trade_date))

    @_auto_refresh
    def get_daily_basic_by_date(self, trade_date: date) -> pd.DataFrame:
        return self._fetch("daily_basic", trade_date=_fmt_date(trade_date))

    @_auto_refresh
    def get_moneyflow_by_date(self, trade_date: date) -> pd.DataFrame:
        return self._fetch("moneyflow", trade_date=_fmt_date(trade_date))

    @_auto_refresh
    def get_adj_factor_by_date(self, trade_date: date) -> pd.DataFrame:
        return self._fetch("adj_factor", trade_date=_fmt_date(trade_date))

    @_auto_refresh
    def get_margin_detail_by_date(self, trade_date: date) -> pd.DataFrame:
        return self._fetch("margin_detail", trade_date=_fmt_date(trade_date))

    @_auto_refresh
    def get_chip_perf_by_date(self, trade_date: date) -> pd.DataFrame:
        return self._fetch("chip_perf", trade_date=_fmt_date(trade_date))

    @_auto_refresh
    def get_cctv_news_by_date(self, trade_date: date) -> pd.DataFrame:
        """新闻联播文字稿 (按章节分段返回: title + content).

        时效: T 日 19:30 联播结束, tushare 入库 ~20:15-20:45. 20:00 调用可能空.
        Caller 应带 retry (失败 → 15/30/45 min 后再试).
        """
        return self._fetch("cctv_news", date=_fmt_date(trade_date))

    @_auto_refresh
    def get_dividend_by_ex_date(self, ex_date: date) -> pd.DataFrame:
        """某天发生除权除息的全市场股票 — 用于 adj_factor 智能修正。

        返回字段含 ts_code / ex_date / div_proc / cash_div_tax 等。
        除权当天的所有受影响 codes 必须 full-refresh adj_factor 历史。
        """
        return self._fetch("dividend", ex_date=_fmt_date(ex_date))

    # --- date-range 个股接口 (统一 ts_code/start_date/end_date 三参) ----------

    @_auto_refresh
    def get_raw_daily(
        self, code: str, start: date | None = None, end: date | None = None
    ) -> pd.DataFrame:
        return self._fetch("raw_daily", **self._date_range_kwargs(code, start, end))

    @_auto_refresh
    def get_adj_factor(
        self, code: str, start: date | None = None, end: date | None = None
    ) -> pd.DataFrame:
        return self._fetch("adj_factor", **self._date_range_kwargs(code, start, end))

    @_auto_refresh
    def get_index_daily(
        self, code: str, start: date | None = None, end: date | None = None
    ) -> pd.DataFrame:
        # 申万指数 (.SI 后缀) 走 sw_daily 接口，其它走 index_daily
        spec = "sw_daily" if code.endswith(".SI") else "index_daily"
        return self._fetch(spec, **self._date_range_kwargs(code, start, end))

    @_auto_refresh
    def get_chip_perf(
        self, code: str, start: date | None = None, end: date | None = None
    ) -> pd.DataFrame:
        return self._fetch("chip_perf", **self._date_range_kwargs(code, start, end))

    @_auto_refresh
    def get_chip_dist(
        self, code: str, start: date | None = None, end: date | None = None
    ) -> pd.DataFrame:
        return self._fetch("chip_dist", **self._date_range_kwargs(code, start, end))

    @_auto_refresh
    def get_earnings_forecast(
        self, code: str, start: date | None = None, end: date | None = None
    ) -> pd.DataFrame:
        df = self._fetch("forecast", **self._date_range_kwargs(code, start, end))
        # Tushare 部分研报的 org_name / author_name 为 null，而这两列在
        # EarningsForecast.PK 里 (NOT NULL)。用占位符避免 IntegrityError。
        if not df.empty:
            for col in ("org_name", "author_name"):
                if col in df.columns:
                    df[col] = df[col].fillna("-")
        return df

    # --- P0/P1 新增 ----------------------------------------------------------

    @_auto_refresh
    def get_daily_basic(
        self, code: str, start: date | None = None, end: date | None = None
    ) -> pd.DataFrame:
        """每日基本面 (PE/PB/PS/股息/总市值/换手率)。"""
        return self._fetch("daily_basic", **self._date_range_kwargs(code, start, end))

    @_auto_refresh
    def get_moneyflow(
        self, code: str, start: date | None = None, end: date | None = None
    ) -> pd.DataFrame:
        """个股资金流向 (大单流入流出)。"""
        return self._fetch("moneyflow", **self._date_range_kwargs(code, start, end))

    @_auto_refresh
    def get_moneyflow_hsgt(
        self, start: date | None = None, end: date | None = None
    ) -> pd.DataFrame:
        """沪深港通北向/南向资金 (市场级别，无 code 参数)。"""
        return self._fetch(
            "moneyflow_hsgt",
            start_date=_fmt_date(start),
            end_date=_fmt_date(end),
        )

    @_auto_refresh
    def get_index_weight(
        self,
        index_code: str,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        """指数成分股权重 (Tushare 用 index_code 而非 ts_code)。"""
        return self._fetch(
            "index_weight",
            index_code=index_code,
            start_date=_fmt_date(start),
            end_date=_fmt_date(end),
        )

    @_auto_refresh
    def get_income(self, code: str) -> pd.DataFrame:
        """完整利润表 (Tushare 一次返全历史)。"""
        return self._fetch("income", ts_code=code)

    @_auto_refresh
    def get_balancesheet(self, code: str) -> pd.DataFrame:
        """资产负债表 (Tushare 一次返全历史)。"""
        return self._fetch("balancesheet", ts_code=code)

    @_auto_refresh
    def get_cashflow(self, code: str) -> pd.DataFrame:
        """现金流量表 (Tushare 一次返全历史)。"""
        return self._fetch("cashflow", ts_code=code)

    # --- Wave A 扩展 ---------------------------------------------------------

    @_auto_refresh
    def get_fina_indicator(self, code: str) -> pd.DataFrame:
        """财务衍生指标 (ROE/毛利率/YoY 等，一次返全史)。"""
        return self._fetch("fina_indicator", ts_code=code)

    @_auto_refresh
    def get_holder_number(self, code: str) -> pd.DataFrame:
        """股东户数 (按公告，一次返全史)。"""
        return self._fetch("holder_number", ts_code=code)

    @_auto_refresh
    def get_dividend(self, code: str) -> pd.DataFrame:
        """分红送股明细 (按公告事件，一次返全史)。"""
        return self._fetch("dividend", ts_code=code)

    # --- Bulk (VIP) 接口 — 按 ann_date 拉全市场, 1 call vs per-code 5500 call ---
    # 适用于 daily-sync 场景, 季度末披露日期前后的增量同步.
    # 注: 首次回补历史可用 period=YYYYMMDD (按 end_date) 会更完整.

    def _bulk_fetch_by_ann_date(
        self, spec_key: str, start: date | None, end: date | None
    ) -> pd.DataFrame:
        """按 ann_date 单天循环拉 — tushare VIP 的 start/end 区间参数在某些 API 上坏掉
        (fina_indicator_vip 传 start_date+end_date 返 0 行, 传 ann_date 正常), 按天迭代最稳.

        start=end=None 时退化为"今天", 调用方应给具体区间.
        """
        if start is None or end is None:
            return self._fetch(spec_key)  # 不带 date, 让 API 按 default 行为 (一般是最近一天)
        frames: list[pd.DataFrame] = []
        cur = start
        while cur <= end:
            df = self._fetch(spec_key, ann_date=_fmt_date(cur))
            if not df.empty:
                frames.append(df)
            cur += timedelta(days=1)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    @_auto_refresh
    def get_income_bulk(self, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        """全市场利润表, 按 ann_date 区间 (income_vip)."""
        return self._bulk_fetch_by_ann_date(_SPEC_INCOME_BULK, start, end)

    @_auto_refresh
    def get_balancesheet_bulk(
        self, start: date | None = None, end: date | None = None
    ) -> pd.DataFrame:
        """全市场资产负债表, 按 ann_date 区间 (balancesheet_vip)."""
        return self._bulk_fetch_by_ann_date(_SPEC_BALANCESHEET_BULK, start, end)

    @_auto_refresh
    def get_cashflow_bulk(self, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        """全市场现金流量表, 按 ann_date 区间 (cashflow_vip)."""
        return self._bulk_fetch_by_ann_date(_SPEC_CASHFLOW_BULK, start, end)

    @_auto_refresh
    def get_fina_indicator_bulk(
        self, start: date | None = None, end: date | None = None
    ) -> pd.DataFrame:
        """全市场财务衍生指标, 按 ann_date 区间 (fina_indicator_vip)."""
        return self._bulk_fetch_by_ann_date(_SPEC_FINA_INDICATOR_BULK, start, end)

    @_auto_refresh
    def get_holder_number_bulk(
        self, start: date | None = None, end: date | None = None
    ) -> pd.DataFrame:
        """全市场股东户数, 按 ann_date 区间 (stk_holdernumber)."""
        return self._bulk_fetch_by_ann_date(_SPEC_HOLDER_NUMBER_BULK, start, end)

    @_auto_refresh
    def get_dividend_bulk(self, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        """全市场分红送股, 按 ann_date 区间 (dividend)."""
        return self._bulk_fetch_by_ann_date(_SPEC_DIVIDEND_BULK, start, end)

    @_auto_refresh
    def get_margin_detail(
        self,
        code: str,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        """个股两融明细 (日频)。"""
        return self._fetch("margin_detail", **self._date_range_kwargs(code, start, end))

    @_auto_refresh
    def get_top_list(self, trade_date: date) -> pd.DataFrame:
        """龙虎榜每日个股 (按交易日批量)。"""
        return self._fetch("top_list", trade_date=_fmt_date(trade_date))

    @_auto_refresh
    def get_top_inst(self, trade_date: date) -> pd.DataFrame:
        """龙虎榜机构席位明细 (按交易日批量)。"""
        df = self._fetch("top_inst", trade_date=_fmt_date(trade_date))
        # side 列 Tushare 返 int 0/1，统一 cast 到 str 避免 PK 类型歧义
        if not df.empty and "side" in df.columns:
            df["side"] = df["side"].astype(str)
        return df

    # --- Wave B 宏观 ---------------------------------------------------------

    @_auto_refresh
    def get_cn_cpi(self, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        """居民消费价格指数 (月频)。"""
        return self._fetch("cn_cpi", start_m=_fmt_month(start), end_m=_fmt_month(end))

    @_auto_refresh
    def get_cn_ppi(self, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        """工业生产者出厂价格指数 (月频)。"""
        return self._fetch("cn_ppi", start_m=_fmt_month(start), end_m=_fmt_month(end))

    @_auto_refresh
    def get_shibor(self, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        """SHIBOR 利率 (日频)。"""
        return self._fetch(
            "shibor",
            start_date=_fmt_date(start),
            end_date=_fmt_date(end),
        )

    @_auto_refresh
    def get_fx_daily(
        self,
        code: str = "USDCNY.FXCM",
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        """外汇日行情 (默认美元人民币)。"""
        return self._fetch(
            "fx_daily",
            ts_code=code,
            start_date=_fmt_date(start),
            end_date=_fmt_date(end),
        )

    # --- 概念板块 (同花顺优先, Tushare 自家概念后备) ---------------------------

    @_auto_refresh
    def get_concept_boards(self) -> pd.DataFrame:
        """概念板块清单。先试同花顺 (ths_index, type='N'), 失败回退 Tushare concept。"""
        try:
            df = self.pro.ths_index(exchange="A", type="N")
            if df.empty:
                raise DataSourceError("ths_index empty")
            return df.rename(
                columns={
                    "ts_code": "board_code",
                    "name": "board_name",
                    "count": "member_count",
                }
            )[["board_code", "board_name", "member_count", "list_date"]].assign(src="ths")
        except Exception as exc:
            logger.warning(f"ths_index 失败 ({exc}), 退回 Tushare concept(src='ts')")
            df = self.pro.concept(src="ts")
            if df.empty:
                return pd.DataFrame()
            return df.rename(columns={"code": "board_code", "name": "board_name"})[
                ["board_code", "board_name"]
            ].assign(src="ts")

    @_auto_refresh
    def get_concept_members(self, board_code: str) -> pd.DataFrame:
        """单个概念板块的成分股 (按 board_code 后缀自动选 ths/ts API)。"""
        if board_code.endswith(".TI"):
            # ths_member 列: ts_code(板块) / con_code(成分) / con_name
            df = self.pro.ths_member(ts_code=board_code)
            if df.empty:
                return pd.DataFrame()
            df = df.rename(columns={"con_code": "code", "con_name": "name"})
            df["board_code"] = board_code
            cols = [
                c for c in ("board_code", "code", "name", "weight", "in_date") if c in df.columns
            ]
            return df[cols]
        # Tushare 概念走 concept_detail
        df = self.pro.concept_detail(id=board_code)
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"id": "board_code", "ts_code": "code"})
        cols = [c for c in ("board_code", "code", "name", "in_date", "out_date") if c in df.columns]
        return df[cols]

    def get_daily(
        self,
        code: str,
        start: date | None = None,
        end: date | None = None,
        adj: AdjustType = AdjustType.QFQ,
    ) -> pd.DataFrame:
        df = self.get_raw_daily(code, start, end)
        if df.empty or adj == AdjustType.NONE:
            return df
        adj_df = self.get_adj_factor(code, start, end)
        if not adj_df.empty:
            df = df.merge(adj_df[["trade_date", "adj_factor"]], on="trade_date")
            df = self._apply_adj(df, adj)
        return df

    @staticmethod
    def _apply_adj(df: pd.DataFrame, adj: AdjustType) -> pd.DataFrame:
        if adj == AdjustType.QFQ:
            factor = df["adj_factor"] / df["adj_factor"].iloc[-1]
        elif adj == AdjustType.HFQ:
            factor = df["adj_factor"] / df["adj_factor"].iloc[0]
        else:
            return df
        for col in ("open", "high", "low", "close"):
            df[col] = df[col] * factor
        return df

    def get_minute(
        self,
        code: str,
        start: date | None = None,
        end: date | None = None,
        freq: TimeFrame = TimeFrame.MIN_5,
    ) -> pd.DataFrame:
        return pd.DataFrame()  # high tier only

    @_auto_refresh
    def get_trade_calendar(
        self,
        start: date | None = None,
        end: date | None = None,
    ) -> list[date]:
        df = self.pro.trade_cal(
            exchange="SSE",
            start_date=_fmt_date(start),
            end_date=_fmt_date(end),
            is_open=1,
        )
        return sorted(pd.to_datetime(df["cal_date"]).dt.date.tolist())

    @_auto_refresh
    def get_broker_picks(self, month: str) -> pd.DataFrame:
        """券商月度金股 (broker_recommend)。month 格式: 'YYYYMM'。"""
        df = self.pro.broker_recommend(month=month)
        return df.rename(columns={"ts_code": "code"}) if not df.empty else df


# === 模块级辅助函数 ===========================================================


def _load_cached_token() -> str | None:
    """从 ~/.floatshare/tushare_token 读取上次 refresh 的 token。"""
    try:
        text = _TOKEN_CACHE_PATH.read_text(encoding="utf-8").strip()
        return text or None
    except (OSError, FileNotFoundError):
        return None


def _save_cached_token(token: str) -> None:
    """持久化 token 到 ~/.floatshare/tushare_token，权限 0600。"""
    _TOKEN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TOKEN_CACHE_PATH.write_text(token, encoding="utf-8")
    # Windows 不支持 chmod，静默跳过
    with contextlib.suppress(OSError):
        _TOKEN_CACHE_PATH.chmod(0o600)  # 仅当前用户可读


def _fmt_date(d: date | None) -> str | None:
    """date → 'YYYYMMDD' 字符串，None 透传。"""
    return d.strftime("%Y%m%d") if d else None


def _fmt_month(d: date | None) -> str | None:
    """date → 'YYYYMM' 字符串，月频接口用，None 透传。"""
    return d.strftime("%Y%m") if d else None


def _normalize_date_col(s: pd.Series | None) -> pd.Series | None:
    """Tushare 返回的日期列是 'YYYYMMDD' 字符串，统一转 'YYYY-MM-DD'。"""
    if s is None:
        return None
    parsed = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), None)


def _is_token_error(exc: Exception) -> bool:
    """判断异常是否为 token 失效导致。"""
    msg = str(exc).lower()
    return any(
        keyword in msg
        for keyword in ("token", "权限", "认证", "auth", "credential", "抱歉，您没有访问该接口")
    )


def _extract_token(raw: str) -> str | None:
    """从纯文本响应提取 token（jingjingtech 接口格式）。

    判定条件: ≥20 字符的纯字母数字字符串（含下划线）。
    """
    raw = raw.strip()
    if len(raw) >= 20 and raw.replace("_", "").isalnum():
        return raw
    return None
