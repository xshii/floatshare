"""枚举常量 — 替换全部魔法字符串。"""

from __future__ import annotations

from enum import Enum, StrEnum


class AdjustType(StrEnum):
    """复权类型。"""

    NONE = ""
    QFQ = "qfq"  # 前复权
    HFQ = "hfq"  # 后复权


class ExchangeSuffix(StrEnum):
    """tushare/AKShare 证券代码后缀 (`600000.SH` 的 `.SH` 部分)."""

    SH = ".SH"  # 上海证券交易所
    SZ = ".SZ"  # 深圳证券交易所
    BJ = ".BJ"  # 北京证券交易所


class MaskReason(StrEnum):
    """audit_mask 当日不可交易/不可训练的原因."""

    ST = "ST"  # 名称含 ST / *ST — 涨跌停规则不同
    EX_DIVIDEND = "ex_dividend"  # 除权日 — raw close 有伪跳空
    SUSPENDED = "suspended"  # 停牌 — panel 里不存在


class PipelineStage(StrEnum):
    """pipeline stage — 数字=执行阶段, 字母=同阶段内可并行.

    依赖图:
        S1 → {S2a, S2b}      (S2a db_integrity 与 S2b prep_features 并行, 都仅依赖 S1)
        S1c (并行 S1)         (cctv_news 20:00+ 发布, 无 pipeline 内依赖, SOFT — 失败 → news 特征 0)
        S2b → {S3a, S3b}     (S3a tushare_check 与 S3b feature_audit 并行, 都依赖 S2b feats)
        S2b → S4 → S5         (S4 train 依赖 S2b feats; S5 recommend 依赖 S4 ckpt)
    """

    # --- Evening phase (T 17:00 起) ---
    S1_SYNC = "S1_sync"  # 行情 sync
    S1C_NEWS_INGEST = "S1C_news_ingest"  # cctv_news ingest (并行 S1, T 20:00+ tushare 发布)
    S2A_DB_INTEGRITY = "S2A_db_integrity"  # DB 完整性 (读 raw_daily)
    S2B_PREP_FEATURES = "S2B_prep_features"  # compute_features (并行 S2A)
    S3A_TUSHARE_CHECK = "S3A_tushare_check"  # RSI/KDJ/MACD 对拍业界
    S3B_FEATURE_AUDIT = "S3B_feature_audit"  # rolling 252 winsorize (并行 S3A)
    S4_TRAIN = "S4_train"  # warm-start 训练 (跨夜)
    # --- Morning phase (T+1 07:00) ---
    S5_RECOMMEND = "S5_recommend"  # best ckpt 预测 top-K


class StageStatus(StrEnum):
    """pipeline stage / 测试任务的成功标志 — 替代散落的 "OK"/"FAIL"."""

    OK = "OK"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"  # 幂等跳过 (already_ran_today 命中)
    GATE_BLOCKED = "GATE_BLOCKED"  # 前置 gate 不满足 (非自身执行失败)


class FailPolicy(StrEnum):
    """Stage 失败时的处理策略 (runner.py 使用)."""

    FAST = "fast"  # 失败 → 整条 pipeline abort (S1/S2 数据层默认)
    SOFT = "soft"  # 失败 → 记 FAIL 继续 (S3/S4/S5 模型层默认)


class Direction(StrEnum):
    """交易方向。"""

    BUY = "buy"
    SELL = "sell"


class DataSourceKind:
    """数据源 — 嵌套 namespace 表达"分桶 + 名称"两级。

    访问: DataSourceKind.PAID_REMOTE.TUSHARE → "tushare"
    遍历分桶: list(DataSourceKind.FREE_REMOTE) → [AKSHARE, EASTMONEY]
    遍历全部: DataSourceKind.all()
    反查: DataSourceKind.from_value("tushare")
    """

    class PAID_REMOTE(StrEnum):
        TUSHARE = "tushare"

    class FREE_REMOTE(StrEnum):
        AKSHARE = "akshare"
        EASTMONEY = "eastmoney"

    class LOCAL_PERSIST(StrEnum):
        LOCALDB = "localdb"

    class LOCAL_CACHE(StrEnum):
        CACHED = "cached"

    @classmethod
    def _groups(cls) -> tuple[type[StrEnum], ...]:
        return (cls.PAID_REMOTE, cls.FREE_REMOTE, cls.LOCAL_PERSIST, cls.LOCAL_CACHE)

    @classmethod
    def all(cls) -> list[StrEnum]:
        return [m for grp in cls._groups() for m in grp]

    @classmethod
    def from_value(cls, s: str) -> StrEnum:
        for grp in cls._groups():
            try:
                return grp(s)
            except ValueError:
                continue
        raise ValueError(f"unknown DataSourceKind: {s}")


class DataKind:
    """同步的数据类型 — 嵌套 namespace 表达"分类 + 类型"两级。

    访问: DataKind.DAILY.RAW_DAILY → "raw_daily"
    遍历分类: list(DataKind.DAILY)
    遍历全部: DataKind.all() (按 sync 推荐顺序)
    反查: DataKind.from_value("raw_daily")
    """

    class REFERENCE(StrEnum):
        """准静态参考表 (年频/事件触发)。"""

        LIFECYCLE = "lifecycle"
        INDEX_WEIGHT = "index_weight"  # 指数成分股权重 (沪深300/中证500/...)
        INDUSTRY = "industry"  # 申万行业分类 + 个股归属
        CONCEPT = "concept"  # 同花顺/Tushare 概念板块 + 成分股 (多对多)

    class DAILY(StrEnum):
        """日频每股 1 行。"""

        RAW_DAILY = "raw_daily"
        ADJ_FACTOR = "adj_factor"
        DAILY_BASIC = "daily_basic"  # PE/PB/PS/股息/总市值/换手率
        CHIP_PERF = "chip_perf"
        MONEYFLOW = "moneyflow"  # 个股大单资金流
        MARGIN_DETAIL = "margin_detail"  # 个股融资融券明细

    class INTRADAY_HEAVY(StrEnum):
        """日频但量大 (每股每天 ~50 价位)。"""

        CHIP_DIST = "chip_dist"

    class FUNDAMENTAL(StrEnum):
        """季频财务。"""

        INCOME = "income"  # 完整利润表 (~30 字段)
        BALANCESHEET = "balancesheet"  # 资产负债表
        CASHFLOW = "cashflow"  # 现金流量表
        FINA_INDICATOR = "fina_indicator"  # 财务衍生指标 (ROE/毛利率/YoY 等)
        HOLDER_NUMBER = "stk_holder_number"  # 股东户数 (筹码集中度)
        FORECAST = "forecast"  # 券商盈利预测

    class MARKET(StrEnum):
        """市场层级 / 宏观 (非个股)。"""

        MONEYFLOW_HSGT = "moneyflow_hsgt"  # 沪深港通北向/南向资金
        INDEX_DAILY = "index_daily"  # 指数日线 OHLCV (风格代理)
        CN_CPI = "cn_cpi"  # 居民消费价格指数 (月频)
        CN_PPI = "cn_ppi"  # 工业生产者出厂价格指数 (月频)
        SHIBOR = "shibor"  # 上海银行间同业拆借利率 (日频)
        FX_DAILY = "fx_daily"  # 外汇日行情 (日频)

    class EVENT(StrEnum):
        """离散事件 (月度/公告触发)。"""

        BROKER_PICKS = "broker_picks"
        DIVIDEND = "dividend"  # 分红送股明细 (按公告)
        TOP_LIST = "top_list"  # 龙虎榜每日个股
        TOP_INST = "top_inst"  # 龙虎榜机构席位明细

    @classmethod
    def _groups(cls) -> tuple[type[StrEnum], ...]:
        return (
            cls.REFERENCE,
            cls.DAILY,
            cls.FUNDAMENTAL,
            cls.MARKET,
            cls.INTRADAY_HEAVY,
            cls.EVENT,
        )

    @classmethod
    def all(cls) -> list[StrEnum]:
        return [m for grp in cls._groups() for m in grp]

    @classmethod
    def from_value(cls, s: str) -> StrEnum:
        for grp in cls._groups():
            try:
                return grp(s)
            except ValueError:
                continue
        raise ValueError(f"unknown DataKind: {s}")


class HealthStatus(StrEnum):
    """健康检查 probe 结果。"""

    OK = "OK"
    FAIL = "FAIL"


class OutputFormat(StrEnum):
    """CLI 输出格式。"""

    TABLE = "table"
    JSON = "json"


class ListStatus(StrEnum):
    """A 股个股生命周期状态 (Tushare list_status 字段)。"""

    LISTED = "L"  # 上市中
    DELISTED = "D"  # 已退市
    PAUSED = "P"  # 暂停上市


class TxnType(StrEnum):
    """资金流水类型。"""

    DEPOSIT = "deposit"  # 存钱 (外部 → 活期)
    WITHDRAW = "withdraw"  # 取钱 (活期 → 外部)
    DCA_BUY = "dca_buy"  # 定投扣款 (活期 → 标的持仓)


class DcaFrequency(StrEnum):
    """定投频率。"""

    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"


class PlanStatus(StrEnum):
    """定投计划状态。"""

    ACTIVE = "active"  # 执行中
    PAUSED = "paused"  # 暂停 (next_run_date 不推进)
    STOPPED = "stopped"  # 已终止 (不再执行)


# 非字符串语义的纯枚举
class TimeFrame(Enum):
    """K 线频次。"""

    DAY = "D"
    WEEK = "W"
    MONTH = "M"
    MIN_1 = "1min"
    MIN_5 = "5min"
    MIN_15 = "15min"
    MIN_30 = "30min"
    MIN_60 = "60min"
