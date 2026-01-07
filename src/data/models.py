"""数据模型定义"""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Optional, Dict, Any


# ============================================================
# 枚举定义
# ============================================================


class Market(Enum):
    """交易市场"""

    # A股
    SH = "SH"  # 上海证券交易所
    SZ = "SZ"  # 深圳证券交易所
    BJ = "BJ"  # 北京证券交易所

    # 港股
    HK = "HK"  # 香港交易所

    # 美股
    NYSE = "NYSE"  # 纽约证券交易所
    NASDAQ = "NASDAQ"  # 纳斯达克
    AMEX = "AMEX"  # 美国证券交易所

    # 期货交易所
    SHFE = "SHFE"  # 上海期货交易所
    DCE = "DCE"  # 大连商品交易所
    CZCE = "CZCE"  # 郑州商品交易所
    CFFEX = "CFFEX"  # 中国金融期货交易所
    INE = "INE"  # 上海国际能源交易中心
    GFEX = "GFEX"  # 广州期货交易所

    # 债券市场
    IB = "IB"  # 银行间市场

    @classmethod
    def from_code(cls, code: str) -> "Market":
        """根据代码推断市场"""
        if "." in code:
            suffix = code.split(".")[-1].upper()
            try:
                return cls(suffix)
            except ValueError:
                pass

        # A股代码规则
        ticker = code.split(".")[0] if "." in code else code
        if ticker.startswith("6"):
            return cls.SH
        elif ticker.startswith(("0", "3")):
            return cls.SZ
        elif ticker.startswith(("4", "8")):
            return cls.BJ

        raise ValueError(f"无法识别市场: {code}")


class AssetType(Enum):
    """资产类型"""

    EQUITY = "equity"  # 股票
    BOND = "bond"  # 债券
    CONVERTIBLE = "convertible"  # 可转债
    FUTURE = "future"  # 期货
    OPTION = "option"  # 期权
    FUND = "fund"  # 基金
    ETF = "etf"  # ETF
    INDEX = "index"  # 指数
    REPO = "repo"  # 回购
    FOREX = "forex"  # 外汇


class Currency(Enum):
    """货币类型"""

    CNY = "CNY"  # 人民币
    HKD = "HKD"  # 港币
    USD = "USD"  # 美元
    EUR = "EUR"  # 欧元
    JPY = "JPY"  # 日元
    GBP = "GBP"  # 英镑


class Region(Enum):
    """地区/国家"""

    CN = "CN"  # 中国大陆
    HK = "HK"  # 香港
    US = "US"  # 美国
    EU = "EU"  # 欧洲
    JP = "JP"  # 日本
    UK = "UK"  # 英国


# 市场与货币映射
MARKET_CURRENCY: Dict[Market, Currency] = {
    Market.SH: Currency.CNY,
    Market.SZ: Currency.CNY,
    Market.BJ: Currency.CNY,
    Market.HK: Currency.HKD,
    Market.NYSE: Currency.USD,
    Market.NASDAQ: Currency.USD,
    Market.AMEX: Currency.USD,
    Market.SHFE: Currency.CNY,
    Market.DCE: Currency.CNY,
    Market.CZCE: Currency.CNY,
    Market.CFFEX: Currency.CNY,
    Market.INE: Currency.CNY,
    Market.GFEX: Currency.CNY,
    Market.IB: Currency.CNY,
}

# 市场与地区映射
MARKET_REGION: Dict[Market, Region] = {
    Market.SH: Region.CN,
    Market.SZ: Region.CN,
    Market.BJ: Region.CN,
    Market.HK: Region.HK,
    Market.NYSE: Region.US,
    Market.NASDAQ: Region.US,
    Market.AMEX: Region.US,
    Market.SHFE: Region.CN,
    Market.DCE: Region.CN,
    Market.CZCE: Region.CN,
    Market.CFFEX: Region.CN,
    Market.INE: Region.CN,
    Market.GFEX: Region.CN,
    Market.IB: Region.CN,
}


# ============================================================
# 金融工具基类
# ============================================================


@dataclass
class Instrument:
    """金融工具基类"""

    code: str  # 原始代码（不含市场后缀）
    name: str  # 名称
    market: Market  # 交易市场
    asset_type: AssetType = field(init=False)  # 资产类型（由子类设置）
    currency: Currency = Currency.CNY  # 交易货币
    list_date: Optional[date] = None  # 上市日期
    delist_date: Optional[date] = None  # 退市日期
    metadata: Dict[str, Any] = field(default_factory=dict)  # 扩展字段

    def __post_init__(self):
        """初始化后处理"""
        # 自动设置货币
        if self.currency == Currency.CNY and self.market in MARKET_CURRENCY:
            self.currency = MARKET_CURRENCY[self.market]

    @property
    def symbol(self) -> str:
        """统一标识符（如：000001.SZ）"""
        return f"{self.code}.{self.market.value}"

    @property
    def is_active(self) -> bool:
        """是否在市/可交易"""
        return self.delist_date is None

    @property
    def region(self) -> Region:
        """所属地区"""
        return MARKET_REGION.get(self.market, Region.CN)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "code": self.code,
            "name": self.name,
            "symbol": self.symbol,
            "market": self.market.value,
            "asset_type": self.asset_type.value,
            "currency": self.currency.value,
            "list_date": self.list_date.isoformat() if self.list_date else None,
            "delist_date": self.delist_date.isoformat() if self.delist_date else None,
            "is_active": self.is_active,
            "metadata": self.metadata,
        }


# ============================================================
# 具体资产类型
# ============================================================


@dataclass
class Equity(Instrument):
    """股票"""

    industry: Optional[str] = None  # 所属行业
    sector: Optional[str] = None  # 所属板块
    is_st: bool = False  # 是否ST
    is_kcb: bool = False  # 是否科创板
    is_cyb: bool = False  # 是否创业板

    def __post_init__(self):
        self.asset_type = AssetType.EQUITY
        super().__post_init__()

        # 根据代码判断板块
        if self.code.startswith("688"):
            self.is_kcb = True
        elif self.code.startswith("3"):
            self.is_cyb = True


@dataclass
class Bond(Instrument):
    """债券"""

    bond_type: Optional[str] = None  # 债券类型（国债、企业债、金融债等）
    coupon_rate: Optional[float] = None  # 票面利率
    maturity_date: Optional[date] = None  # 到期日
    par_value: float = 100.0  # 面值
    issue_size: Optional[float] = None  # 发行规模

    def __post_init__(self):
        self.asset_type = AssetType.BOND
        super().__post_init__()


@dataclass
class Convertible(Instrument):
    """可转债"""

    underlying_code: Optional[str] = None  # 正股代码
    conversion_price: Optional[float] = None  # 转股价
    conversion_start_date: Optional[date] = None  # 转股起始日
    conversion_end_date: Optional[date] = None  # 转股截止日
    maturity_date: Optional[date] = None  # 到期日
    coupon_rate: Optional[float] = None  # 票面利率

    def __post_init__(self):
        self.asset_type = AssetType.CONVERTIBLE
        super().__post_init__()

    @property
    def conversion_value(self) -> Optional[float]:
        """转股价值（需要正股价格计算）"""
        return None  # 需要外部传入正股价格


@dataclass
class Future(Instrument):
    """期货"""

    underlying: Optional[str] = None  # 标的物
    contract_size: Optional[float] = None  # 合约乘数
    price_tick: Optional[float] = None  # 最小变动价位
    delivery_month: Optional[str] = None  # 交割月份
    last_trade_date: Optional[date] = None  # 最后交易日
    delivery_date: Optional[date] = None  # 交割日

    def __post_init__(self):
        self.asset_type = AssetType.FUTURE
        super().__post_init__()

    @property
    def is_main_contract(self) -> bool:
        """是否主力合约"""
        return self.code.endswith("88") or "主力" in self.name


@dataclass
class Fund(Instrument):
    """基金"""

    fund_type: Optional[str] = None  # 基金类型（股票型、债券型、混合型等）
    manager: Optional[str] = None  # 基金经理
    custodian: Optional[str] = None  # 托管人
    benchmark: Optional[str] = None  # 业绩基准
    nav: Optional[float] = None  # 最新净值
    total_nav: Optional[float] = None  # 累计净值

    def __post_init__(self):
        # 判断是否ETF
        if self.fund_type and "ETF" in self.fund_type.upper():
            self.asset_type = AssetType.ETF
        else:
            self.asset_type = AssetType.FUND
        super().__post_init__()


@dataclass
class Index(Instrument):
    """指数"""

    index_type: Optional[str] = None  # 指数类型（规模、行业、主题等）
    publisher: Optional[str] = None  # 发布机构
    base_date: Optional[date] = None  # 基期
    base_point: float = 1000.0  # 基点

    def __post_init__(self):
        self.asset_type = AssetType.INDEX
        super().__post_init__()


# ============================================================
# 向后兼容：保留原有 StockInfo
# ============================================================


@dataclass
class StockInfo:
    """股票基本信息（向后兼容）"""

    code: str  # 股票代码
    name: str  # 股票名称
    market: str  # 市场（SH/SZ/BJ）
    list_date: Optional[date] = None  # 上市日期
    delist_date: Optional[date] = None  # 退市日期
    industry: Optional[str] = None  # 行业
    is_st: bool = False  # 是否ST

    @property
    def symbol(self) -> str:
        """完整代码（如：000001.SZ）"""
        return f"{self.code}.{self.market}"

    @property
    def is_active(self) -> bool:
        """是否在市"""
        return self.delist_date is None

    def to_equity(self) -> Equity:
        """转换为新的 Equity 类型"""
        return Equity(
            code=self.code,
            name=self.name,
            market=Market(self.market),
            industry=self.industry,
            is_st=self.is_st,
            list_date=self.list_date,
            delist_date=self.delist_date,
        )


# ============================================================
# 行情数据模型（存储不复权价格）
# ============================================================


class AdjustMethod(Enum):
    """复权方式"""

    NONE = "none"  # 不复权
    QFQ = "qfq"  # 前复权（以最新价格为基准）
    HFQ = "hfq"  # 后复权（以上市价格为基准）


@dataclass
class OHLCV:
    """
    OHLCV 行情数据基类

    存储原则：
    - open/high/low/close 存储【不复权】原始价格
    - adj_factor 存储累计复权因子
    - 使用时通过 adjusted() 方法动态计算复权价格

    复权因子说明：
    - 初始值为 1.0
    - 发生送股/转增时，因子 = 原因子 * (1 + 送转比例)
    - 发生配股时，因子 = 原因子 * (配股前价格) / (配股后价格)
    """

    code: str  # 证券代码
    trade_date: date  # 交易日期
    open: float  # 开盘价（不复权）
    high: float  # 最高价（不复权）
    low: float  # 最低价（不复权）
    close: float  # 收盘价（不复权）
    volume: float  # 成交量
    amount: float  # 成交额
    adj_factor: float = 1.0  # 累计复权因子

    def adjusted(
        self, method: AdjustMethod = AdjustMethod.HFQ, latest_factor: Optional[float] = None
    ) -> "OHLCV":
        """
        返回复权后的副本

        Args:
            method: 复权方式
            latest_factor: 最新复权因子（前复权时需要）

        Returns:
            复权后的新 OHLCV 对象
        """
        if method == AdjustMethod.NONE:
            return self

        if method == AdjustMethod.HFQ:
            # 后复权：直接乘以复权因子
            factor = self.adj_factor
        elif method == AdjustMethod.QFQ:
            # 前复权：除以最新因子，归一化到最新价格
            if latest_factor is None:
                raise ValueError("前复权需要提供 latest_factor")
            factor = self.adj_factor / latest_factor
        else:
            factor = 1.0

        return OHLCV(
            code=self.code,
            trade_date=self.trade_date,
            open=self.open * factor,
            high=self.high * factor,
            low=self.low * factor,
            close=self.close * factor,
            volume=self.volume,
            amount=self.amount,
            adj_factor=self.adj_factor,
        )

    @property
    def hfq_close(self) -> float:
        """后复权收盘价"""
        return self.close * self.adj_factor

    @property
    def vwap(self) -> float:
        """成交量加权平均价"""
        if self.volume > 0:
            return self.amount / self.volume
        return self.close

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "code": self.code,
            "trade_date": self.trade_date.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "amount": self.amount,
            "adj_factor": self.adj_factor,
        }


@dataclass
class DailyBar(OHLCV):
    """
    日线行情数据

    继承自 OHLCV，添加日线特有字段
    """

    pre_close: Optional[float] = None  # 前收盘价（不复权）
    change: Optional[float] = None  # 涨跌额
    pct_change: Optional[float] = None  # 涨跌幅 (%)
    turnover: Optional[float] = None  # 换手率 (%)
    total_shares: Optional[float] = None  # 总股本
    float_shares: Optional[float] = None  # 流通股本
    total_mv: Optional[float] = None  # 总市值
    float_mv: Optional[float] = None  # 流通市值

    def adjusted(
        self, method: AdjustMethod = AdjustMethod.HFQ, latest_factor: Optional[float] = None
    ) -> "DailyBar":
        """返回复权后的副本"""
        if method == AdjustMethod.NONE:
            return self

        if method == AdjustMethod.HFQ:
            factor = self.adj_factor
        elif method == AdjustMethod.QFQ:
            if latest_factor is None:
                raise ValueError("前复权需要提供 latest_factor")
            factor = self.adj_factor / latest_factor
        else:
            factor = 1.0

        return DailyBar(
            code=self.code,
            trade_date=self.trade_date,
            open=self.open * factor,
            high=self.high * factor,
            low=self.low * factor,
            close=self.close * factor,
            volume=self.volume,
            amount=self.amount,
            adj_factor=self.adj_factor,
            pre_close=self.pre_close * factor if self.pre_close else None,
            change=self.change * factor if self.change else None,
            pct_change=self.pct_change,  # 涨跌幅不变
            turnover=self.turnover,
            total_shares=self.total_shares,
            float_shares=self.float_shares,
            total_mv=self.total_mv,
            float_mv=self.float_mv,
        )


@dataclass
class MinuteBar(OHLCV):
    """
    分钟线行情数据

    注意：分钟线通常不需要复权处理
    """

    time: Optional[datetime] = None  # 具体时间（可选，trade_date + time）

    def __post_init__(self):
        # 分钟线默认不需要复权因子
        if self.adj_factor == 1.0:
            pass


# ============================================================
# 向后兼容：保留原有类名
# ============================================================


@dataclass
class StockDaily(DailyBar):
    """日线数据（向后兼容）"""

    @property
    def adj_close(self) -> float:
        """后复权收盘价（向后兼容）"""
        return self.hfq_close


@dataclass
class StockMinute:
    """分钟线数据（向后兼容）"""

    code: str
    datetime: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float

    def to_minute_bar(self) -> MinuteBar:
        """转换为新的 MinuteBar 类型"""
        return MinuteBar(
            code=self.code,
            trade_date=self.datetime.date(),
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            amount=self.amount,
            time=self.datetime,
        )


@dataclass
class IndexDaily:
    """指数日线数据"""

    code: str  # 指数代码
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    pre_close: Optional[float] = None
    change: Optional[float] = None
    pct_change: Optional[float] = None

    def to_daily_bar(self) -> DailyBar:
        """转换为 DailyBar"""
        return DailyBar(
            code=self.code,
            trade_date=self.trade_date,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            amount=self.amount,
            adj_factor=1.0,  # 指数无复权
            pre_close=self.pre_close,
            change=self.change,
            pct_change=self.pct_change,
        )


@dataclass
class FinancialData:
    """财务数据"""

    code: str
    report_date: date  # 报告期
    ann_date: date  # 公告日期

    # 基本指标
    eps: Optional[float] = None  # 每股收益
    bps: Optional[float] = None  # 每股净资产
    roe: Optional[float] = None  # 净资产收益率
    roa: Optional[float] = None  # 总资产收益率

    # 利润表
    revenue: Optional[float] = None  # 营业收入
    net_profit: Optional[float] = None  # 净利润
    gross_profit: Optional[float] = None  # 毛利润

    # 资产负债表
    total_assets: Optional[float] = None  # 总资产
    total_liab: Optional[float] = None  # 总负债
    total_equity: Optional[float] = None  # 股东权益
