"""数据模型定义"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional


@dataclass
class StockInfo:
    """股票基本信息"""

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


@dataclass
class StockDaily:
    """日线数据"""

    code: str
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float  # 成交量（股）
    amount: float  # 成交额（元）
    pre_close: Optional[float] = None  # 前收盘价
    change: Optional[float] = None  # 涨跌额
    pct_change: Optional[float] = None  # 涨跌幅
    turnover: Optional[float] = None  # 换手率
    adj_factor: Optional[float] = None  # 复权因子

    @property
    def adj_close(self) -> float:
        """后复权收盘价"""
        if self.adj_factor:
            return self.close * self.adj_factor
        return self.close


@dataclass
class StockMinute:
    """分钟线数据"""

    code: str
    datetime: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float


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
