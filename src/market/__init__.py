"""市场分析模块"""

from src.market.indicators import (
    # 枚举
    MarketTrend,
    TrendStrength,
    Volatility,
    # 数据类
    IndicatorResult,
    MarketState,
    # 基类
    MarketIndicator,
    # 内置指标
    ADXIndicator,
    MAPositionIndicator,
    VolatilityIndicator,
    MomentumIndicator,
    PricePositionIndicator,
    BollWidthIndicator,
    # 注册表和分析器
    IndicatorRegistry,
    MarketAnalyzer,
)

__all__ = [
    "MarketTrend",
    "TrendStrength",
    "Volatility",
    "IndicatorResult",
    "MarketState",
    "MarketIndicator",
    "IndicatorRegistry",
    "MarketAnalyzer",
]
