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

from src.market.objective import (
    # 数据类
    ObjectiveValue,
    # 基类
    ObjectiveIndicator,
    # 内置客观指标
    MAIndicator,
    ATRIndicator,
    RSIIndicator,
    MACDIndicator,
    ADXObjectiveIndicator,
    BollIndicator,
    KDJIndicator,
    MomentumObjIndicator,
    PriceRangeIndicator,
    BottomTrendlineIndicator,
    TopTrendlineIndicator,
    # 注册表
    ObjectiveRegistry,
)

from src.market.subjective import (
    # 枚举
    SignalType,
    MarketPhase,
    # 数据类
    SubjectiveResult,
    # 基类
    SubjectiveIndicator,
    # 内置主观指标
    TrendJudgment,
    OverboughtOversoldJudgment,
    MomentumTurnJudgment,
    VolatilityJudgment,
    ComprehensiveJudgment,
    # 注册表和分析器
    SubjectiveRegistry,
    MarketJudgmentAnalyzer,
)

__all__ = [
    # indicators.py
    "MarketTrend",
    "TrendStrength",
    "Volatility",
    "IndicatorResult",
    "MarketState",
    "MarketIndicator",
    "IndicatorRegistry",
    "MarketAnalyzer",
    # objective.py
    "ObjectiveValue",
    "ObjectiveIndicator",
    "MAIndicator",
    "ATRIndicator",
    "RSIIndicator",
    "MACDIndicator",
    "ADXObjectiveIndicator",
    "BollIndicator",
    "KDJIndicator",
    "MomentumObjIndicator",
    "PriceRangeIndicator",
    "BottomTrendlineIndicator",
    "TopTrendlineIndicator",
    "ObjectiveRegistry",
    # subjective.py
    "SignalType",
    "MarketPhase",
    "SubjectiveResult",
    "SubjectiveIndicator",
    "TrendJudgment",
    "OverboughtOversoldJudgment",
    "MomentumTurnJudgment",
    "VolatilityJudgment",
    "ComprehensiveJudgment",
    "SubjectiveRegistry",
    "MarketJudgmentAnalyzer",
]
