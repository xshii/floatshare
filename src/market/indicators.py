"""市场状态指标插件系统"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Type
from enum import Enum
import pandas as pd
import numpy as np


# ============================================================
# 市场状态枚举
# ============================================================

class MarketTrend(Enum):
    """市场趋势"""
    BULL = "bull"           # 牛市
    BEAR = "bear"           # 熊市
    SIDEWAYS = "sideways"   # 震荡


class TrendStrength(Enum):
    """趋势强度"""
    STRONG = "strong"       # 强趋势
    MODERATE = "moderate"   # 中等
    WEAK = "weak"           # 弱趋势


class Volatility(Enum):
    """波动性"""
    HIGH = "high"           # 高波动
    MEDIUM = "medium"       # 中等
    LOW = "low"             # 低波动


class IndicatorType(Enum):
    """指标类型"""
    OBJECTIVE = "objective"     # 客观指标：纯数学计算，无主观判断
    SUBJECTIVE = "subjective"   # 主观指标：包含阈值判断、分类等人为设定


# ============================================================
# 指标结果数据类
# ============================================================

@dataclass
class IndicatorResult:
    """指标计算结果"""
    name: str                           # 指标名称
    value: float                        # 原始数值（客观）
    signal: str                         # 信号: bullish/bearish/neutral（主观）
    score: float                        # 得分: -1 到 +1（主观）
    indicator_type: str = "objective"   # 指标类型
    description: str = ""               # 描述
    raw_values: Dict[str, float] = field(default_factory=dict)  # 原始计算值（客观）
    params: Dict[str, Any] = field(default_factory=dict)  # 参数
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "value": self.value,
            "signal": self.signal,
            "score": self.score,
            "indicator_type": self.indicator_type,
            "description": self.description,
            "raw_values": self.raw_values,
            "params": self.params,
            "timestamp": self.timestamp.isoformat(),
        }

    @property
    def is_objective(self) -> bool:
        return self.indicator_type == "objective"

    @property
    def is_subjective(self) -> bool:
        return self.indicator_type == "subjective"


@dataclass
class MarketState:
    """市场状态汇总"""
    code: str                           # 标的代码（个股/指数）
    date: date                          # 日期
    trend: MarketTrend                  # 趋势方向
    trend_strength: TrendStrength       # 趋势强度
    volatility: Volatility              # 波动性
    total_score: float                  # 综合得分 -1 到 +1
    position_advice: float              # 建议仓位 0 到 1
    indicators: List[IndicatorResult] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "code": self.code,
            "date": self.date.isoformat(),
            "trend": self.trend.value,
            "trend_strength": self.trend_strength.value,
            "volatility": self.volatility.value,
            "total_score": self.total_score,
            "position_advice": self.position_advice,
            "indicators": [i.to_dict() for i in self.indicators],
        }

    def summary(self) -> str:
        """生成摘要"""
        trend_emoji = {"bull": "🐂", "bear": "🐻", "sideways": "↔️"}
        return (f"{self.code} [{self.date}] "
                f"{trend_emoji.get(self.trend.value, '')} {self.trend.value} "
                f"| 强度:{self.trend_strength.value} "
                f"| 波动:{self.volatility.value} "
                f"| 得分:{self.total_score:+.2f} "
                f"| 建议仓位:{self.position_advice:.0%}")


# ============================================================
# 指标基类
# ============================================================

class MarketIndicator(ABC):
    """市场指标基类"""

    name: str = "BaseIndicator"
    description: str = ""
    category: str = "general"  # trend/volatility/momentum/sentiment
    indicator_type: str = "objective"  # objective/subjective

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}

    def get_param(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        计算指标

        Args:
            data: 包含 OHLCV 的 DataFrame
                  必须包含: close, high, low, volume, trade_date

        Returns:
            IndicatorResult
        """
        pass


# ============================================================
# 内置指标实现
# ============================================================

class ADXIndicator(MarketIndicator):
    """ADX 平均趋向指标 - 客观计算 + 主观判断"""

    name = "ADX"
    description = "平均趋向指标，判断趋势强度"
    category = "trend"
    indicator_type = "subjective"  # 包含阈值判断(25/20)

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)
        self.period = self.get_param("period", 14)
        # 主观阈值参数
        self.strong_threshold = self.get_param("strong_threshold", 25)
        self.weak_threshold = self.get_param("weak_threshold", 20)

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period * 2:
            return IndicatorResult(
                name=self.name, value=0, signal="neutral", score=0,
                indicator_type=self.indicator_type,
                description="数据不足"
            )

        high = data["high"].values
        low = data["low"].values
        close = data["close"].values

        # ===== 客观计算部分 =====
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        plus_dm = np.where(
            (high[1:] - high[:-1]) > (low[:-1] - low[1:]),
            np.maximum(high[1:] - high[:-1], 0), 0
        )
        minus_dm = np.where(
            (low[:-1] - low[1:]) > (high[1:] - high[:-1]),
            np.maximum(low[:-1] - low[1:], 0), 0
        )

        atr = pd.Series(tr).rolling(self.period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(self.period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(self.period).mean() / atr

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(self.period).mean()

        adx_value = adx.iloc[-1] if not np.isnan(adx.iloc[-1]) else 20
        plus_di_value = plus_di.iloc[-1] if not np.isnan(plus_di.iloc[-1]) else 0
        minus_di_value = minus_di.iloc[-1] if not np.isnan(minus_di.iloc[-1]) else 0

        # 客观原始值
        raw_values = {
            "adx": adx_value,
            "plus_di": plus_di_value,
            "minus_di": minus_di_value,
            "atr": atr.iloc[-1] if not np.isnan(atr.iloc[-1]) else 0,
        }

        # ===== 主观判断部分 =====
        if adx_value > self.strong_threshold:
            signal = "bullish" if plus_di_value > minus_di_value else "bearish"
            score = 0.5 if signal == "bullish" else -0.5
            desc = f"强趋势 (ADX={adx_value:.1f})"
        elif adx_value < self.weak_threshold:
            signal = "neutral"
            score = 0
            desc = f"震荡市 (ADX={adx_value:.1f})"
        else:
            signal = "neutral"
            score = 0.1 if plus_di_value > minus_di_value else -0.1
            desc = f"过渡期 (ADX={adx_value:.1f})"

        return IndicatorResult(
            name=self.name,
            value=adx_value,
            signal=signal,
            score=score,
            indicator_type=self.indicator_type,
            description=desc,
            raw_values=raw_values,
            params={"period": self.period, "strong_threshold": self.strong_threshold}
        )


class MAPositionIndicator(MarketIndicator):
    """均线位置指标"""

    name = "MA_Position"
    description = "价格相对均线位置，判断牛熊"
    category = "trend"

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)
        self.periods = self.get_param("periods", [60, 120, 250])

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        close = data["close"]
        current = close.iloc[-1]

        mas = {}
        above_count = 0

        for period in self.periods:
            if len(close) >= period:
                ma = close.rolling(period).mean().iloc[-1]
                mas[f"MA{period}"] = ma
                if current > ma:
                    above_count += 1

        if not mas:
            return IndicatorResult(
                name=self.name, value=0, signal="neutral", score=0,
                description="数据不足"
            )

        ratio = above_count / len(mas)

        if ratio >= 0.8:
            signal = "bullish"
            score = 0.6
            desc = f"多头排列，站上{above_count}/{len(mas)}条均线"
        elif ratio <= 0.2:
            signal = "bearish"
            score = -0.6
            desc = f"空头排列，跌破{len(mas)-above_count}/{len(mas)}条均线"
        else:
            signal = "neutral"
            score = (ratio - 0.5) * 0.6
            desc = f"均线交织，站上{above_count}/{len(mas)}条均线"

        return IndicatorResult(
            name=self.name,
            value=ratio,
            signal=signal,
            score=score,
            description=desc,
            params={"periods": self.periods, "mas": mas}
        )


class VolatilityIndicator(MarketIndicator):
    """波动率指标"""

    name = "Volatility"
    description = "ATR/价格比率，判断波动性"
    category = "volatility"

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)
        self.period = self.get_param("period", 14)
        self.high_threshold = self.get_param("high_threshold", 0.03)
        self.low_threshold = self.get_param("low_threshold", 0.015)

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            return IndicatorResult(
                name=self.name, value=0, signal="neutral", score=0,
                description="数据不足"
            )

        high = data["high"].values
        low = data["low"].values
        close = data["close"].values

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        atr = pd.Series(tr).rolling(self.period).mean().iloc[-1]
        ratio = atr / close[-1]

        if ratio > self.high_threshold:
            signal = "high"
            score = 0  # 高波动本身不代表方向
            desc = f"高波动 ({ratio:.2%})，适合突破策略"
        elif ratio < self.low_threshold:
            signal = "low"
            score = 0
            desc = f"低波动 ({ratio:.2%})，适合均值回归"
        else:
            signal = "medium"
            score = 0
            desc = f"中等波动 ({ratio:.2%})"

        return IndicatorResult(
            name=self.name,
            value=ratio,
            signal=signal,
            score=score,
            description=desc,
            params={"period": self.period, "atr": atr}
        )


class MomentumIndicator(MarketIndicator):
    """动量指标"""

    name = "Momentum"
    description = "N日涨跌幅，判断动量方向"
    category = "momentum"

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)
        self.periods = self.get_param("periods", [20, 60, 120])

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        close = data["close"]
        current = close.iloc[-1]

        returns = {}
        for period in self.periods:
            if len(close) > period:
                ret = (current / close.iloc[-period-1] - 1)
                returns[f"ret_{period}d"] = ret

        if not returns:
            return IndicatorResult(
                name=self.name, value=0, signal="neutral", score=0,
                description="数据不足"
            )

        avg_return = np.mean(list(returns.values()))

        if avg_return > 0.1:
            signal = "bullish"
            score = min(avg_return, 0.5)
            desc = f"强势上涨，平均涨幅{avg_return:.1%}"
        elif avg_return < -0.1:
            signal = "bearish"
            score = max(avg_return, -0.5)
            desc = f"弱势下跌，平均跌幅{avg_return:.1%}"
        else:
            signal = "neutral"
            score = avg_return
            desc = f"动量中性，平均涨跌{avg_return:.1%}"

        return IndicatorResult(
            name=self.name,
            value=avg_return,
            signal=signal,
            score=score,
            description=desc,
            params={"returns": returns}
        )


class PricePositionIndicator(MarketIndicator):
    """价格位置指标"""

    name = "Price_Position"
    description = "价格在N日高低点中的位置"
    category = "momentum"

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)
        self.period = self.get_param("period", 60)

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            return IndicatorResult(
                name=self.name, value=0.5, signal="neutral", score=0,
                description="数据不足"
            )

        recent = data.tail(self.period)
        high = recent["high"].max()
        low = recent["low"].min()
        current = data["close"].iloc[-1]

        position = (current - low) / (high - low) if high != low else 0.5

        if position > 0.8:
            signal = "overbought"
            score = -0.2  # 高位略看空
            desc = f"高位区 ({position:.0%})，注意回调"
        elif position < 0.2:
            signal = "oversold"
            score = 0.2  # 低位略看多
            desc = f"低位区 ({position:.0%})，可能反弹"
        else:
            signal = "neutral"
            score = 0
            desc = f"中间位置 ({position:.0%})"

        return IndicatorResult(
            name=self.name,
            value=position,
            signal=signal,
            score=score,
            description=desc,
            params={"period": self.period, "high": high, "low": low}
        )


class BollWidthIndicator(MarketIndicator):
    """布林带宽度指标"""

    name = "Boll_Width"
    description = "布林带宽度，判断蓄势/突破"
    category = "volatility"

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)
        self.period = self.get_param("period", 20)
        self.std_mult = self.get_param("std_mult", 2)

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        if len(data) < self.period:
            return IndicatorResult(
                name=self.name, value=0, signal="neutral", score=0,
                description="数据不足"
            )

        close = data["close"]
        ma = close.rolling(self.period).mean()
        std = close.rolling(self.period).std()

        width = (self.std_mult * 2 * std.iloc[-1]) / ma.iloc[-1]

        # 计算宽度变化
        if len(data) > self.period + 5:
            prev_width = (self.std_mult * 2 * std.iloc[-6]) / ma.iloc[-6]
            expanding = width > prev_width * 1.1
            contracting = width < prev_width * 0.9
        else:
            expanding = False
            contracting = False

        if width > 0.1:
            signal = "high"
            desc = f"带宽扩张 ({width:.1%})"
            if expanding:
                desc += "，趋势形成"
        elif width < 0.05:
            signal = "low"
            desc = f"带宽收窄 ({width:.1%})"
            if contracting:
                desc += "，蓄势待变"
        else:
            signal = "medium"
            desc = f"带宽正常 ({width:.1%})"

        return IndicatorResult(
            name=self.name,
            value=width,
            signal=signal,
            score=0,  # 带宽本身不代表方向
            description=desc,
            params={"period": self.period, "expanding": expanding, "contracting": contracting}
        )


# ============================================================
# 指标注册表
# ============================================================

class IndicatorRegistry:
    """指标注册表"""

    _indicators: Dict[str, Type[MarketIndicator]] = {}

    @classmethod
    def register(cls, name: str):
        """注册指标装饰器"""
        def decorator(indicator_cls: Type[MarketIndicator]):
            cls._indicators[name] = indicator_cls
            return indicator_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Type[MarketIndicator]]:
        return cls._indicators.get(name)

    @classmethod
    def create(cls, name: str, params: Optional[Dict] = None) -> Optional[MarketIndicator]:
        indicator_cls = cls.get(name)
        if indicator_cls:
            return indicator_cls(params)
        return None

    @classmethod
    def list_indicators(cls) -> List[str]:
        return list(cls._indicators.keys())

    @classmethod
    def register_class(cls, name: str, indicator_cls: Type[MarketIndicator]):
        cls._indicators[name] = indicator_cls


# 注册内置指标
IndicatorRegistry.register_class("adx", ADXIndicator)
IndicatorRegistry.register_class("ma_position", MAPositionIndicator)
IndicatorRegistry.register_class("volatility", VolatilityIndicator)
IndicatorRegistry.register_class("momentum", MomentumIndicator)
IndicatorRegistry.register_class("price_position", PricePositionIndicator)
IndicatorRegistry.register_class("boll_width", BollWidthIndicator)


# ============================================================
# 市场分析器
# ============================================================

class MarketAnalyzer:
    """市场状态分析器"""

    def __init__(self, indicators: Optional[List[str]] = None):
        """
        初始化分析器

        Args:
            indicators: 要使用的指标列表，None表示使用全部
        """
        if indicators is None:
            indicators = IndicatorRegistry.list_indicators()

        self.indicators: List[MarketIndicator] = []
        for name in indicators:
            indicator = IndicatorRegistry.create(name)
            if indicator:
                self.indicators.append(indicator)

    def analyze(self, data: pd.DataFrame, code: str = "unknown") -> MarketState:
        """
        分析市场状态

        Args:
            data: OHLCV 数据
            code: 标的代码

        Returns:
            MarketState
        """
        results = []
        total_score = 0

        for indicator in self.indicators:
            result = indicator.calculate(data)
            results.append(result)
            total_score += result.score

        # 归一化得分
        if self.indicators:
            avg_score = total_score / len(self.indicators)
        else:
            avg_score = 0

        # 判断趋势方向
        if avg_score > 0.2:
            trend = MarketTrend.BULL
        elif avg_score < -0.2:
            trend = MarketTrend.BEAR
        else:
            trend = MarketTrend.SIDEWAYS

        # 判断趋势强度
        abs_score = abs(avg_score)
        if abs_score > 0.4:
            strength = TrendStrength.STRONG
        elif abs_score > 0.2:
            strength = TrendStrength.MODERATE
        else:
            strength = TrendStrength.WEAK

        # 判断波动性
        vol_result = next((r for r in results if r.name == "Volatility"), None)
        if vol_result:
            if vol_result.signal == "high":
                volatility = Volatility.HIGH
            elif vol_result.signal == "low":
                volatility = Volatility.LOW
            else:
                volatility = Volatility.MEDIUM
        else:
            volatility = Volatility.MEDIUM

        # 建议仓位
        if trend == MarketTrend.BULL:
            position_advice = 0.6 + avg_score * 0.4  # 0.6 ~ 1.0
        elif trend == MarketTrend.BEAR:
            position_advice = max(0, 0.3 + avg_score)  # 0 ~ 0.3
        else:
            position_advice = 0.3 + (avg_score + 0.2) * 0.5  # 0.2 ~ 0.5

        position_advice = max(0, min(1, position_advice))

        current_date = data["trade_date"].iloc[-1]
        if isinstance(current_date, pd.Timestamp):
            current_date = current_date.date()

        return MarketState(
            code=code,
            date=current_date,
            trend=trend,
            trend_strength=strength,
            volatility=volatility,
            total_score=avg_score,
            position_advice=position_advice,
            indicators=results,
        )

    def get_strategy_recommendation(self, state: MarketState) -> Dict[str, Any]:
        """根据市场状态推荐策略"""

        recommendations = {
            "trend_strategies": [],
            "revert_strategies": [],
            "breakout_strategies": [],
            "position_size": state.position_advice,
        }

        if state.trend in [MarketTrend.BULL, MarketTrend.BEAR]:
            if state.trend_strength == TrendStrength.STRONG:
                recommendations["trend_strategies"] = ["turtle", "momentum", "ma_cross"]
                recommendations["primary"] = "趋势跟踪"
            else:
                recommendations["trend_strategies"] = ["ma_cross", "macd"]
                recommendations["primary"] = "趋势确认"
        else:
            if state.volatility == Volatility.LOW:
                recommendations["breakout_strategies"] = ["dual_thrust", "boll_breakout"]
                recommendations["primary"] = "等待突破"
            else:
                recommendations["revert_strategies"] = ["rsi", "kdj", "boll_revert"]
                recommendations["primary"] = "高抛低吸"

        return recommendations
