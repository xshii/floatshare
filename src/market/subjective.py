"""主观指标 - 基于客观指标的综合判断"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from abc import ABC, abstractmethod
from enum import Enum
import pandas as pd

from src.market.objective import (
    ObjectiveValue,
    ObjectiveIndicator,
    ObjectiveRegistry,
)


class SignalType(Enum):
    """信号类型"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    WEAK_BUY = "weak_buy"
    NEUTRAL = "neutral"
    WEAK_SELL = "weak_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class MarketPhase(Enum):
    """市场阶段"""
    BULL = "bull"  # 牛市
    BEAR = "bear"  # 熊市
    CONSOLIDATION = "consolidation"  # 盘整
    RECOVERY = "recovery"  # 复苏
    DISTRIBUTION = "distribution"  # 派发


@dataclass
class SubjectiveResult:
    """主观判断结果"""
    name: str
    signal: SignalType
    score: float  # -100 到 100
    confidence: float  # 0 到 1
    components: Dict[str, Any]  # 组成部分
    reasons: List[str]  # 判断理由
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "signal": self.signal.value,
            "score": self.score,
            "confidence": self.confidence,
            "components": self.components,
            "reasons": self.reasons,
            "timestamp": self.timestamp.isoformat(),
        }


class SubjectiveIndicator(ABC):
    """主观指标基类 - 组合客观指标进行判断"""

    name: str = "BaseSubjective"
    description: str = ""
    required_objectives: List[str] = []  # 需要的客观指标

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        self._objective_indicators: Dict[str, ObjectiveIndicator] = {}
        self._init_objectives()

    def _init_objectives(self):
        """初始化需要的客观指标"""
        for name in self.required_objectives:
            indicator = ObjectiveRegistry.create(name, self.params.get(name))
            if indicator:
                self._objective_indicators[name] = indicator

    def get_param(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def calculate_objectives(self, data: pd.DataFrame) -> Dict[str, ObjectiveValue]:
        """计算所有客观指标"""
        results = {}
        for name, indicator in self._objective_indicators.items():
            results[name] = indicator.calculate(data)
        return results

    @abstractmethod
    def evaluate(self, objectives: Dict[str, ObjectiveValue]) -> SubjectiveResult:
        """基于客观指标进行主观判断"""
        pass

    def analyze(self, data: pd.DataFrame) -> SubjectiveResult:
        """完整分析流程"""
        objectives = self.calculate_objectives(data)
        return self.evaluate(objectives)


# ============================================================
# 内置主观指标
# ============================================================

class TrendJudgment(SubjectiveIndicator):
    """趋势判断 - 综合MA、ADX判断趋势"""

    name = "TrendJudgment"
    description = "趋势方向和强度判断"
    required_objectives = ["ma", "adx", "momentum"]

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)
        self.adx_trend_threshold = self.get_param("adx_trend_threshold", 25)
        self.adx_strong_threshold = self.get_param("adx_strong_threshold", 40)

    def evaluate(self, objectives: Dict[str, ObjectiveValue]) -> SubjectiveResult:
        ma_values = objectives.get("ma", ObjectiveValue("ma", {}))
        adx_values = objectives.get("adx", ObjectiveValue("adx", {}))
        momentum_values = objectives.get("momentum", ObjectiveValue("momentum", {}))

        reasons = []
        score = 0
        confidence = 0.5

        # 1. MA排列判断
        price = ma_values.get("price", 0)
        ma5 = ma_values.get("ma5", price)
        ma10 = ma_values.get("ma10", price)
        ma20 = ma_values.get("ma20", price)
        ma60 = ma_values.get("ma60", price)

        ma_score = 0
        if price > ma5 > ma10 > ma20:
            ma_score = 30
            reasons.append("短期均线多头排列")
        elif price < ma5 < ma10 < ma20:
            ma_score = -30
            reasons.append("短期均线空头排列")

        if ma20 > ma60:
            ma_score += 20
            reasons.append("中期趋势向上")
        elif ma20 < ma60:
            ma_score -= 20
            reasons.append("中期趋势向下")

        # 2. ADX判断趋势强度
        adx = adx_values.get("adx", 0)
        plus_di = adx_values.get("plus_di", 0)
        minus_di = adx_values.get("minus_di", 0)

        adx_score = 0
        if adx > self.adx_trend_threshold:
            confidence += 0.2
            if plus_di > minus_di:
                adx_score = 20 if adx > self.adx_strong_threshold else 10
                reasons.append(f"ADX={adx:.1f}显示上涨趋势")
            else:
                adx_score = -20 if adx > self.adx_strong_threshold else -10
                reasons.append(f"ADX={adx:.1f}显示下跌趋势")
        else:
            reasons.append(f"ADX={adx:.1f}趋势不明显")

        # 3. 动量判断
        ret_20d = momentum_values.get("ret_20d", 0)
        momentum_score = 0
        if ret_20d > 0.1:
            momentum_score = 20
            reasons.append(f"20日动量强劲: {ret_20d:.1%}")
        elif ret_20d < -0.1:
            momentum_score = -20
            reasons.append(f"20日动量疲弱: {ret_20d:.1%}")

        # 综合得分
        score = ma_score + adx_score + momentum_score
        confidence = min(confidence, 1.0)

        # 确定信号
        if score >= 50:
            signal = SignalType.STRONG_BUY
        elif score >= 25:
            signal = SignalType.BUY
        elif score >= 10:
            signal = SignalType.WEAK_BUY
        elif score <= -50:
            signal = SignalType.STRONG_SELL
        elif score <= -25:
            signal = SignalType.SELL
        elif score <= -10:
            signal = SignalType.WEAK_SELL
        else:
            signal = SignalType.NEUTRAL

        return SubjectiveResult(
            name=self.name,
            signal=signal,
            score=score,
            confidence=confidence,
            components={
                "ma_score": ma_score,
                "adx_score": adx_score,
                "momentum_score": momentum_score,
            },
            reasons=reasons,
        )


class OverboughtOversoldJudgment(SubjectiveIndicator):
    """超买超卖判断 - 综合RSI、KDJ、布林带"""

    name = "OverboughtOversold"
    description = "超买超卖状态判断"
    required_objectives = ["rsi", "kdj", "boll"]

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)
        self.rsi_overbought = self.get_param("rsi_overbought", 70)
        self.rsi_oversold = self.get_param("rsi_oversold", 30)
        self.kdj_overbought = self.get_param("kdj_overbought", 80)
        self.kdj_oversold = self.get_param("kdj_oversold", 20)

    def evaluate(self, objectives: Dict[str, ObjectiveValue]) -> SubjectiveResult:
        rsi_values = objectives.get("rsi", ObjectiveValue("rsi", {"rsi": 50}))
        kdj_values = objectives.get("kdj", ObjectiveValue("kdj", {"k": 50, "d": 50, "j": 50}))
        boll_values = objectives.get("boll", ObjectiveValue("boll", {"position": 0.5}))

        reasons = []
        overbought_count = 0
        oversold_count = 0
        confidence = 0.5

        # 1. RSI判断
        rsi = rsi_values.get("rsi", 50)
        if rsi > self.rsi_overbought:
            overbought_count += 1
            reasons.append(f"RSI={rsi:.1f}超买")
        elif rsi < self.rsi_oversold:
            oversold_count += 1
            reasons.append(f"RSI={rsi:.1f}超卖")

        # 2. KDJ判断
        k = kdj_values.get("k", 50)
        j = kdj_values.get("j", 50)
        if k > self.kdj_overbought or j > 100:
            overbought_count += 1
            reasons.append(f"KDJ K={k:.1f}, J={j:.1f}超买")
        elif k < self.kdj_oversold or j < 0:
            oversold_count += 1
            reasons.append(f"KDJ K={k:.1f}, J={j:.1f}超卖")

        # 3. 布林带位置判断
        boll_position = boll_values.get("position", 0.5)
        if boll_position > 0.95:
            overbought_count += 1
            reasons.append("价格触及布林带上轨")
        elif boll_position < 0.05:
            oversold_count += 1
            reasons.append("价格触及布林带下轨")

        # 综合判断
        if overbought_count >= 2:
            score = -50 - (overbought_count - 2) * 20
            signal = SignalType.SELL if overbought_count >= 3 else SignalType.WEAK_SELL
            confidence = 0.6 + overbought_count * 0.1
            reasons.append(f"综合判断: {overbought_count}个指标显示超买")
        elif oversold_count >= 2:
            score = 50 + (oversold_count - 2) * 20
            signal = SignalType.BUY if oversold_count >= 3 else SignalType.WEAK_BUY
            confidence = 0.6 + oversold_count * 0.1
            reasons.append(f"综合判断: {oversold_count}个指标显示超卖")
        else:
            score = (oversold_count - overbought_count) * 15
            signal = SignalType.NEUTRAL
            reasons.append("无明显超买超卖信号")

        return SubjectiveResult(
            name=self.name,
            signal=signal,
            score=max(min(score, 100), -100),
            confidence=min(confidence, 1.0),
            components={
                "overbought_count": overbought_count,
                "oversold_count": oversold_count,
                "rsi": rsi,
                "kdj_k": k,
                "boll_position": boll_position,
            },
            reasons=reasons,
        )


class MomentumTurnJudgment(SubjectiveIndicator):
    """动量转折判断 - 综合MACD、KDJ金叉死叉"""

    name = "MomentumTurn"
    description = "动量转折信号判断"
    required_objectives = ["macd", "kdj"]

    def evaluate(self, objectives: Dict[str, ObjectiveValue]) -> SubjectiveResult:
        macd_values = objectives.get("macd", ObjectiveValue("macd", {}))
        kdj_values = objectives.get("kdj", ObjectiveValue("kdj", {}))

        reasons = []
        score = 0
        confidence = 0.5

        # 1. MACD金叉死叉
        dif = macd_values.get("dif", 0)
        dea = macd_values.get("dea", 0)
        prev_dif = macd_values.get("prev_dif", 0)
        prev_dea = macd_values.get("prev_dea", 0)
        macd_hist = macd_values.get("macd", 0)

        macd_cross = None
        if prev_dif <= prev_dea and dif > dea:
            macd_cross = "golden"
            score += 30
            confidence += 0.15
            if dif < 0:
                reasons.append("MACD零轴下方金叉（底部信号）")
                score += 10
            else:
                reasons.append("MACD金叉")
        elif prev_dif >= prev_dea and dif < dea:
            macd_cross = "dead"
            score -= 30
            confidence += 0.15
            if dif > 0:
                reasons.append("MACD零轴上方死叉（顶部信号）")
                score -= 10
            else:
                reasons.append("MACD死叉")

        # MACD柱状图趋势
        if macd_hist > 0 and macd_cross != "dead":
            score += 10
            reasons.append("MACD柱状图为正")
        elif macd_hist < 0 and macd_cross != "golden":
            score -= 10
            reasons.append("MACD柱状图为负")

        # 2. KDJ金叉死叉
        k = kdj_values.get("k", 50)
        d = kdj_values.get("d", 50)
        prev_k = kdj_values.get("prev_k", 50)
        prev_d = kdj_values.get("prev_d", 50)

        kdj_cross = None
        if prev_k <= prev_d and k > d:
            kdj_cross = "golden"
            score += 20
            confidence += 0.1
            if k < 30:
                reasons.append("KDJ超卖区金叉（强买入信号）")
                score += 15
            else:
                reasons.append("KDJ金叉")
        elif prev_k >= prev_d and k < d:
            kdj_cross = "dead"
            score -= 20
            confidence += 0.1
            if k > 70:
                reasons.append("KDJ超买区死叉（强卖出信号）")
                score -= 15
            else:
                reasons.append("KDJ死叉")

        # 综合信号强度
        if macd_cross == "golden" and kdj_cross == "golden":
            score += 20
            confidence += 0.15
            reasons.append("MACD和KDJ同时金叉，共振信号")
        elif macd_cross == "dead" and kdj_cross == "dead":
            score -= 20
            confidence += 0.15
            reasons.append("MACD和KDJ同时死叉，共振信号")

        # 确定信号
        if score >= 60:
            signal = SignalType.STRONG_BUY
        elif score >= 30:
            signal = SignalType.BUY
        elif score >= 15:
            signal = SignalType.WEAK_BUY
        elif score <= -60:
            signal = SignalType.STRONG_SELL
        elif score <= -30:
            signal = SignalType.SELL
        elif score <= -15:
            signal = SignalType.WEAK_SELL
        else:
            signal = SignalType.NEUTRAL
            if not reasons:
                reasons.append("无明显转折信号")

        return SubjectiveResult(
            name=self.name,
            signal=signal,
            score=max(min(score, 100), -100),
            confidence=min(confidence, 1.0),
            components={
                "macd_cross": macd_cross,
                "kdj_cross": kdj_cross,
                "dif": dif,
                "dea": dea,
                "k": k,
                "d": d,
            },
            reasons=reasons,
        )


class VolatilityJudgment(SubjectiveIndicator):
    """波动率判断 - 综合ATR、布林带宽度"""

    name = "VolatilityJudgment"
    description = "市场波动率状态判断"
    required_objectives = ["atr", "boll"]

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)
        self.atr_high_threshold = self.get_param("atr_high_threshold", 0.03)
        self.atr_low_threshold = self.get_param("atr_low_threshold", 0.015)
        self.boll_wide_threshold = self.get_param("boll_wide_threshold", 0.15)
        self.boll_narrow_threshold = self.get_param("boll_narrow_threshold", 0.05)

    def evaluate(self, objectives: Dict[str, ObjectiveValue]) -> SubjectiveResult:
        atr_values = objectives.get("atr", ObjectiveValue("atr", {}))
        boll_values = objectives.get("boll", ObjectiveValue("boll", {}))

        reasons = []
        volatility_level = "normal"
        confidence = 0.6

        # ATR百分比
        atr_pct = atr_values.get("atr_pct", 0.02)
        # 布林带宽度
        boll_width = boll_values.get("width", 0.1)

        high_vol_count = 0
        low_vol_count = 0

        # ATR判断
        if atr_pct > self.atr_high_threshold:
            high_vol_count += 1
            reasons.append(f"ATR={atr_pct:.2%}显示高波动")
        elif atr_pct < self.atr_low_threshold:
            low_vol_count += 1
            reasons.append(f"ATR={atr_pct:.2%}显示低波动")

        # 布林带宽度判断
        if boll_width > self.boll_wide_threshold:
            high_vol_count += 1
            reasons.append(f"布林带宽度={boll_width:.2%}显示高波动")
        elif boll_width < self.boll_narrow_threshold:
            low_vol_count += 1
            reasons.append(f"布林带宽度={boll_width:.2%}显示低波动（可能酝酿突破）")

        # 综合判断
        if high_vol_count >= 2:
            volatility_level = "high"
            score = 0  # 高波动本身不是买卖信号
            signal = SignalType.NEUTRAL
            confidence = 0.8
            reasons.append("综合判断: 高波动市场，注意风险控制")
        elif low_vol_count >= 2:
            volatility_level = "low"
            score = 0
            signal = SignalType.NEUTRAL
            confidence = 0.8
            reasons.append("综合判断: 低波动市场，可能即将突破")
        else:
            volatility_level = "normal"
            score = 0
            signal = SignalType.NEUTRAL
            reasons.append("波动率正常")

        return SubjectiveResult(
            name=self.name,
            signal=signal,
            score=score,
            confidence=confidence,
            components={
                "volatility_level": volatility_level,
                "atr_pct": atr_pct,
                "boll_width": boll_width,
                "high_vol_count": high_vol_count,
                "low_vol_count": low_vol_count,
            },
            reasons=reasons,
        )


class ComprehensiveJudgment(SubjectiveIndicator):
    """综合判断 - 整合所有主观指标"""

    name = "Comprehensive"
    description = "综合市场判断"
    required_objectives = ["ma", "adx", "rsi", "macd", "kdj", "boll", "atr", "momentum"]

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)
        # 初始化子判断器
        self._sub_judgments = [
            TrendJudgment(params),
            OverboughtOversoldJudgment(params),
            MomentumTurnJudgment(params),
            VolatilityJudgment(params),
        ]

    def evaluate(self, objectives: Dict[str, ObjectiveValue]) -> SubjectiveResult:
        # 收集所有子判断结果
        sub_results = []
        total_score = 0
        total_confidence = 0
        all_reasons = []

        for judgment in self._sub_judgments:
            result = judgment.evaluate(objectives)
            sub_results.append(result)
            # 加权平均（可配置权重）
            weight = self.get_param(f"{judgment.name}_weight", 1.0)
            total_score += result.score * weight
            total_confidence += result.confidence * weight
            all_reasons.extend(result.reasons)

        # 计算平均
        num_judgments = len(self._sub_judgments)
        avg_score = total_score / num_judgments
        avg_confidence = total_confidence / num_judgments

        # 确定最终信号
        if avg_score >= 40:
            signal = SignalType.STRONG_BUY
        elif avg_score >= 20:
            signal = SignalType.BUY
        elif avg_score >= 10:
            signal = SignalType.WEAK_BUY
        elif avg_score <= -40:
            signal = SignalType.STRONG_SELL
        elif avg_score <= -20:
            signal = SignalType.SELL
        elif avg_score <= -10:
            signal = SignalType.WEAK_SELL
        else:
            signal = SignalType.NEUTRAL

        # 生成综合理由
        summary_reasons = []
        buy_signals = sum(1 for r in sub_results if r.score > 10)
        sell_signals = sum(1 for r in sub_results if r.score < -10)

        if buy_signals > sell_signals:
            summary_reasons.append(f"{buy_signals}/{num_judgments}个指标看多")
        elif sell_signals > buy_signals:
            summary_reasons.append(f"{sell_signals}/{num_judgments}个指标看空")
        else:
            summary_reasons.append("多空信号均衡")

        return SubjectiveResult(
            name=self.name,
            signal=signal,
            score=max(min(avg_score, 100), -100),
            confidence=min(avg_confidence, 1.0),
            components={
                "sub_results": [r.to_dict() for r in sub_results],
                "buy_signals": buy_signals,
                "sell_signals": sell_signals,
            },
            reasons=summary_reasons + all_reasons[:5],  # 只保留前5个详细理由
        )


# ============================================================
# 主观指标注册表
# ============================================================

class SubjectiveRegistry:
    """主观指标注册表"""

    _indicators: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, indicator_cls: type):
        cls._indicators[name] = indicator_cls

    @classmethod
    def get(cls, name: str) -> Optional[type]:
        return cls._indicators.get(name)

    @classmethod
    def create(cls, name: str, params: Optional[Dict] = None) -> Optional[SubjectiveIndicator]:
        indicator_cls = cls.get(name)
        if indicator_cls:
            return indicator_cls(params)
        return None

    @classmethod
    def list_indicators(cls) -> List[str]:
        return list(cls._indicators.keys())


# 注册内置主观指标
SubjectiveRegistry.register("trend", TrendJudgment)
SubjectiveRegistry.register("overbought_oversold", OverboughtOversoldJudgment)
SubjectiveRegistry.register("momentum_turn", MomentumTurnJudgment)
SubjectiveRegistry.register("volatility", VolatilityJudgment)
SubjectiveRegistry.register("comprehensive", ComprehensiveJudgment)


# ============================================================
# 市场综合分析器
# ============================================================

class MarketJudgmentAnalyzer:
    """市场综合判断分析器"""

    def __init__(self, indicators: Optional[List[str]] = None):
        """
        初始化分析器

        Args:
            indicators: 要使用的主观指标列表，None则使用全部
        """
        self.indicator_names = indicators or SubjectiveRegistry.list_indicators()
        self._indicators: Dict[str, SubjectiveIndicator] = {}

        for name in self.indicator_names:
            indicator = SubjectiveRegistry.create(name)
            if indicator:
                self._indicators[name] = indicator

    def analyze(self, data: pd.DataFrame) -> Dict[str, SubjectiveResult]:
        """
        分析市场数据

        Args:
            data: 包含OHLCV的DataFrame

        Returns:
            各指标的判断结果
        """
        results = {}
        for name, indicator in self._indicators.items():
            results[name] = indicator.analyze(data)
        return results

    def get_recommendation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取综合推荐

        Args:
            data: 包含OHLCV的DataFrame

        Returns:
            综合推荐结果
        """
        results = self.analyze(data)

        # 统计信号
        buy_count = 0
        sell_count = 0
        total_score = 0

        for result in results.values():
            if result.signal in [SignalType.STRONG_BUY, SignalType.BUY, SignalType.WEAK_BUY]:
                buy_count += 1
            elif result.signal in [SignalType.STRONG_SELL, SignalType.SELL, SignalType.WEAK_SELL]:
                sell_count += 1
            total_score += result.score

        avg_score = total_score / len(results) if results else 0

        # 综合推荐
        if avg_score >= 30:
            action = "买入"
            reason = "多数指标看涨"
        elif avg_score <= -30:
            action = "卖出"
            reason = "多数指标看跌"
        elif avg_score >= 10:
            action = "观望偏多"
            reason = "信号偏多但不够强"
        elif avg_score <= -10:
            action = "观望偏空"
            reason = "信号偏空但不够强"
        else:
            action = "观望"
            reason = "信号不明确"

        return {
            "action": action,
            "reason": reason,
            "score": avg_score,
            "buy_signals": buy_count,
            "sell_signals": sell_count,
            "neutral_signals": len(results) - buy_count - sell_count,
            "details": {name: r.to_dict() for name, r in results.items()},
        }
