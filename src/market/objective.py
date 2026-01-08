"""客观指标 - 纯数学计算，无主观判断"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


@dataclass
class ObjectiveValue:
    """客观指标值"""
    name: str
    values: Dict[str, float]  # 计算结果
    params: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def get(self, key: str, default: float = 0.0) -> float:
        return self.values.get(key, default)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "values": self.values,
            "params": self.params,
            "timestamp": self.timestamp.isoformat(),
        }


class ObjectiveIndicator(ABC):
    """客观指标基类 - 只计算，不判断"""

    name: str = "BaseObjective"
    description: str = ""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}

    def get_param(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> ObjectiveValue:
        """计算客观指标值"""
        pass


# ============================================================
# 内置客观指标
# ============================================================

class MAIndicator(ObjectiveIndicator):
    """均线指标"""

    name = "MA"
    description = "移动平均线"

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)
        self.periods = self.get_param("periods", [5, 10, 20, 60, 120, 250])

    def calculate(self, data: pd.DataFrame) -> ObjectiveValue:
        close = data["close"]
        values = {"price": close.iloc[-1]}

        for period in self.periods:
            if len(close) >= period:
                values[f"ma{period}"] = close.rolling(period).mean().iloc[-1]

        return ObjectiveValue(
            name=self.name,
            values=values,
            params={"periods": self.periods}
        )


class ATRIndicator(ObjectiveIndicator):
    """真实波幅指标"""

    name = "ATR"
    description = "平均真实波幅"

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)
        self.period = self.get_param("period", 14)

    def calculate(self, data: pd.DataFrame) -> ObjectiveValue:
        if len(data) < self.period:
            return ObjectiveValue(name=self.name, values={"atr": 0, "atr_pct": 0})

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
        atr_pct = atr / close[-1]

        return ObjectiveValue(
            name=self.name,
            values={
                "atr": atr,
                "atr_pct": atr_pct,
                "price": close[-1],
            },
            params={"period": self.period}
        )


class RSIIndicator(ObjectiveIndicator):
    """RSI指标"""

    name = "RSI"
    description = "相对强弱指标"

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)
        self.period = self.get_param("period", 14)

    def calculate(self, data: pd.DataFrame) -> ObjectiveValue:
        if len(data) < self.period + 1:
            return ObjectiveValue(name=self.name, values={"rsi": 50})

        close = data["close"]
        delta = close.diff()

        gain = delta.where(delta > 0, 0).rolling(self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.period).mean()

        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return ObjectiveValue(
            name=self.name,
            values={
                "rsi": rsi.iloc[-1],
                "avg_gain": gain.iloc[-1],
                "avg_loss": loss.iloc[-1],
            },
            params={"period": self.period}
        )


class MACDIndicator(ObjectiveIndicator):
    """MACD指标"""

    name = "MACD"
    description = "指数平滑异同移动平均线"

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)
        self.fast = self.get_param("fast", 12)
        self.slow = self.get_param("slow", 26)
        self.signal = self.get_param("signal", 9)

    def calculate(self, data: pd.DataFrame) -> ObjectiveValue:
        if len(data) < self.slow + self.signal:
            return ObjectiveValue(name=self.name, values={"dif": 0, "dea": 0, "macd": 0})

        close = data["close"]

        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=self.signal, adjust=False).mean()
        macd = (dif - dea) * 2

        return ObjectiveValue(
            name=self.name,
            values={
                "dif": dif.iloc[-1],
                "dea": dea.iloc[-1],
                "macd": macd.iloc[-1],
                "prev_dif": dif.iloc[-2] if len(dif) > 1 else 0,
                "prev_dea": dea.iloc[-2] if len(dea) > 1 else 0,
            },
            params={"fast": self.fast, "slow": self.slow, "signal": self.signal}
        )


class ADXObjectiveIndicator(ObjectiveIndicator):
    """ADX指标（客观部分）"""

    name = "ADX"
    description = "平均趋向指标"

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)
        self.period = self.get_param("period", 14)

    def calculate(self, data: pd.DataFrame) -> ObjectiveValue:
        if len(data) < self.period * 2:
            return ObjectiveValue(
                name=self.name,
                values={"adx": 0, "plus_di": 0, "minus_di": 0}
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

        return ObjectiveValue(
            name=self.name,
            values={
                "adx": adx.iloc[-1] if not np.isnan(adx.iloc[-1]) else 0,
                "plus_di": plus_di.iloc[-1] if not np.isnan(plus_di.iloc[-1]) else 0,
                "minus_di": minus_di.iloc[-1] if not np.isnan(minus_di.iloc[-1]) else 0,
                "atr": atr.iloc[-1] if not np.isnan(atr.iloc[-1]) else 0,
            },
            params={"period": self.period}
        )


class BollIndicator(ObjectiveIndicator):
    """布林带指标"""

    name = "Boll"
    description = "布林带"

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)
        self.period = self.get_param("period", 20)
        self.std_mult = self.get_param("std_mult", 2)

    def calculate(self, data: pd.DataFrame) -> ObjectiveValue:
        if len(data) < self.period:
            return ObjectiveValue(
                name=self.name,
                values={"upper": 0, "middle": 0, "lower": 0, "width": 0, "position": 0.5}
            )

        close = data["close"]
        price = close.iloc[-1]

        middle = close.rolling(self.period).mean().iloc[-1]
        std = close.rolling(self.period).std().iloc[-1]
        upper = middle + self.std_mult * std
        lower = middle - self.std_mult * std
        width = (upper - lower) / middle
        position = (price - lower) / (upper - lower) if upper != lower else 0.5

        return ObjectiveValue(
            name=self.name,
            values={
                "upper": upper,
                "middle": middle,
                "lower": lower,
                "width": width,
                "position": position,
                "price": price,
            },
            params={"period": self.period, "std_mult": self.std_mult}
        )


class KDJIndicator(ObjectiveIndicator):
    """KDJ指标"""

    name = "KDJ"
    description = "随机指标"

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)
        self.n = self.get_param("n", 9)
        self.m1 = self.get_param("m1", 3)
        self.m2 = self.get_param("m2", 3)

    def calculate(self, data: pd.DataFrame) -> ObjectiveValue:
        if len(data) < self.n + 5:
            return ObjectiveValue(name=self.name, values={"k": 50, "d": 50, "j": 50})

        high = data["high"]
        low = data["low"]
        close = data["close"]

        lowest = low.rolling(self.n).min()
        highest = high.rolling(self.n).max()

        rsv = (close - lowest) / (highest - lowest + 1e-10) * 100

        k = rsv.ewm(com=self.m1 - 1, adjust=False).mean()
        d = k.ewm(com=self.m2 - 1, adjust=False).mean()
        j = 3 * k - 2 * d

        return ObjectiveValue(
            name=self.name,
            values={
                "k": k.iloc[-1],
                "d": d.iloc[-1],
                "j": j.iloc[-1],
                "prev_k": k.iloc[-2] if len(k) > 1 else k.iloc[-1],
                "prev_d": d.iloc[-2] if len(d) > 1 else d.iloc[-1],
            },
            params={"n": self.n, "m1": self.m1, "m2": self.m2}
        )


class MomentumObjIndicator(ObjectiveIndicator):
    """动量指标"""

    name = "Momentum"
    description = "价格动量"

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)
        self.periods = self.get_param("periods", [5, 10, 20, 60])

    def calculate(self, data: pd.DataFrame) -> ObjectiveValue:
        close = data["close"]
        current = close.iloc[-1]

        values = {"price": current}
        for period in self.periods:
            if len(close) > period:
                past = close.iloc[-period - 1]
                values[f"ret_{period}d"] = (current - past) / past
                values[f"price_{period}d_ago"] = past

        return ObjectiveValue(
            name=self.name,
            values=values,
            params={"periods": self.periods}
        )


class PriceRangeIndicator(ObjectiveIndicator):
    """价格区间指标"""

    name = "PriceRange"
    description = "N日价格区间位置"

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)
        self.period = self.get_param("period", 60)

    def calculate(self, data: pd.DataFrame) -> ObjectiveValue:
        if len(data) < self.period:
            return ObjectiveValue(
                name=self.name,
                values={"position": 0.5, "high": 0, "low": 0}
            )

        recent = data.tail(self.period)
        high = recent["high"].max()
        low = recent["low"].min()
        price = data["close"].iloc[-1]

        position = (price - low) / (high - low) if high != low else 0.5

        return ObjectiveValue(
            name=self.name,
            values={
                "position": position,
                "high": high,
                "low": low,
                "price": price,
                "from_high": (price - high) / high,
                "from_low": (price - low) / low,
            },
            params={"period": self.period}
        )


class BottomTrendlineIndicator(ObjectiveIndicator):
    """筑底趋势线指标

    识别局部低点并拟合支撑趋势线，用于判断底部形态。

    返回值说明:
    - slope: 趋势线斜率（正=上升趋势，负=下降趋势）
    - intercept: 趋势线截距
    - support_price: 当前趋势线支撑价位
    - distance_pct: 当前价格距支撑线的百分比距离（正=在支撑线上方）
    - touch_count: 触及趋势线次数（越多越有效）
    - r_squared: 拟合优度（越接近1越好）
    - is_valid: 是否为有效趋势线
    - lows: 识别到的低点列表
    """

    name = "BottomTrendline"
    description = "筑底趋势线"

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)
        self.lookback = self.get_param("lookback", 60)  # 回看周期
        self.min_lows = self.get_param("min_lows", 3)   # 最少低点数
        self.swing_window = self.get_param("swing_window", 5)  # 低点识别窗口
        self.touch_threshold = self.get_param("touch_threshold", 0.02)  # 触及阈值2%

    def _find_swing_lows(self, data: pd.DataFrame) -> List[Dict]:
        """识别摆动低点（局部最低点）"""
        lows = data["low"].values
        swing_lows = []
        window = self.swing_window

        for i in range(window, len(lows) - window):
            # 检查是否是局部最低点
            is_low = True
            for j in range(i - window, i + window + 1):
                if j != i and lows[j] < lows[i]:
                    is_low = False
                    break

            if is_low:
                swing_lows.append({
                    "index": i,
                    "price": lows[i],
                    "date": data.index[i] if isinstance(data.index, pd.DatetimeIndex)
                            else data.iloc[i].get("trade_date", i),
                })

        return swing_lows

    def _fit_trendline(self, lows: List[Dict]) -> Dict:
        """最小二乘法拟合趋势线"""
        if len(lows) < 2:
            return {"slope": 0, "intercept": 0, "r_squared": 0}

        x = np.array([l["index"] for l in lows])
        y = np.array([l["price"] for l in lows])

        # 线性回归: y = slope * x + intercept
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x ** 2)

        denominator = n * sum_x2 - sum_x ** 2
        if abs(denominator) < 1e-10:
            return {"slope": 0, "intercept": y.mean(), "r_squared": 0}

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n

        # 计算R²
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / (ss_tot + 1e-10)

        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": max(0, r_squared),
        }

    def _count_touches(self, data: pd.DataFrame, slope: float, intercept: float) -> int:
        """统计价格触及趋势线的次数"""
        touches = 0
        lows = data["low"].values

        for i, low in enumerate(lows):
            support = slope * i + intercept
            if support > 0:
                distance_pct = (low - support) / support
                if abs(distance_pct) <= self.touch_threshold:
                    touches += 1

        return touches

    def calculate(self, data: pd.DataFrame) -> ObjectiveValue:
        if len(data) < self.lookback:
            return ObjectiveValue(
                name=self.name,
                values={
                    "slope": 0,
                    "intercept": 0,
                    "support_price": 0,
                    "distance_pct": 0,
                    "touch_count": 0,
                    "r_squared": 0,
                    "is_valid": 0,
                    "low_count": 0,
                }
            )

        recent = data.tail(self.lookback).reset_index(drop=True)

        # 1. 识别摆动低点
        swing_lows = self._find_swing_lows(recent)

        if len(swing_lows) < 2:
            current_price = recent["close"].iloc[-1]
            return ObjectiveValue(
                name=self.name,
                values={
                    "slope": 0,
                    "intercept": current_price,
                    "support_price": current_price * 0.95,
                    "distance_pct": 0.05,
                    "touch_count": 0,
                    "r_squared": 0,
                    "is_valid": 0,
                    "low_count": len(swing_lows),
                }
            )

        # 2. 拟合趋势线
        fit = self._fit_trendline(swing_lows)
        slope = fit["slope"]
        intercept = fit["intercept"]
        r_squared = fit["r_squared"]

        # 3. 计算当前支撑价位
        current_idx = len(recent) - 1
        support_price = slope * current_idx + intercept

        # 4. 计算距离
        current_price = recent["close"].iloc[-1]
        distance_pct = (current_price - support_price) / support_price if support_price > 0 else 0

        # 5. 统计触及次数
        touch_count = self._count_touches(recent, slope, intercept)

        # 6. 判断有效性
        is_valid = (
            len(swing_lows) >= self.min_lows and
            touch_count >= 2 and
            r_squared > 0.5 and
            slope >= 0  # 筑底应该是水平或上升趋势线
        )

        return ObjectiveValue(
            name=self.name,
            values={
                "slope": slope,
                "slope_pct": slope / intercept * 100 if intercept > 0 else 0,  # 每天涨跌%
                "intercept": intercept,
                "support_price": support_price,
                "current_price": current_price,
                "distance_pct": distance_pct,
                "touch_count": touch_count,
                "r_squared": r_squared,
                "is_valid": 1 if is_valid else 0,
                "low_count": len(swing_lows),
                "last_low_price": swing_lows[-1]["price"] if swing_lows else 0,
            },
            params={
                "lookback": self.lookback,
                "min_lows": self.min_lows,
                "swing_window": self.swing_window,
                "lows": [(l["index"], l["price"]) for l in swing_lows],
            }
        )


class TopTrendlineIndicator(ObjectiveIndicator):
    """筑顶趋势线指标

    识别局部高点并拟合阻力趋势线，用于判断顶部形态。
    """

    name = "TopTrendline"
    description = "筑顶趋势线"

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)
        self.lookback = self.get_param("lookback", 60)
        self.min_highs = self.get_param("min_highs", 3)
        self.swing_window = self.get_param("swing_window", 5)
        self.touch_threshold = self.get_param("touch_threshold", 0.02)

    def _find_swing_highs(self, data: pd.DataFrame) -> List[Dict]:
        """识别摆动高点（局部最高点）"""
        highs = data["high"].values
        swing_highs = []
        window = self.swing_window

        for i in range(window, len(highs) - window):
            is_high = True
            for j in range(i - window, i + window + 1):
                if j != i and highs[j] > highs[i]:
                    is_high = False
                    break

            if is_high:
                swing_highs.append({
                    "index": i,
                    "price": highs[i],
                })

        return swing_highs

    def _fit_trendline(self, highs: List[Dict]) -> Dict:
        if len(highs) < 2:
            return {"slope": 0, "intercept": 0, "r_squared": 0}

        x = np.array([h["index"] for h in highs])
        y = np.array([h["price"] for h in highs])

        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x ** 2)

        denominator = n * sum_x2 - sum_x ** 2
        if abs(denominator) < 1e-10:
            return {"slope": 0, "intercept": y.mean(), "r_squared": 0}

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n

        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / (ss_tot + 1e-10)

        return {"slope": slope, "intercept": intercept, "r_squared": max(0, r_squared)}

    def _count_touches(self, data: pd.DataFrame, slope: float, intercept: float) -> int:
        touches = 0
        highs = data["high"].values

        for i, high in enumerate(highs):
            resistance = slope * i + intercept
            if resistance > 0:
                distance_pct = (high - resistance) / resistance
                if abs(distance_pct) <= self.touch_threshold:
                    touches += 1

        return touches

    def calculate(self, data: pd.DataFrame) -> ObjectiveValue:
        if len(data) < self.lookback:
            return ObjectiveValue(
                name=self.name,
                values={"slope": 0, "resistance_price": 0, "is_valid": 0}
            )

        recent = data.tail(self.lookback).reset_index(drop=True)
        swing_highs = self._find_swing_highs(recent)

        if len(swing_highs) < 2:
            current_price = recent["close"].iloc[-1]
            return ObjectiveValue(
                name=self.name,
                values={
                    "slope": 0,
                    "resistance_price": current_price * 1.05,
                    "distance_pct": -0.05,
                    "is_valid": 0,
                    "high_count": len(swing_highs),
                }
            )

        fit = self._fit_trendline(swing_highs)
        slope = fit["slope"]
        intercept = fit["intercept"]
        r_squared = fit["r_squared"]

        current_idx = len(recent) - 1
        resistance_price = slope * current_idx + intercept
        current_price = recent["close"].iloc[-1]
        distance_pct = (current_price - resistance_price) / resistance_price if resistance_price > 0 else 0

        touch_count = self._count_touches(recent, slope, intercept)

        is_valid = (
            len(swing_highs) >= self.min_highs and
            touch_count >= 2 and
            r_squared > 0.5 and
            slope <= 0  # 筑顶应该是水平或下降趋势线
        )

        return ObjectiveValue(
            name=self.name,
            values={
                "slope": slope,
                "slope_pct": slope / intercept * 100 if intercept > 0 else 0,
                "intercept": intercept,
                "resistance_price": resistance_price,
                "current_price": current_price,
                "distance_pct": distance_pct,
                "touch_count": touch_count,
                "r_squared": r_squared,
                "is_valid": 1 if is_valid else 0,
                "high_count": len(swing_highs),
            },
            params={
                "lookback": self.lookback,
                "highs": [(h["index"], h["price"]) for h in swing_highs],
            }
        )


# ============================================================
# 客观指标注册表
# ============================================================

class ObjectiveRegistry:
    """客观指标注册表"""

    _indicators: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, indicator_cls: type):
        cls._indicators[name] = indicator_cls

    @classmethod
    def get(cls, name: str) -> Optional[type]:
        return cls._indicators.get(name)

    @classmethod
    def create(cls, name: str, params: Optional[Dict] = None) -> Optional[ObjectiveIndicator]:
        indicator_cls = cls.get(name)
        if indicator_cls:
            return indicator_cls(params)
        return None

    @classmethod
    def list_indicators(cls) -> List[str]:
        return list(cls._indicators.keys())


# 注册内置客观指标
ObjectiveRegistry.register("ma", MAIndicator)
ObjectiveRegistry.register("atr", ATRIndicator)
ObjectiveRegistry.register("rsi", RSIIndicator)
ObjectiveRegistry.register("macd", MACDIndicator)
ObjectiveRegistry.register("adx", ADXObjectiveIndicator)
ObjectiveRegistry.register("boll", BollIndicator)
ObjectiveRegistry.register("kdj", KDJIndicator)
ObjectiveRegistry.register("momentum", MomentumObjIndicator)
ObjectiveRegistry.register("price_range", PriceRangeIndicator)
ObjectiveRegistry.register("bottom_trendline", BottomTrendlineIndicator)
ObjectiveRegistry.register("top_trendline", TopTrendlineIndicator)
