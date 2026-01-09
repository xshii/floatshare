"""趋势线筑底策略

策略逻辑:
1. 两次筑底间隔30天以上
2. 整体趋势上升状态半年(MA120向上)
3. 同步确定筑顶线，筑顶线也要上升趋势
4. 在下一次触及筑底线时满仓购入
5. 在触及筑顶线时提前一天半仓卖出
6. 每次涨幅超过2%时，第二天如果量比当天小，则第三天开盘价抛出10%
7. 止损线为买入价的-10%
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np

from src.market.objective import (
    ObjectiveRegistry,
    BottomTrendlineIndicator,
    TopTrendlineIndicator,
)


class PositionAction(Enum):
    """持仓操作"""
    HOLD = "hold"
    BUY_FULL = "buy_full"       # 满仓买入
    SELL_HALF = "sell_half"     # 半仓卖出
    SELL_10PCT = "sell_10pct"   # 卖出10%
    STOP_LOSS = "stop_loss"     # 止损卖出


@dataclass
class StrategyState:
    """策略状态"""
    position: float = 0.0           # 当前仓位 (0-1)
    entry_price: float = 0.0        # 入场价格
    cost_basis: float = 0.0         # 成本基准
    stop_loss_price: float = 0.0    # 止损价

    # 涨幅追踪
    last_gain_day: int = -1         # 上次涨幅超2%的日期索引
    volume_check_day: int = -1      # 需要检查量比的日期
    sell_trigger_day: int = -1      # 需要卖出10%的日期

    # 历史记录
    trades: List[Dict] = field(default_factory=list)


@dataclass
class TrendlineSignal:
    """趋势线信号"""
    date: datetime
    action: PositionAction
    price: float
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


class TrendlineStrategy:
    """趋势线筑底策略"""

    def __init__(self, params: Optional[Dict] = None):
        self.params = params or {}

        # 策略参数
        self.min_bottom_interval = self.params.get("min_bottom_interval", 30)  # 筑底间隔天数
        self.trend_period = self.params.get("trend_period", 120)  # 趋势判断周期(半年约120交易日)
        self.touch_threshold = self.params.get("touch_threshold", 0.02)  # 触及阈值2%
        self.gain_threshold = self.params.get("gain_threshold", 0.02)  # 涨幅阈值2%
        self.stop_loss_pct = self.params.get("stop_loss_pct", 0.10)  # 止损比例10%

        # 指标
        self.bottom_indicator = BottomTrendlineIndicator({
            "lookback": self.trend_period,
            "min_lows": 2,
            "swing_window": 5,
            "touch_threshold": self.touch_threshold,
        })
        self.top_indicator = TopTrendlineIndicator({
            "lookback": self.trend_period,
            "min_highs": 2,
            "swing_window": 5,
            "touch_threshold": self.touch_threshold,
        })

        # 状态
        self.state = StrategyState()

    def _check_uptrend_half_year(self, data: pd.DataFrame) -> bool:
        """检查半年趋势是否上升"""
        if len(data) < self.trend_period:
            return False

        close = data["close"].values

        # MA120是否上升
        ma120 = pd.Series(close).rolling(120).mean()
        if len(ma120.dropna()) < 20:
            return False

        recent_ma = ma120.iloc[-20:].values
        slope = (recent_ma[-1] - recent_ma[0]) / 20

        # MA120斜率为正且当前价在MA120上方
        return slope > 0 and close[-1] > ma120.iloc[-1]

    def _check_bottom_interval(self, bottom_result) -> bool:
        """检查筑底间隔是否超过30天"""
        lows = bottom_result.params.get("lows", [])
        if len(lows) < 2:
            return False

        # 检查最后两个低点的间隔
        last_two = lows[-2:]
        interval = last_two[1][0] - last_two[0][0]
        return interval >= self.min_bottom_interval

    def _is_near_support(self, price: float, support: float) -> bool:
        """检查价格是否接近支撑线"""
        if support <= 0:
            return False
        distance = (price - support) / support
        return distance <= self.touch_threshold

    def _is_near_resistance(self, price: float, resistance: float) -> bool:
        """检查价格是否接近阻力线"""
        if resistance <= 0:
            return False
        distance = (resistance - price) / resistance
        return distance <= self.touch_threshold

    def _calculate_volume_ratio(self, data: pd.DataFrame, idx: int) -> float:
        """计算量比 (当日成交量 / 5日均量)"""
        if idx < 5:
            return 1.0

        volume = data["volume"].values
        avg_vol = np.mean(volume[idx-5:idx])
        return volume[idx] / avg_vol if avg_vol > 0 else 1.0

    def analyze(self, data: pd.DataFrame) -> TrendlineSignal:
        """
        分析当前数据，生成交易信号

        Args:
            data: 包含OHLCV的DataFrame

        Returns:
            TrendlineSignal: 交易信号
        """
        if len(data) < self.trend_period:
            return TrendlineSignal(
                date=datetime.now(),
                action=PositionAction.HOLD,
                price=data["close"].iloc[-1] if len(data) > 0 else 0,
                reason="数据不足",
            )

        current_idx = len(data) - 1
        current_price = data["close"].iloc[-1]
        current_low = data["low"].iloc[-1]
        current_high = data["high"].iloc[-1]
        current_date = data["trade_date"].iloc[-1] if "trade_date" in data.columns else datetime.now()

        # 1. 计算趋势线
        bottom_result = self.bottom_indicator.calculate(data)
        top_result = self.top_indicator.calculate(data)

        support_price = bottom_result.get("support_price", 0)
        resistance_price = top_result.get("resistance_price", 0)
        bottom_valid = bottom_result.get("is_valid", 0) == 1
        top_valid = top_result.get("is_valid", 0) == 1
        bottom_slope = bottom_result.get("slope", 0)
        top_slope = top_result.get("slope", 0)

        details = {
            "support_price": support_price,
            "resistance_price": resistance_price,
            "bottom_valid": bottom_valid,
            "top_valid": top_valid,
            "bottom_slope": bottom_slope,
            "top_slope": top_slope,
            "position": self.state.position,
            "entry_price": self.state.entry_price,
            "stop_loss_price": self.state.stop_loss_price,
        }

        # 2. 检查止损
        if self.state.position > 0 and self.state.stop_loss_price > 0:
            if current_low <= self.state.stop_loss_price:
                return TrendlineSignal(
                    date=current_date,
                    action=PositionAction.STOP_LOSS,
                    price=self.state.stop_loss_price,
                    reason=f"触发止损线 {self.state.stop_loss_price:.2f}",
                    details=details,
                )

        # 3. 检查是否需要在第三天卖出10%（涨幅超2%后的量比检查）
        if self.state.sell_trigger_day == current_idx and self.state.position > 0.1:
            return TrendlineSignal(
                date=current_date,
                action=PositionAction.SELL_10PCT,
                price=data["open"].iloc[-1],  # 开盘价卖出
                reason="涨幅超2%后量比缩小，第三天开盘卖出10%",
                details=details,
            )

        # 4. 检查量比（第二天检查）
        if self.state.volume_check_day == current_idx and self.state.position > 0:
            today_volume_ratio = self._calculate_volume_ratio(data, current_idx)
            yesterday_volume_ratio = self._calculate_volume_ratio(data, current_idx - 1)

            if today_volume_ratio < yesterday_volume_ratio:
                # 量比缩小，设置第三天卖出
                self.state.sell_trigger_day = current_idx + 1
                details["volume_ratio_today"] = today_volume_ratio
                details["volume_ratio_yesterday"] = yesterday_volume_ratio

        # 5. 检查涨幅是否超过2%
        if self.state.position > 0 and self.state.entry_price > 0:
            gain = (current_price - self.state.entry_price) / self.state.entry_price
            if gain >= self.gain_threshold and self.state.last_gain_day != current_idx:
                self.state.last_gain_day = current_idx
                self.state.volume_check_day = current_idx + 1  # 明天检查量比
                details["current_gain"] = gain

        # 6. 检查是否接近阻力线（提前一天半仓卖出）
        if self.state.position >= 0.5 and resistance_price > 0:
            # 预测明天是否会触及阻力线
            daily_move = (current_high - current_low) / current_price
            projected_high = current_price * (1 + daily_move)

            if self._is_near_resistance(projected_high, resistance_price):
                return TrendlineSignal(
                    date=current_date,
                    action=PositionAction.SELL_HALF,
                    price=current_price,
                    reason=f"预计明天触及阻力线 {resistance_price:.2f}，提前半仓卖出",
                    details=details,
                )

        # 7. 检查买入条件
        if self.state.position == 0:
            # 条件1: 半年趋势上升
            uptrend = self._check_uptrend_half_year(data)

            # 条件2: 有效筑底且间隔超30天
            valid_bottom = bottom_valid and self._check_bottom_interval(bottom_result)

            # 条件3: 筑顶线也要上升趋势 (slope >= 0)
            top_uptrend = top_slope >= 0

            # 条件4: 价格接近支撑线
            near_support = self._is_near_support(current_low, support_price)

            details["uptrend_half_year"] = uptrend
            details["valid_bottom_interval"] = valid_bottom
            details["top_uptrend"] = top_uptrend
            details["near_support"] = near_support

            if uptrend and valid_bottom and top_uptrend and near_support:
                # 设置止损价
                stop_loss = current_price * (1 - self.stop_loss_pct)

                return TrendlineSignal(
                    date=current_date,
                    action=PositionAction.BUY_FULL,
                    price=current_price,
                    reason=f"满足所有买入条件，触及支撑线 {support_price:.2f}，满仓买入",
                    details={**details, "stop_loss_price": stop_loss},
                )

        return TrendlineSignal(
            date=current_date,
            action=PositionAction.HOLD,
            price=current_price,
            reason="无交易信号",
            details=details,
        )

    def execute(self, signal: TrendlineSignal, capital: float = 100000) -> Dict:
        """
        执行交易信号

        Args:
            signal: 交易信号
            capital: 可用资金

        Returns:
            交易结果
        """
        result = {
            "date": signal.date,
            "action": signal.action.value,
            "price": signal.price,
            "reason": signal.reason,
            "before_position": self.state.position,
            "after_position": self.state.position,
            "shares": 0,
            "amount": 0,
        }

        if signal.action == PositionAction.BUY_FULL:
            # 满仓买入
            shares = int(capital / signal.price / 100) * 100
            amount = shares * signal.price

            self.state.position = 1.0
            self.state.entry_price = signal.price
            self.state.cost_basis = signal.price
            self.state.stop_loss_price = signal.details.get("stop_loss_price", signal.price * 0.9)

            result["after_position"] = 1.0
            result["shares"] = shares
            result["amount"] = amount

        elif signal.action == PositionAction.SELL_HALF:
            # 半仓卖出
            sell_ratio = 0.5
            result["after_position"] = self.state.position * (1 - sell_ratio)
            result["sell_ratio"] = sell_ratio
            self.state.position = result["after_position"]

        elif signal.action == PositionAction.SELL_10PCT:
            # 卖出10%
            sell_ratio = 0.1
            result["after_position"] = self.state.position * (1 - sell_ratio)
            result["sell_ratio"] = sell_ratio
            self.state.position = result["after_position"]

            # 更新止损线为当前价格的-10%
            self.state.stop_loss_price = signal.price * (1 - self.stop_loss_pct)
            result["new_stop_loss"] = self.state.stop_loss_price

        elif signal.action == PositionAction.STOP_LOSS:
            # 止损清仓
            result["after_position"] = 0
            result["loss"] = (signal.price - self.state.entry_price) / self.state.entry_price

            self.state.position = 0
            self.state.entry_price = 0
            self.state.stop_loss_price = 0

        self.state.trades.append(result)
        return result

    def backtest(self, data: pd.DataFrame, initial_capital: float = 100000) -> Dict:
        """
        回测策略

        Args:
            data: 历史数据
            initial_capital: 初始资金

        Returns:
            回测结果
        """
        # 重置状态
        self.state = StrategyState()

        capital = initial_capital
        shares = 0
        trades = []
        equity_curve = []

        # 需要足够的数据来计算指标
        start_idx = self.trend_period

        for i in range(start_idx, len(data)):
            current_data = data.iloc[:i+1].copy()
            current_price = data["close"].iloc[i]
            current_date = data["trade_date"].iloc[i] if "trade_date" in data.columns else i

            # 获取信号
            signal = self.analyze(current_data)

            # 执行交易
            if signal.action != PositionAction.HOLD:
                if signal.action == PositionAction.BUY_FULL:
                    buy_shares = int(capital / signal.price / 100) * 100
                    if buy_shares > 0:
                        cost = buy_shares * signal.price
                        capital -= cost
                        shares = buy_shares
                        self.state.position = 1.0
                        self.state.entry_price = signal.price
                        self.state.stop_loss_price = signal.price * (1 - self.stop_loss_pct)
                        trades.append({
                            "date": current_date,
                            "action": "BUY",
                            "price": signal.price,
                            "shares": buy_shares,
                            "amount": cost,
                            "reason": signal.reason,
                        })

                elif signal.action == PositionAction.SELL_HALF:
                    sell_shares = int(shares * 0.5 / 100) * 100
                    if sell_shares > 0:
                        revenue = sell_shares * signal.price
                        capital += revenue
                        shares -= sell_shares
                        self.state.position = shares * signal.price / (capital + shares * signal.price)
                        trades.append({
                            "date": current_date,
                            "action": "SELL_HALF",
                            "price": signal.price,
                            "shares": sell_shares,
                            "amount": revenue,
                            "reason": signal.reason,
                        })

                elif signal.action == PositionAction.SELL_10PCT:
                    sell_shares = int(shares * 0.1 / 100) * 100
                    if sell_shares > 0:
                        revenue = sell_shares * signal.price
                        capital += revenue
                        shares -= sell_shares
                        self.state.position = shares * signal.price / (capital + shares * signal.price) if (capital + shares * signal.price) > 0 else 0
                        # 更新止损线
                        self.state.stop_loss_price = signal.price * (1 - self.stop_loss_pct)
                        trades.append({
                            "date": current_date,
                            "action": "SELL_10PCT",
                            "price": signal.price,
                            "shares": sell_shares,
                            "amount": revenue,
                            "reason": signal.reason,
                        })

                elif signal.action == PositionAction.STOP_LOSS:
                    if shares > 0:
                        revenue = shares * signal.price
                        capital += revenue
                        trades.append({
                            "date": current_date,
                            "action": "STOP_LOSS",
                            "price": signal.price,
                            "shares": shares,
                            "amount": revenue,
                            "reason": signal.reason,
                        })
                        shares = 0
                        self.state.position = 0
                        self.state.entry_price = 0
                        self.state.stop_loss_price = 0

            # 记录权益
            equity = capital + shares * current_price
            equity_curve.append({
                "date": current_date,
                "equity": equity,
                "capital": capital,
                "shares": shares,
                "price": current_price,
                "position": self.state.position,
            })

        # 计算统计
        equity_df = pd.DataFrame(equity_curve)
        final_equity = equity_df["equity"].iloc[-1] if len(equity_df) > 0 else initial_capital

        returns = equity_df["equity"].pct_change().dropna()

        total_return = (final_equity - initial_capital) / initial_capital
        annual_return = total_return * 252 / len(equity_df) if len(equity_df) > 0 else 0

        max_drawdown = 0
        peak = initial_capital
        for eq in equity_df["equity"]:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_drawdown:
                max_drawdown = dd

        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        win_trades = [t for t in trades if t["action"] in ["SELL_HALF", "SELL_10PCT"] and t.get("amount", 0) > 0]

        return {
            "initial_capital": initial_capital,
            "final_equity": final_equity,
            "total_return": total_return,
            "annual_return": annual_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
            "total_trades": len(trades),
            "trades": trades,
            "equity_curve": equity_df,
        }


# 注册策略
def create_trendline_strategy(params: Optional[Dict] = None) -> TrendlineStrategy:
    """创建趋势线策略实例"""
    return TrendlineStrategy(params)
