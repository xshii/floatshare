"""趋势线超跌策略

策略逻辑:
1. 频繁筑底模式：3次筑底（每次间隔≥7天），第4次筑底或超跌时买入
2. 或传统模式：两次筑底间隔≥30天后买入
3. 超跌时全仓买入（RSI<30 或 价格跌破支撑线）
4. 超涨且价量不匹配时卖出10%（RSI>70 + 量比缩小）
5. 每次超跌补满全仓
6. 止损线为买入均价的-10%
7. 移动止盈：盈利达到阈值后启动，从最高点回撤超过阈值时全部卖出
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
    RSIIndicator,
)


class PositionAction(Enum):
    """持仓操作"""
    HOLD = "hold"
    BUY_FULL = "buy_full"           # 全仓买入
    BUY_REFILL = "buy_refill"       # 补仓到满仓
    SELL_10PCT = "sell_10pct"       # 卖出10%
    SELL_HALF = "sell_half"         # 止盈卖出50%
    SELL_ALL = "sell_all"           # 移动止损全部卖出
    STOP_LOSS = "stop_loss"         # 止损卖出


@dataclass
class StrategyState:
    """策略状态"""
    position: float = 0.0               # 当前仓位 (0-1)
    total_shares: int = 0               # 总持股数
    avg_cost: float = 0.0               # 持仓均价
    stop_loss_price: float = 0.0        # 止损价

    # 筑底确认
    bottom_confirmed: bool = False      # 是否确认筑底
    bottom_confirm_date: datetime = None  # 筑底确认日期
    wait_days: int = 0                  # 等待天数

    # 频繁筑底追踪
    valid_bottoms: int = 0              # 有效筑底次数
    last_bottom_idx: int = -1           # 上一次筑底的索引

    # 移动止盈
    highest_price: float = 0.0          # 持仓期间最高价
    trailing_stop_active: bool = False  # 移动止损是否激活

    # 交易记录
    trades: List[Dict] = field(default_factory=list)
    last_buy_date: datetime = None


@dataclass
class StrategySignal:
    """策略信号"""
    date: datetime
    action: PositionAction
    price: float
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


class TrendlineOversoldStrategy:
    """趋势线超跌策略"""

    def __init__(self, params: Optional[Dict] = None):
        self.params = params or {}

        # 策略参数
        self.min_bottom_interval = self.params.get("min_bottom_interval", 30)  # 传统模式间隔
        self.frequent_bottom_interval = self.params.get("frequent_bottom_interval", 7)  # 频繁模式最小间隔
        self.frequent_bottom_count = self.params.get("frequent_bottom_count", 3)  # 频繁模式需要的筑底次数
        self.wait_after_bottom = self.params.get("wait_after_bottom", 5)  # 筑底后等待天数
        self.lookback = self.params.get("lookback", 120)

        # 超买超卖阈值
        self.oversold_rsi = self.params.get("oversold_rsi", 30)
        self.overbought_rsi = self.params.get("overbought_rsi", 70)
        self.support_break_threshold = self.params.get("support_break_threshold", 0.02)

        # 价量配合
        self.volume_shrink_threshold = self.params.get("volume_shrink_threshold", 0.8)

        # 止损
        self.stop_loss_pct = self.params.get("stop_loss_pct", 0.10)

        # 止盈参数 - 上升趋势（斜率>0）：放宽止盈，让利润跑
        self.up_trailing_trigger = self.params.get("up_trailing_trigger", 0.25)  # 25%启动
        self.up_trailing_pct = self.params.get("up_trailing_pct", 0.12)  # 12%回撤

        # 止盈参数 - 下降趋势（斜率<=0）：收紧止盈，快速锁利
        self.down_trailing_trigger = self.params.get("down_trailing_trigger", 0.10)  # 10%启动
        self.down_trailing_pct = self.params.get("down_trailing_pct", 0.05)  # 5%回撤

        # 是否启用趋势自适应（默认开启）
        self.adaptive_mode = self.params.get("adaptive_mode", True)

        # 趋势过滤：下降趋势中不买入（默认关闭，因为趋势预测准确率≈50%）
        self.trend_filter = self.params.get("trend_filter", False)

        # 指标
        self.bottom_indicator = BottomTrendlineIndicator({
            "lookback": self.lookback,
            "min_lows": 2,
            "swing_window": 5,
        })
        self.rsi_indicator = RSIIndicator({"period": 14})

        # 状态
        self.state = StrategyState()

    def _check_frequent_bottoms(self, bottom_result) -> tuple:
        """
        检查频繁筑底模式：3次筑底（每次间隔≥7天）

        Returns:
            (is_ready, bottom_count, intervals): 是否满足条件, 筑底次数, 间隔列表
        """
        lows = bottom_result.params.get("lows", [])
        if len(lows) < self.frequent_bottom_count:
            return False, len(lows), []

        # 检查最近N个低点的间隔
        recent_lows = lows[-(self.frequent_bottom_count + 1):]  # 取最近4个低点（检查3个间隔）
        intervals = []
        valid_count = 0

        for i in range(len(recent_lows) - 1):
            interval = recent_lows[i + 1][0] - recent_lows[i][0]
            intervals.append(interval)
            if interval >= self.frequent_bottom_interval:
                valid_count += 1

        # 需要至少3个有效间隔（即4个低点，3个间隔都>=7天）
        is_ready = valid_count >= self.frequent_bottom_count

        return is_ready, len(lows), intervals

    def _check_bottom_formation(self, bottom_result) -> tuple:
        """
        检查是否形成有效筑底

        Returns:
            (is_valid, mode, details): 是否有效, 模式(frequent/traditional), 详情
        """
        lows = bottom_result.params.get("lows", [])
        if len(lows) < 2:
            return False, None, {}

        # 1. 检查频繁筑底模式
        freq_ready, freq_count, freq_intervals = self._check_frequent_bottoms(bottom_result)
        if freq_ready:
            return True, "frequent", {
                "bottom_count": freq_count,
                "intervals": freq_intervals,
                "mode": "频繁筑底模式"
            }

        # 2. 检查传统模式（最后两个低点间隔>=30天）
        last_two = lows[-2:]
        interval = last_two[1][0] - last_two[0][0]

        if interval >= self.min_bottom_interval and bottom_result.get("is_valid", 0) == 1:
            return True, "traditional", {
                "interval": interval,
                "mode": "传统筑底模式"
            }

        return False, None, {"intervals": [lows[i+1][0] - lows[i][0] for i in range(len(lows)-1)] if len(lows) > 1 else []}

    def _is_oversold(self, rsi: float, price: float, support: float) -> tuple:
        """判断是否超跌"""
        reasons = []
        is_oversold = False

        # RSI超跌
        if rsi < self.oversold_rsi:
            is_oversold = True
            reasons.append(f"RSI={rsi:.1f}<{self.oversold_rsi}")

        # 跌破支撑线
        if support > 0:
            break_pct = (price - support) / support
            if break_pct < -self.support_break_threshold:
                is_oversold = True
                reasons.append(f"跌破支撑{break_pct:.1%}")

        return is_oversold, reasons

    def _is_overbought_with_volume_divergence(self, data: pd.DataFrame, rsi: float) -> tuple:
        """判断是否超涨且价量不匹配"""
        reasons = []
        is_signal = False

        if len(data) < 5:
            return False, []

        # RSI超买
        if rsi > self.overbought_rsi:
            # 检查价量背离：价格上涨但成交量萎缩
            recent_5d = data.tail(5)
            price_change = (recent_5d['close'].iloc[-1] - recent_5d['close'].iloc[0]) / recent_5d['close'].iloc[0]

            # 计算量比（当前量 vs 5日均量）
            avg_volume = data['volume'].tail(10).head(5).mean()  # 前5日均量
            current_volume = data['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

            if price_change > 0.02 and volume_ratio < self.volume_shrink_threshold:
                is_signal = True
                reasons.append(f"RSI={rsi:.1f}>{self.overbought_rsi}")
                reasons.append(f"价涨{price_change:.1%}但量缩至{volume_ratio:.1%}")

        return is_signal, reasons

    def _check_take_profit(self, current_price: float, current_high: float, slope: float = 0) -> tuple:
        """
        检查移动止盈条件（趋势自适应模式）

        Args:
            current_price: 当前价格
            current_high: 当日最高价
            slope: 筑底线斜率，>0为上升趋势

        Returns:
            (should_sell, sell_pct, reason): 是否卖出, 卖出比例, 原因
        """
        if self.state.avg_cost <= 0:
            return False, 0, ""

        gain = (current_price - self.state.avg_cost) / self.state.avg_cost

        # 更新最高价
        if current_high > self.state.highest_price:
            self.state.highest_price = current_high

        # 根据趋势选择参数
        if self.adaptive_mode:
            if slope > 0:
                # 上升趋势：放宽止盈
                trigger = self.up_trailing_trigger
                drawdown_pct = self.up_trailing_pct
                trend_label = "↑"
            else:
                # 下降趋势：收紧止盈
                trigger = self.down_trailing_trigger
                drawdown_pct = self.down_trailing_pct
                trend_label = "↓"
        else:
            # 非自适应模式，使用下降趋势参数作为默认
            trigger = self.down_trailing_trigger
            drawdown_pct = self.down_trailing_pct
            trend_label = ""

        # 移动止盈：盈利达到阈值后启动，从最高点回撤超过阈值时全部卖出
        if gain >= trigger:
            self.state.trailing_stop_active = True

        if self.state.trailing_stop_active and self.state.highest_price > 0:
            drawdown = (self.state.highest_price - current_price) / self.state.highest_price
            if drawdown >= drawdown_pct:
                highest_gain = (self.state.highest_price - self.state.avg_cost) / self.state.avg_cost
                return True, 1.0, f"移动止盈{trend_label}: 最高盈利{highest_gain:.1%}, 回撤{drawdown:.1%}(阈值{drawdown_pct:.0%})"

        return False, 0, ""

    def _calculate_volume_ratio(self, data: pd.DataFrame) -> float:
        """计算量比"""
        if len(data) < 6:
            return 1.0
        avg_vol = data['volume'].tail(6).head(5).mean()
        current_vol = data['volume'].iloc[-1]
        return current_vol / avg_vol if avg_vol > 0 else 1.0

    def analyze(self, data: pd.DataFrame) -> StrategySignal:
        """分析当前数据，生成交易信号"""
        if len(data) < self.lookback:
            return StrategySignal(
                date=datetime.now(),
                action=PositionAction.HOLD,
                price=0,
                reason="数据不足",
            )

        current_price = data["close"].iloc[-1]
        current_low = data["low"].iloc[-1]
        current_high = data["high"].iloc[-1]
        current_date = data["trade_date"].iloc[-1] if "trade_date" in data.columns else datetime.now()

        # 1. 计算指标
        bottom_result = self.bottom_indicator.calculate(data)
        rsi_result = self.rsi_indicator.calculate(data)

        support_price = bottom_result.get("support_price", 0)
        slope = bottom_result.get("slope", 0)  # 筑底线斜率，用于趋势判断
        rsi = rsi_result.get("rsi", 50)
        volume_ratio = self._calculate_volume_ratio(data)

        # 趋势判断（用均线：MA20 > MA60 为上升趋势）
        ma20 = data["close"].tail(20).mean() if len(data) >= 20 else current_price
        ma60 = data["close"].tail(60).mean() if len(data) >= 60 else current_price
        trend = "up" if ma20 > ma60 else "down"
        trend_slope = slope  # 保留斜率用于止盈自适应

        details = {
            "support_price": support_price,
            "slope": slope,
            "trend": trend,
            "ma20": ma20,
            "ma60": ma60,
            "rsi": rsi,
            "volume_ratio": volume_ratio,
            "position": self.state.position,
            "avg_cost": self.state.avg_cost,
            "stop_loss_price": self.state.stop_loss_price,
            "bottom_confirmed": self.state.bottom_confirmed,
            "wait_days": self.state.wait_days,
        }

        # 2. 检查止损
        if self.state.position > 0 and self.state.stop_loss_price > 0:
            if current_low <= self.state.stop_loss_price:
                return StrategySignal(
                    date=current_date,
                    action=PositionAction.STOP_LOSS,
                    price=self.state.stop_loss_price,
                    reason=f"触发止损线 {self.state.stop_loss_price:.2f}",
                    details=details,
                )

        # 3. 检查筑底状态
        is_bottom_valid, bottom_mode, bottom_details = self._check_bottom_formation(bottom_result)
        details["bottom_mode"] = bottom_mode
        details["bottom_details"] = bottom_details

        if not self.state.bottom_confirmed:
            if is_bottom_valid:
                self.state.bottom_confirmed = True
                self.state.bottom_confirm_date = current_date
                self.state.wait_days = 0
                details["bottom_confirmed"] = True
                details["bottom_confirm_date"] = current_date
                details["bottom_mode"] = bottom_mode

        # 4. 筑底确认后等待
        if self.state.bottom_confirmed and self.state.bottom_confirm_date:
            if hasattr(current_date, 'date') and hasattr(self.state.bottom_confirm_date, 'date'):
                days_since = (current_date - self.state.bottom_confirm_date).days
            else:
                days_since = self.state.wait_days
            self.state.wait_days = days_since
            details["wait_days"] = days_since

            # 等待期未满
            if days_since < self.wait_after_bottom:
                return StrategySignal(
                    date=current_date,
                    action=PositionAction.HOLD,
                    price=current_price,
                    reason=f"筑底确认后等待中 ({days_since}/{self.wait_after_bottom}天) [{bottom_mode or ''}]",
                    details=details,
                )

        # 5. 持仓状态下检查卖出信号
        if self.state.position > 0.1:
            # 5.1 检查移动止盈（传入斜率用于趋势自适应）
            should_take_profit, _, tp_reason = self._check_take_profit(current_price, current_high, trend_slope)
            if should_take_profit:
                return StrategySignal(
                    date=current_date,
                    action=PositionAction.SELL_ALL,
                    price=current_price,
                    reason=tp_reason,
                    details={**details, "highest_price": self.state.highest_price, "trend": trend},
                )

            # 5.2 检查超涨价量背离
            is_overbought, ob_reasons = self._is_overbought_with_volume_divergence(data, rsi)
            if is_overbought:
                return StrategySignal(
                    date=current_date,
                    action=PositionAction.SELL_10PCT,
                    price=current_price,
                    reason=f"超涨价量背离: {', '.join(ob_reasons)}",
                    details=details,
                )

        # 6. 检查买入信号（筑底确认且等待期已过）
        if self.state.bottom_confirmed and self.state.wait_days >= self.wait_after_bottom:
            # 趋势过滤：下降趋势中不买入（MA20 < MA60）
            if self.trend_filter and trend == "down":
                return StrategySignal(
                    date=current_date,
                    action=PositionAction.HOLD,
                    price=current_price,
                    reason=f"趋势过滤: MA20({ma20:.2f})<MA60({ma60:.2f}), 等待上升趋势",
                    details=details,
                )

            is_oversold, os_reasons = self._is_oversold(rsi, current_price, support_price)

            if is_oversold:
                mode_str = f"[{bottom_mode}]" if bottom_mode else ""
                if self.state.position == 0:
                    # 空仓全仓买入
                    return StrategySignal(
                        date=current_date,
                        action=PositionAction.BUY_FULL,
                        price=current_price,
                        reason=f"超跌买入{mode_str}: {', '.join(os_reasons)}",
                        details={**details, "stop_loss_price": current_price * (1 - self.stop_loss_pct)},
                    )
                elif self.state.position < 0.95:
                    # 非满仓补仓
                    return StrategySignal(
                        date=current_date,
                        action=PositionAction.BUY_REFILL,
                        price=current_price,
                        reason=f"超跌补仓{mode_str}: {', '.join(os_reasons)}",
                        details=details,
                    )

        return StrategySignal(
            date=current_date,
            action=PositionAction.HOLD,
            price=current_price,
            reason="无交易信号",
            details=details,
        )

    def backtest(self, data: pd.DataFrame, initial_capital: float = 100000) -> Dict:
        """回测策略"""
        # 重置状态
        self.state = StrategyState()

        capital = initial_capital
        shares = 0
        trades = []
        equity_curve = []

        start_idx = self.lookback

        for i in range(start_idx, len(data)):
            current_data = data.iloc[:i+1].copy()
            current_price = data["close"].iloc[i]
            current_open = data["open"].iloc[i]
            current_date = data["trade_date"].iloc[i] if "trade_date" in data.columns else i

            # 更新等待天数
            if self.state.bottom_confirmed and self.state.bottom_confirm_date:
                if hasattr(current_date, 'timestamp'):
                    self.state.wait_days = (current_date - self.state.bottom_confirm_date).days

            # 获取信号
            signal = self.analyze(current_data)

            # 执行交易
            if signal.action == PositionAction.BUY_FULL:
                buy_shares = int(capital / current_price / 100) * 100
                if buy_shares > 0:
                    cost = buy_shares * current_price
                    capital -= cost
                    shares = buy_shares
                    self.state.position = 1.0
                    self.state.total_shares = shares
                    self.state.avg_cost = current_price
                    self.state.stop_loss_price = current_price * (1 - self.stop_loss_pct)
                    self.state.last_buy_date = current_date
                    # 重置止盈跟踪状态
                    self.state.highest_price = current_price
                    self.state.trailing_stop_active = False
                    trades.append({
                        "date": current_date,
                        "action": "BUY_FULL",
                        "price": current_price,
                        "shares": buy_shares,
                        "amount": cost,
                        "reason": signal.reason,
                    })

            elif signal.action == PositionAction.BUY_REFILL:
                # 补仓到满仓
                total_value = capital + shares * current_price
                target_shares = int(total_value / current_price / 100) * 100
                add_shares = target_shares - shares

                if add_shares > 0 and capital >= add_shares * current_price:
                    cost = add_shares * current_price
                    # 更新均价
                    new_avg = (self.state.avg_cost * shares + current_price * add_shares) / (shares + add_shares)
                    capital -= cost
                    shares += add_shares
                    self.state.position = 1.0
                    self.state.total_shares = shares
                    self.state.avg_cost = new_avg
                    self.state.stop_loss_price = new_avg * (1 - self.stop_loss_pct)
                    trades.append({
                        "date": current_date,
                        "action": "BUY_REFILL",
                        "price": current_price,
                        "shares": add_shares,
                        "amount": cost,
                        "new_avg_cost": new_avg,
                        "reason": signal.reason,
                    })

            elif signal.action == PositionAction.SELL_10PCT:
                sell_shares = int(shares * 0.1 / 100) * 100
                if sell_shares > 0:
                    revenue = sell_shares * current_price
                    capital += revenue
                    shares -= sell_shares
                    self.state.total_shares = shares
                    self.state.position = shares * current_price / (capital + shares * current_price) if (capital + shares * current_price) > 0 else 0
                    trades.append({
                        "date": current_date,
                        "action": "SELL_10PCT",
                        "price": current_price,
                        "shares": sell_shares,
                        "amount": revenue,
                        "reason": signal.reason,
                    })

            elif signal.action == PositionAction.SELL_HALF:
                # 止盈卖出50%
                sell_shares = int(shares * 0.5 / 100) * 100
                if sell_shares > 0:
                    revenue = sell_shares * current_price
                    gain_pct = (current_price - self.state.avg_cost) / self.state.avg_cost
                    capital += revenue
                    shares -= sell_shares
                    self.state.total_shares = shares
                    self.state.position = shares * current_price / (capital + shares * current_price) if (capital + shares * current_price) > 0 else 0
                    trades.append({
                        "date": current_date,
                        "action": "SELL_HALF",
                        "price": current_price,
                        "shares": sell_shares,
                        "amount": revenue,
                        "gain": gain_pct,
                        "reason": signal.reason,
                    })

            elif signal.action == PositionAction.SELL_ALL:
                # 移动止损全部卖出
                if shares > 0:
                    revenue = shares * current_price
                    gain_pct = (current_price - self.state.avg_cost) / self.state.avg_cost
                    capital += revenue
                    trades.append({
                        "date": current_date,
                        "action": "SELL_ALL",
                        "price": current_price,
                        "shares": shares,
                        "amount": revenue,
                        "gain": gain_pct,
                        "highest_price": self.state.highest_price,
                        "reason": signal.reason,
                    })
                    shares = 0
                    self.state.position = 0
                    self.state.total_shares = 0
                    self.state.avg_cost = 0
                    self.state.stop_loss_price = 0
                    self.state.highest_price = 0
                    self.state.trailing_stop_active = False
                    # 重置筑底状态，重新寻找机会
                    self.state.bottom_confirmed = False
                    self.state.bottom_confirm_date = None

            elif signal.action == PositionAction.STOP_LOSS:
                if shares > 0:
                    revenue = shares * signal.price
                    loss_pct = (signal.price - self.state.avg_cost) / self.state.avg_cost
                    capital += revenue
                    trades.append({
                        "date": current_date,
                        "action": "STOP_LOSS",
                        "price": signal.price,
                        "shares": shares,
                        "amount": revenue,
                        "loss": loss_pct,
                        "reason": signal.reason,
                    })
                    shares = 0
                    self.state.position = 0
                    self.state.total_shares = 0
                    self.state.avg_cost = 0
                    self.state.stop_loss_price = 0
                    self.state.highest_price = 0
                    self.state.trailing_stop_active = False
                    # 重置筑底状态，重新寻找机会
                    self.state.bottom_confirmed = False
                    self.state.bottom_confirm_date = None

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

        sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 and returns.std() > 0 else 0

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

    def rolling_backtest(
        self,
        data: pd.DataFrame,
        window_days: int = 180,
        step_days: int = 90,
        initial_capital: float = 100000,
    ) -> Dict:
        """
        滚动回测：将数据分成多个时间窗口，每个窗口独立回测

        Args:
            data: 完整数据
            window_days: 每个回测窗口的天数（默认180天/半年）
            step_days: 窗口滚动步长（默认90天/季度）
            initial_capital: 初始资金

        Returns:
            包含每个窗口结果和汇总统计的字典
        """
        if len(data) < self.lookback + window_days:
            return {"error": "数据不足", "windows": []}

        windows = []
        start_idx = self.lookback

        while start_idx + window_days <= len(data):
            # 取当前窗口数据（包含预热期）
            window_data = data.iloc[start_idx - self.lookback : start_idx + window_days].copy()

            # 获取窗口日期范围
            if "trade_date" in window_data.columns:
                start_date = window_data["trade_date"].iloc[self.lookback]
                end_date = window_data["trade_date"].iloc[-1]
            else:
                start_date = start_idx
                end_date = start_idx + window_days

            # 回测当前窗口
            result = self.backtest(window_data, initial_capital)

            # 计算同期买入持有收益
            first_price = window_data["close"].iloc[self.lookback]
            last_price = window_data["close"].iloc[-1]
            buy_hold = (last_price - first_price) / first_price

            windows.append({
                "start_date": start_date,
                "end_date": end_date,
                "strategy_return": result["total_return"],
                "buy_hold": buy_hold,
                "excess": result["total_return"] - buy_hold,
                "max_drawdown": result["max_drawdown"],
                "trades": result["total_trades"],
                "sharpe": result["sharpe_ratio"],
            })

            start_idx += step_days

        # 汇总统计
        if windows:
            avg_return = sum(w["strategy_return"] for w in windows) / len(windows)
            avg_buyhold = sum(w["buy_hold"] for w in windows) / len(windows)
            avg_excess = sum(w["excess"] for w in windows) / len(windows)
            avg_drawdown = sum(w["max_drawdown"] for w in windows) / len(windows)
            win_count = sum(1 for w in windows if w["excess"] > 0)
            win_rate = win_count / len(windows)

            # 收益标准差（稳定性）
            returns = [w["strategy_return"] for w in windows]
            return_std = np.std(returns) if len(returns) > 1 else 0

            summary = {
                "total_windows": len(windows),
                "avg_return": avg_return,
                "avg_buyhold": avg_buyhold,
                "avg_excess": avg_excess,
                "avg_drawdown": avg_drawdown,
                "win_rate": win_rate,
                "return_std": return_std,
            }
        else:
            summary = {}

        return {
            "windows": windows,
            "summary": summary,
        }
