"""回测引擎"""

from typing import Optional, Dict, List, Any
from datetime import date, datetime
import pandas as pd

from .context import BacktestContext
from .matcher import OrderMatcher
from .report import BacktestReport
from ..strategy.base import Strategy, Signal
from ..execution.order import Order, OrderStatus
from config.trading import TradingConfig


class BacktestEngine:
    """回测引擎"""

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        commission: Optional[float] = None,
        slippage: Optional[float] = None,
        trading_config: Optional[TradingConfig] = None,
    ):
        """
        初始化回测引擎

        Args:
            initial_capital: 初始资金
            commission: 佣金率（可选，默认使用TradingConfig）
            slippage: 滑点（可选）
            trading_config: 交易配置
        """
        self.initial_capital = initial_capital
        self.trading_config = trading_config or TradingConfig()

        if commission is not None:
            self.trading_config.commission_rate = commission
        if slippage is not None:
            self.trading_config.slippage = slippage

        self.context: Optional[BacktestContext] = None
        self.matcher = OrderMatcher(self.trading_config)
        self.strategy: Optional[Strategy] = None

        # 回测结果
        self._daily_returns: List[Dict] = []
        self._trades: List[Dict] = []
        self._orders: List[Order] = []

    def run(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        benchmark: Optional[pd.DataFrame] = None,
    ) -> BacktestReport:
        """
        运行回测

        Args:
            strategy: 策略实例
            data: 行情数据（需包含: code, trade_date, open, high, low, close, volume）
            start_date: 开始日期
            end_date: 结束日期
            benchmark: 基准数据

        Returns:
            回测报告
        """
        self.strategy = strategy

        # 数据预处理
        data = self._prepare_data(data, start_date, end_date)
        trade_dates = sorted(data["trade_date"].unique())

        # 初始化上下文
        self.context = BacktestContext(
            initial_capital=self.initial_capital,
            start_date=trade_dates[0],
            end_date=trade_dates[-1],
        )

        # 策略初始化
        strategy.init(self.context.to_strategy_context())

        # 逐日回测
        for current_date in trade_dates:
            self._process_day(current_date, data)

        # 生成报告
        return self._generate_report(benchmark)

    def _prepare_data(
        self,
        data: pd.DataFrame,
        start_date: Optional[date],
        end_date: Optional[date],
    ) -> pd.DataFrame:
        """预处理数据"""
        df = data.copy()

        # 确保日期格式
        if not pd.api.types.is_datetime64_any_dtype(df["trade_date"]):
            df["trade_date"] = pd.to_datetime(df["trade_date"])

        df["trade_date"] = df["trade_date"].dt.date

        # 日期过滤
        if start_date:
            df = df[df["trade_date"] >= start_date]
        if end_date:
            df = df[df["trade_date"] <= end_date]

        return df.sort_values("trade_date")

    def _process_day(self, current_date: date, data: pd.DataFrame) -> None:
        """处理单日"""
        self.context.current_date = current_date

        # 获取当日数据
        daily_data = data[data["trade_date"] == current_date]

        # 盘前处理
        self.strategy.before_trading(self.context.to_strategy_context())

        # 更新持仓市值
        self._update_positions(daily_data)

        # 策略处理
        signals = self.strategy.handle_data(
            self.context.to_strategy_context(), daily_data
        )

        # 处理信号生成订单
        orders = self._process_signals(signals, daily_data)

        # 撮合订单
        for order in orders:
            self._execute_order(order, daily_data)

        # 盘后处理
        self.strategy.after_trading(self.context.to_strategy_context())

        # 记录每日收益
        self._record_daily(current_date)

    def _update_positions(self, daily_data: pd.DataFrame) -> None:
        """更新持仓市值"""
        for code, position in self.context.positions.items():
            stock_data = daily_data[daily_data["code"] == code]
            if not stock_data.empty:
                price = stock_data["close"].iloc[0]
                position.current_price = price
                position.market_value = price * position.quantity

    def _process_signals(
        self, signals: List[Signal], daily_data: pd.DataFrame
    ) -> List[Order]:
        """处理信号生成订单"""
        orders = []

        for signal in signals:
            stock_data = daily_data[daily_data["code"] == signal.code]
            if stock_data.empty:
                continue

            price = signal.price or stock_data["close"].iloc[0]

            if signal.direction == "buy":
                # 计算可买数量
                available_cash = self.context.cash * signal.strength
                quantity = self._calculate_buy_quantity(available_cash, price)

                if quantity > 0:
                    order = Order(
                        code=signal.code,
                        direction="buy",
                        quantity=quantity,
                        price=price,
                        order_type="limit",
                    )
                    orders.append(order)

            elif signal.direction == "sell":
                # 获取持仓数量
                position = self.context.positions.get(signal.code)
                if position and position.quantity > 0:
                    sell_quantity = int(position.quantity * signal.strength)
                    sell_quantity = (sell_quantity // 100) * 100  # 整手

                    if sell_quantity > 0:
                        order = Order(
                            code=signal.code,
                            direction="sell",
                            quantity=sell_quantity,
                            price=price,
                            order_type="limit",
                        )
                        orders.append(order)

        return orders

    def _calculate_buy_quantity(self, cash: float, price: float) -> int:
        """计算可买数量（整手）"""
        # 考虑交易费用
        fee_rate = self.trading_config.commission_rate + self.trading_config.slippage
        effective_cash = cash / (1 + fee_rate)

        quantity = int(effective_cash / price)
        return (quantity // 100) * 100  # 整手

    def _execute_order(self, order: Order, daily_data: pd.DataFrame) -> None:
        """执行订单"""
        filled_order = self.matcher.match(order, daily_data)
        self._orders.append(filled_order)

        if filled_order.status == OrderStatus.FILLED:
            self._update_position_on_fill(filled_order)
            self._record_trade(filled_order)

            # 回调
            self.strategy.on_order_filled(
                self.context.to_strategy_context(),
                filled_order.to_dict(),
            )
        else:
            self.strategy.on_order_rejected(
                self.context.to_strategy_context(),
                filled_order.to_dict(),
                filled_order.reject_reason or "Unknown",
            )

    def _update_position_on_fill(self, order: Order) -> None:
        """订单成交后更新持仓"""
        from ..account.portfolio import Position

        code = order.code
        filled_price = order.filled_price
        filled_quantity = order.filled_quantity
        commission = order.commission

        if order.direction == "buy":
            # 扣除资金
            cost = filled_price * filled_quantity + commission
            self.context.cash -= cost

            # 更新持仓
            if code in self.context.positions:
                pos = self.context.positions[code]
                total_cost = pos.avg_cost * pos.quantity + filled_price * filled_quantity
                pos.quantity += filled_quantity
                pos.avg_cost = total_cost / pos.quantity
            else:
                self.context.positions[code] = Position(
                    code=code,
                    quantity=filled_quantity,
                    avg_cost=filled_price,
                    current_price=filled_price,
                )

        elif order.direction == "sell":
            # 增加资金
            revenue = filled_price * filled_quantity - commission
            self.context.cash += revenue

            # 更新持仓
            if code in self.context.positions:
                pos = self.context.positions[code]
                pos.quantity -= filled_quantity

                if pos.quantity <= 0:
                    del self.context.positions[code]

    def _record_trade(self, order: Order) -> None:
        """记录交易"""
        self._trades.append(
            {
                "date": self.context.current_date,
                "code": order.code,
                "direction": order.direction,
                "quantity": order.filled_quantity,
                "price": order.filled_price,
                "amount": order.filled_price * order.filled_quantity,
                "commission": order.commission,
            }
        )

    def _record_daily(self, current_date: date) -> None:
        """记录每日数据"""
        portfolio_value = self.context.portfolio_value

        self._daily_returns.append(
            {
                "date": current_date,
                "cash": self.context.cash,
                "position_value": sum(
                    p.market_value for p in self.context.positions.values()
                ),
                "portfolio_value": portfolio_value,
                "return": (portfolio_value / self.initial_capital) - 1,
            }
        )

    def _generate_report(
        self, benchmark: Optional[pd.DataFrame] = None
    ) -> BacktestReport:
        """生成回测报告"""
        daily_df = pd.DataFrame(self._daily_returns)
        trades_df = pd.DataFrame(self._trades)

        return BacktestReport(
            daily_data=daily_df,
            trades=trades_df,
            initial_capital=self.initial_capital,
            final_value=self.context.portfolio_value,
            benchmark=benchmark,
        )
