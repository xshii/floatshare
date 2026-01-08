"""组合管理测试"""

import pytest
from datetime import datetime

from src.account.portfolio import Portfolio, Position, CashFlow, FlowType


class TestPosition:
    """持仓测试"""

    def test_basic_position(self):
        pos = Position(code="600000.SH", quantity=1000, avg_cost=10.0)
        assert pos.code == "600000.SH"
        assert pos.quantity == 1000
        assert pos.avg_cost == 10.0
        assert pos.frozen == 0

    def test_available_quantity(self):
        pos = Position(code="600000.SH", quantity=1000, avg_cost=10.0, frozen=200)
        assert pos.available == 800

    def test_market_value(self):
        pos = Position(code="600000.SH", quantity=1000, avg_cost=10.0, current_price=12.0)
        assert pos.market_value == 12000.0

    def test_cost_value(self):
        pos = Position(code="600000.SH", quantity=1000, avg_cost=10.0)
        assert pos.cost_value == 10000.0

    def test_profit(self):
        pos = Position(code="600000.SH", quantity=1000, avg_cost=10.0, current_price=12.0)
        assert pos.profit == 2000.0  # 12000 - 10000

    def test_profit_pct(self):
        pos = Position(code="600000.SH", quantity=1000, avg_cost=10.0, current_price=12.0)
        assert pos.profit_pct == 0.2  # 20%

    def test_profit_pct_zero_cost(self):
        pos = Position(code="600000.SH", quantity=0, avg_cost=0.0)
        assert pos.profit_pct == 0.0


class TestPortfolio:
    """组合测试"""

    def test_initial_state(self):
        p = Portfolio(name="test", initial_capital=100_000)
        assert p.name == "test"
        assert p.initial_capital == 100_000
        assert p.cash == 100_000
        assert p.position_value == 0
        assert p.total_value == 100_000
        assert len(p.positions) == 0

    def test_add_position_new(self):
        """测试新建持仓"""
        p = Portfolio(initial_capital=100_000)
        pos = p.add_position("600000.SH", quantity=1000, price=10.0)

        assert pos.code == "600000.SH"
        assert pos.quantity == 1000
        assert pos.avg_cost == 10.0
        assert pos.current_price == 10.0
        assert "600000.SH" in p.positions

    def test_add_position_existing(self):
        """测试加仓"""
        p = Portfolio(initial_capital=100_000)
        p.add_position("600000.SH", quantity=1000, price=10.0)
        p.add_position("600000.SH", quantity=500, price=12.0)

        pos = p.positions["600000.SH"]
        assert pos.quantity == 1500
        # 平均成本: (10*1000 + 12*500) / 1500 = 16000/1500 ≈ 10.67
        assert abs(pos.avg_cost - 10.6667) < 0.01
        assert pos.current_price == 12.0

    def test_reduce_position(self):
        """测试减仓"""
        p = Portfolio(initial_capital=100_000)
        p.add_position("600000.SH", quantity=1000, price=10.0)

        result = p.reduce_position("600000.SH", quantity=300)
        assert result is True
        assert p.positions["600000.SH"].quantity == 700

    def test_reduce_position_to_zero(self):
        """测试清仓"""
        p = Portfolio(initial_capital=100_000)
        p.add_position("600000.SH", quantity=1000, price=10.0)

        result = p.reduce_position("600000.SH", quantity=1000)
        assert result is True
        assert "600000.SH" not in p.positions

    def test_reduce_position_insufficient(self):
        """测试减仓数量不足"""
        p = Portfolio(initial_capital=100_000)
        p.add_position("600000.SH", quantity=1000, price=10.0)

        result = p.reduce_position("600000.SH", quantity=1500)
        assert result is False
        assert p.positions["600000.SH"].quantity == 1000

    def test_has_position(self):
        p = Portfolio(initial_capital=100_000)
        assert p.has_position("600000.SH") is False

        p.add_position("600000.SH", quantity=1000, price=10.0)
        assert p.has_position("600000.SH") is True

    def test_update_price(self):
        p = Portfolio(initial_capital=100_000)
        p.add_position("600000.SH", quantity=1000, price=10.0)

        result = p.update_price("600000.SH", 12.0)
        assert result is True
        assert p.positions["600000.SH"].current_price == 12.0

    def test_update_prices_batch(self):
        p = Portfolio(initial_capital=100_000)
        p.add_position("600000.SH", quantity=1000, price=10.0)
        p.add_position("000001.SZ", quantity=500, price=20.0)

        p.update_prices({"600000.SH": 11.0, "000001.SZ": 22.0})

        assert p.positions["600000.SH"].current_price == 11.0
        assert p.positions["000001.SZ"].current_price == 22.0

    def test_position_value(self):
        p = Portfolio(initial_capital=100_000)
        p.add_position("600000.SH", quantity=1000, price=10.0)
        p.add_position("000001.SZ", quantity=500, price=20.0)

        assert p.position_value == 20000.0  # 1000*10 + 500*20

    def test_total_value(self):
        p = Portfolio(initial_capital=100_000)
        p.cash = 80_000
        p.add_position("600000.SH", quantity=1000, price=10.0)
        p.add_position("000001.SZ", quantity=500, price=20.0)

        assert p.total_value == 100_000  # 80000 + 20000

    def test_position_ratio(self):
        p = Portfolio(initial_capital=100_000)
        p.cash = 80_000
        p.add_position("600000.SH", quantity=1000, price=10.0)
        p.add_position("000001.SZ", quantity=500, price=20.0)

        assert p.position_ratio == 0.2  # 20000 / 100000

    def test_get_weights(self):
        p = Portfolio(initial_capital=100_000)
        p.cash = 80_000
        p.add_position("600000.SH", quantity=1000, price=10.0)
        p.add_position("000001.SZ", quantity=500, price=20.0)

        weights = p.get_weights()
        assert weights["600000.SH"] == 0.1  # 10000 / 100000
        assert weights["000001.SZ"] == 0.1  # 10000 / 100000

    def test_reset(self):
        p = Portfolio(initial_capital=100_000)
        p.cash = 50_000
        p.add_position("600000.SH", quantity=1000, price=10.0)
        p.deposit(10000)

        p.reset()

        assert p.cash == 100_000
        assert len(p.positions) == 0
        assert len(p.cash_flows) == 0


class TestBuySellWorkflow:
    """买入卖出完整流程测试"""

    def test_buy_stock(self):
        """测试买入股票"""
        p = Portfolio(name="test", initial_capital=100_000)

        # 买入 1000 股，价格 10 元
        buy_amount = 1000 * 10.0
        commission = buy_amount * 0.0003  # 万三手续费

        # 扣除股票金额（手动）
        p.cash -= buy_amount
        # 增加持仓
        p.add_position("600000.SH", quantity=1000, price=10.0)
        # 记录手续费（add_fee 会自动扣除现金）
        p.add_fee(commission, FlowType.COMMISSION, "600000.SH", "买入手续费")

        assert abs(p.cash - (100_000 - 10_000 - commission)) < 0.01
        assert p.positions["600000.SH"].quantity == 1000
        assert abs(p.total_fees - commission) < 0.01

    def test_sell_stock(self):
        """测试卖出股票"""
        p = Portfolio(name="test", initial_capital=100_000)

        # 先买入
        p.cash -= 10_000
        p.add_position("600000.SH", quantity=1000, price=10.0)

        # 卖出 500 股，价格 12 元
        sell_amount = 500 * 12.0
        commission = sell_amount * 0.0003  # 手续费
        stamp_tax = sell_amount * 0.001    # 印花税

        # 增加卖出金额（手动）
        p.cash += sell_amount
        # 减少持仓
        p.reduce_position("600000.SH", quantity=500)
        # 记录费用（add_fee 会自动扣除现金）
        p.add_fee(commission, FlowType.COMMISSION, "600000.SH", "卖出手续费")
        p.add_fee(stamp_tax, FlowType.STAMP_TAX, "600000.SH", "印花税")

        assert p.positions["600000.SH"].quantity == 500
        assert abs(p.cash - (90_000 + 6_000 - commission - stamp_tax)) < 0.01
        assert abs(p.total_fees - (commission + stamp_tax)) < 0.01

    def test_complete_trade_cycle(self):
        """测试完整交易周期：开户 -> 买入 -> 持有 -> 卖出"""
        # 1. 开户，初始资金 10 万
        p = Portfolio(name="my_account", initial_capital=100_000)
        assert p.cash == 100_000
        assert p.total_value == 100_000

        # 2. 买入浦发银行 1000 股，价格 10 元
        code = "600000.SH"
        buy_qty = 1000
        buy_price = 10.0
        buy_amount = buy_qty * buy_price
        buy_commission = max(buy_amount * 0.0003, 5)  # 最低5元

        p.cash -= buy_amount  # 扣除股票金额
        p.add_position(code, quantity=buy_qty, price=buy_price)
        p.add_fee(buy_commission, FlowType.COMMISSION, code, "买入手续费")

        assert p.has_position(code)
        assert p.positions[code].quantity == 1000
        assert p.position_value == 10_000
        assert abs(p.cash - (100_000 - 10_000 - 5)) < 0.01

        # 3. 股价上涨到 12 元
        p.update_price(code, 12.0)
        assert p.positions[code].market_value == 12_000
        assert p.positions[code].profit == 2_000
        assert p.positions[code].profit_pct == 0.2

        # 4. 卖出全部持仓
        sell_qty = 1000
        sell_price = 12.0
        sell_amount = sell_qty * sell_price
        sell_commission = max(sell_amount * 0.0003, 5)
        stamp_tax = sell_amount * 0.001

        p.cash += sell_amount  # 增加卖出金额
        p.reduce_position(code, quantity=sell_qty)
        p.add_fee(sell_commission, FlowType.COMMISSION, code, "卖出手续费")
        p.add_fee(stamp_tax, FlowType.STAMP_TAX, code, "印花税")

        # 5. 验证最终状态
        assert not p.has_position(code)
        assert p.position_value == 0

        # 计算最终现金
        # 初始: 100000
        # 买入: -10000 - 5 = -10005
        # 卖出: +12000 - 5 - 12 = +11983
        # 最终: 100000 - 10005 + 11983 = 101978
        expected_cash = 100_000 - buy_amount - buy_commission + sell_amount - sell_commission - stamp_tax
        assert abs(p.cash - expected_cash) < 0.01

        # 总收益 = 最终资产 - 初始资金
        profit = p.total_value - p.initial_capital
        assert profit > 0  # 赚钱了


class TestCashFlow:
    """资金流水测试"""

    def test_deposit(self):
        p = Portfolio(initial_capital=100_000)
        flow = p.deposit(50_000, "追加资金")

        assert p.cash == 150_000
        assert flow.amount == 50_000
        assert flow.flow_type == FlowType.DEPOSIT
        assert p.total_deposits == 50_000

    def test_deposit_invalid_amount(self):
        p = Portfolio(initial_capital=100_000)
        with pytest.raises(ValueError, match="入金金额必须为正数"):
            p.deposit(-1000)

    def test_withdraw(self):
        p = Portfolio(initial_capital=100_000)
        flow = p.withdraw(20_000, "提现")

        assert p.cash == 80_000
        assert flow.amount == -20_000
        assert flow.flow_type == FlowType.WITHDRAW
        assert p.total_withdrawals == 20_000

    def test_withdraw_insufficient(self):
        p = Portfolio(initial_capital=100_000)
        with pytest.raises(ValueError, match="余额不足"):
            p.withdraw(150_000)

    def test_receive_dividend(self):
        p = Portfolio(initial_capital=100_000)
        flows = p.receive_dividend("600000.SH", amount=1000, tax=100)

        assert p.cash == 100_900  # 100000 + 1000 - 100
        assert len(flows) == 2
        assert p.total_dividends == 1000
        assert p.total_taxes == 100

    def test_receive_dividend_no_tax(self):
        p = Portfolio(initial_capital=100_000)
        flows = p.receive_dividend("600000.SH", amount=1000)

        assert p.cash == 101_000
        assert len(flows) == 1
        assert p.total_dividends == 1000
        assert p.total_taxes == 0

    def test_deduct_tax(self):
        p = Portfolio(initial_capital=100_000)
        flow = p.deduct_tax("600000.SH", 50, "补扣红利税")

        assert p.cash == 99_950
        assert flow.flow_type == FlowType.TAX
        assert p.total_taxes == 50

    def test_add_fee(self):
        p = Portfolio(initial_capital=100_000)
        p.add_fee(5, FlowType.COMMISSION, "600000.SH", "手续费")
        p.add_fee(10, FlowType.STAMP_TAX, "600000.SH", "印花税")
        p.add_fee(1, FlowType.TRANSFER_FEE, "600000.SH", "过户费")

        assert p.cash == 99_984
        assert p.total_fees == 16

    def test_add_interest(self):
        p = Portfolio(initial_capital=100_000)
        flow = p.add_interest(50, "活期利息")

        assert p.cash == 100_050
        assert flow.flow_type == FlowType.INTEREST

    def test_cash_flow_summary(self):
        p = Portfolio(initial_capital=100_000)
        p.deposit(50_000)
        p.withdraw(10_000)
        p.receive_dividend("600000.SH", 1000, tax=100)
        p.add_fee(5, FlowType.COMMISSION)

        summary = p.get_cash_flow_summary()

        assert summary["total_deposits"] == 50_000
        assert summary["total_withdrawals"] == 10_000
        assert summary["net_cash_flow"] == 40_000
        assert summary["total_dividends"] == 1000
        assert summary["total_taxes"] == 100
        assert summary["total_fees"] == 5
        assert summary["flow_count"] == 5

    def test_net_cash_flow(self):
        p = Portfolio(initial_capital=100_000)
        p.deposit(30_000)
        p.deposit(20_000)
        p.withdraw(10_000)

        assert p.net_cash_flow == 40_000  # 30000 + 20000 - 10000
