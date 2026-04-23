"""资金账簿 Treasury — 存取 + 定投 单元测试。"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from floatshare.application.treasury import (
    AccountNotFound,
    InsufficientFunds,
    PlanNotActive,
    PlanNotFound,
    Treasury,
    _add_month,
)
from floatshare.domain.enums import DcaFrequency, PlanStatus
from floatshare.infrastructure.storage.database import DatabaseStorage


@pytest.fixture
def book(tmp_path):
    db = DatabaseStorage(db_path=tmp_path / "book.db")
    db.init_tables()
    t = Treasury(db)
    t.open_account("A", name="主账户")
    return t


# ============================================================================
# 存取
# ============================================================================


class TestDepositWithdraw:
    def test_deposit_updates_balance(self, book):
        book.deposit("A", 100_000)
        assert book.balance("A") == pytest.approx(100_000)

    def test_multiple_deposits_accumulate(self, book):
        book.deposit("A", 100)
        book.deposit("A", 200)
        assert book.balance("A") == pytest.approx(300)

    def test_withdraw_reduces_balance(self, book):
        book.deposit("A", 1_000)
        book.withdraw("A", 400)
        assert book.balance("A") == pytest.approx(600)

    def test_withdraw_insufficient_raises(self, book):
        book.deposit("A", 100)
        with pytest.raises(InsufficientFunds):
            book.withdraw("A", 200)

    def test_deposit_non_positive_raises(self, book):
        with pytest.raises(ValueError, match="positive"):
            book.deposit("A", 0)
        with pytest.raises(ValueError, match="positive"):
            book.deposit("A", -1)

    def test_deposit_on_unknown_account_raises(self, book):
        with pytest.raises(AccountNotFound):
            book.deposit("NOPE", 100)

    def test_balance_on_unknown_account_is_zero(self, book):
        # 未开户但查余额不报错，返回 0 (事件溯源语义)
        assert book.balance("NOPE") == 0.0

    def test_total_invested_excludes_dca_flows(self, book):
        book.deposit("A", 100_000)
        book.create_plan("P", "A", "510300.SH", 1000, DcaFrequency.WEEKLY, date(2026, 1, 1))
        book.execute_plan("P", date(2026, 1, 1), price=4.0)  # DCA_BUY -1000
        # balance 受 DCA 影响, total_invested 只看外部存取
        assert book.balance("A") < 100_000
        assert book.total_invested("A") == pytest.approx(100_000)
        book.withdraw("A", 5_000)
        assert book.total_invested("A") == pytest.approx(95_000)


# ============================================================================
# 定投计划
# ============================================================================


class TestDcaPlan:
    def test_create_plan_sets_next_run_to_start(self, book):
        start = date(2026, 1, 1)
        plan = book.create_plan(
            plan_id="P1",
            account_id="A",
            code="510300.SH",
            amount_per_run=1000,
            frequency=DcaFrequency.MONTHLY,
            start_date=start,
        )
        assert plan.next_run_date == "2026-01-01"
        assert plan.status == PlanStatus.ACTIVE.value

    def test_create_plan_negative_amount_raises(self, book):
        with pytest.raises(ValueError, match="positive"):
            book.create_plan(
                plan_id="P1",
                account_id="A",
                code="510300.SH",
                amount_per_run=0,
                frequency=DcaFrequency.WEEKLY,
                start_date=date(2026, 1, 1),
            )

    def test_pause_resume_stop(self, book):
        book.create_plan(
            "P1",
            "A",
            "510300.SH",
            1000,
            DcaFrequency.MONTHLY,
            date(2026, 1, 1),
        )
        assert book.pause_plan("P1").status == PlanStatus.PAUSED.value
        assert book.resume_plan("P1").status == PlanStatus.ACTIVE.value
        assert book.stop_plan("P1").status == PlanStatus.STOPPED.value

    def test_stop_then_resume_raises(self, book):
        book.create_plan(
            "P1",
            "A",
            "510300.SH",
            1000,
            DcaFrequency.MONTHLY,
            date(2026, 1, 1),
        )
        book.stop_plan("P1")
        with pytest.raises(PlanNotActive):
            book.resume_plan("P1")

    def test_unknown_plan_raises(self, book):
        with pytest.raises(PlanNotFound):
            book.pause_plan("GHOST")


# ============================================================================
# 定投执行
# ============================================================================


class TestDcaExecute:
    def test_execute_deducts_cash_and_buys_shares(self, book):
        book.deposit("A", 10_000)
        book.create_plan(
            "P1",
            "A",
            "510300.SH",
            1000,
            DcaFrequency.MONTHLY,
            date(2026, 1, 1),
        )
        ex = book.execute_plan(
            "P1",
            trade_date=date(2026, 1, 5),
            price=4.0,
            fee=2.5,
        )
        assert ex.shares == pytest.approx(250)  # 1000 / 4
        assert ex.amount == pytest.approx(1000)
        assert ex.fee == pytest.approx(2.5)
        assert ex.total_cost == pytest.approx(1002.5)
        assert book.balance("A") == pytest.approx(10_000 - 1002.5)

    def test_execute_insufficient_raises(self, book):
        book.deposit("A", 500)
        book.create_plan(
            "P1",
            "A",
            "510300.SH",
            1000,
            DcaFrequency.WEEKLY,
            date(2026, 1, 1),
        )
        with pytest.raises(InsufficientFunds):
            book.execute_plan("P1", date(2026, 1, 5), price=10.0)

    def test_execute_advances_next_run_weekly(self, book):
        book.deposit("A", 10_000)
        book.create_plan(
            "P1",
            "A",
            "510300.SH",
            1000,
            DcaFrequency.WEEKLY,
            date(2026, 1, 1),
        )
        book.execute_plan("P1", date(2026, 1, 5), price=5.0)
        # next_run_date = trade_date + 7天 = 2026-01-12
        due = book.list_due_plans(as_of=date(2026, 1, 11))
        assert due == []
        due = book.list_due_plans(as_of=date(2026, 1, 12))
        assert len(due) == 1

    def test_execute_advances_next_run_monthly(self, book):
        book.deposit("A", 10_000)
        book.create_plan(
            "P1",
            "A",
            "510300.SH",
            1000,
            DcaFrequency.MONTHLY,
            date(2026, 1, 31),
        )
        book.execute_plan("P1", date(2026, 1, 31), price=5.0)
        # 下次执行日 = 2026-02-28 (2月末夹紧)
        due = book.list_due_plans(as_of=date(2026, 2, 27))
        assert due == []
        due = book.list_due_plans(as_of=date(2026, 2, 28))
        assert len(due) == 1
        assert due[0].next_run_date == "2026-02-28"

    def test_execute_past_end_date_auto_stops(self, book):
        book.deposit("A", 10_000)
        book.create_plan(
            "P1",
            "A",
            "510300.SH",
            1000,
            DcaFrequency.WEEKLY,
            date(2026, 1, 1),
            end_date=date(2026, 1, 3),
        )
        book.execute_plan("P1", date(2026, 1, 1), price=5.0)
        # 下次执行日=01-08, 已超 end_date 01-03 → 自动 STOPPED
        assert book.list_due_plans(as_of=date(2026, 2, 1)) == []

    def test_execute_on_stopped_plan_raises(self, book):
        book.deposit("A", 10_000)
        book.create_plan(
            "P1",
            "A",
            "510300.SH",
            1000,
            DcaFrequency.WEEKLY,
            date(2026, 1, 1),
        )
        book.stop_plan("P1")
        with pytest.raises(PlanNotActive):
            book.execute_plan("P1", date(2026, 1, 5), price=5.0)

    def test_execute_unknown_plan_raises(self, book):
        with pytest.raises(PlanNotFound):
            book.execute_plan("GHOST", date(2026, 1, 5), price=5.0)

    def test_negative_price_raises(self, book):
        book.deposit("A", 10_000)
        book.create_plan(
            "P1",
            "A",
            "510300.SH",
            1000,
            DcaFrequency.WEEKLY,
            date(2026, 1, 1),
        )
        with pytest.raises(ValueError, match="price"):
            book.execute_plan("P1", date(2026, 1, 5), price=0)


# ============================================================================
# 持仓视图
# ============================================================================


class TestPortfolio:
    def test_portfolio_aggregates_shares_and_avg_cost(self, book):
        book.deposit("A", 100_000)
        book.create_plan(
            "P1",
            "A",
            "510300.SH",
            1000,
            DcaFrequency.WEEKLY,
            date(2026, 1, 1),
        )
        book.execute_plan("P1", date(2026, 1, 1), price=4.0, fee=1.0)  # 250 股
        book.execute_plan("P1", date(2026, 1, 8), price=5.0, fee=1.0)  # 200 股
        # 合计 450 股, total_amount=2000, avg_cost=2000/450=4.444...
        df = book.portfolio("A")
        assert len(df) == 1
        row = df.iloc[0]
        assert row["code"] == "510300.SH"
        assert row["shares"] == pytest.approx(450)
        assert row["total_amount"] == pytest.approx(2000)
        assert row["total_fee"] == pytest.approx(2.0)
        assert row["total_cost"] == pytest.approx(2002.0)
        assert row["avg_cost"] == pytest.approx(2000 / 450)

    def test_portfolio_multiple_codes(self, book):
        book.deposit("A", 100_000)
        book.create_plan("P1", "A", "510300.SH", 1000, DcaFrequency.WEEKLY, date(2026, 1, 1))
        book.create_plan("P2", "A", "510500.SH", 500, DcaFrequency.WEEKLY, date(2026, 1, 1))
        book.execute_plan("P1", date(2026, 1, 1), price=4.0)
        book.execute_plan("P2", date(2026, 1, 1), price=2.0)
        df = book.portfolio("A")
        assert len(df) == 2
        assert set(df["code"]) == {"510300.SH", "510500.SH"}

    def test_portfolio_empty(self, book):
        assert book.portfolio("A").empty


# ============================================================================
# 月末夹紧 helper
# ============================================================================


class TestAddMonth:
    def test_mid_month(self):
        assert _add_month(date(2026, 1, 15)) == date(2026, 2, 15)

    def test_jan_31_to_feb_28(self):
        assert _add_month(date(2026, 1, 31)) == date(2026, 2, 28)

    def test_jan_31_to_feb_29_leap(self):
        # 2024 是闰年
        assert _add_month(date(2024, 1, 31)) == date(2024, 2, 29)

    def test_dec_to_jan_next_year(self):
        assert _add_month(date(2026, 12, 20)) == date(2027, 1, 20)


# ============================================================================
# 跨账户隔离
# ============================================================================


class TestIsolation:
    def test_two_accounts_balances_independent(self, book):
        book.open_account("B", name="账户二")
        book.deposit("A", 100)
        book.deposit("B", 999)
        assert book.balance("A") == pytest.approx(100)
        assert book.balance("B") == pytest.approx(999)

    def test_withdraw_does_not_cross_accounts(self, book):
        book.open_account("B", name="账户二")
        book.deposit("A", 100)
        with pytest.raises(InsufficientFunds):
            book.withdraw("B", 50)  # B 没钱
        assert book.balance("A") == pytest.approx(100)


# ============================================================================
# 频率推进 (unit on hidden helper via execute + next_run)
# ============================================================================


class TestFrequencyAdvance:
    @pytest.mark.parametrize(
        ("freq", "delta_days"),
        [
            (DcaFrequency.DAILY, 1),
            (DcaFrequency.WEEKLY, 7),
            (DcaFrequency.BIWEEKLY, 14),
        ],
    )
    def test_fixed_deltas(self, book, freq, delta_days):
        book.deposit("A", 10_000)
        book.create_plan(
            "P1",
            "A",
            "510300.SH",
            100,
            freq,
            date(2026, 1, 1),
        )
        book.execute_plan("P1", date(2026, 1, 1), price=1.0)
        expected = (date(2026, 1, 1) + timedelta(days=delta_days)).isoformat()
        plan = book.list_due_plans(as_of=date(2027, 1, 1))[0]
        assert plan.next_run_date == expected
