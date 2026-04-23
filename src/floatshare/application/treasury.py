"""资金账簿服务 — 活期现金 + 定投 (DCA)。

设计原则:
- 事件溯源 — CashTxn 是余额的真相来源, balance = SUM(amount)
- 持仓由 DcaExecution 聚合推导, 不单独建表 (避免冗余)
- 价格由外部传入 (Treasury 与行情源解耦)
- 金额按 A 股实务精度 (分) 容差 1e-6

支持的动作:
    open_account   开户
    deposit        存钱
    withdraw       取钱 (余额检查)
    create_plan    创建定投计划
    pause/resume/stop_plan   计划状态切换
    execute_plan   扣款买入一次 (外部传价格, 推进 next_run_date)
    list_due_plans 扫描到期该跑的计划
    portfolio      当前持仓视图

利率/费率约定:
- frequency 推进: daily=+1d, weekly=+7d, biweekly=+14d, monthly=次月同日 (夹月末)
- fee 由调用方传入 (TradingConfig 可用, 但本模块不强依赖)
"""

from __future__ import annotations

import uuid
from calendar import monthrange
from dataclasses import asdict, replace
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pandas as pd

from floatshare.domain.enums import DcaFrequency, PlanStatus, TxnType
from floatshare.domain.records import (
    CashAccount,
    CashTxn,
    DcaExecution,
    DcaPlan,
)

if TYPE_CHECKING:
    from floatshare.infrastructure.storage.database import DatabaseStorage


_BALANCE_EPS = 1e-6


class AccountNotFound(ValueError):
    """账户未开立。"""


class InsufficientFunds(ValueError):
    """余额不足。"""


class PlanNotFound(ValueError):
    """定投计划不存在。"""


class PlanNotActive(ValueError):
    """计划非 ACTIVE — 不能执行扣款。"""


class Treasury:
    """资金账簿。"""

    def __init__(self, db: DatabaseStorage) -> None:
        self._db = db

    # ========================================================================
    # 账户
    # ========================================================================

    def open_account(
        self,
        account_id: str,
        name: str,
        memo: str | None = None,
    ) -> CashAccount:
        acc = CashAccount(
            account_id=account_id,
            name=name,
            created_at=_now_iso(),
            memo=memo,
        )
        self._save_one(acc)
        return acc

    def balance(self, account_id: str) -> float:
        """按流水 SUM 计算活期余额 (真相来源)。"""
        return self._sum_txn_amount(account_id)

    def total_invested(self, account_id: str) -> float:
        """总本金 = deposit - withdraw 的净额 (不含 DCA_BUY 内部转化)。"""
        return self._sum_txn_amount(account_id, types=("deposit", "withdraw"))

    def _sum_txn_amount(
        self,
        account_id: str,
        types: tuple[str, ...] | None = None,
    ) -> float:
        from sqlalchemy import text

        clauses = ["account_id = :aid"]
        params: dict[str, object] = {"aid": account_id}
        if types:
            placeholders = ", ".join(f":t{i}" for i in range(len(types)))
            clauses.append(f"txn_type IN ({placeholders})")
            params.update({f"t{i}": t for i, t in enumerate(types)})
        query = text(
            f"SELECT COALESCE(SUM(amount), 0) FROM {CashTxn.TABLE} WHERE {' AND '.join(clauses)}"
        )
        with self._db.engine.connect() as conn:
            row = conn.execute(query, params).first()
        return float(row[0]) if row else 0.0

    # ========================================================================
    # 存取
    # ========================================================================

    def deposit(
        self,
        account_id: str,
        amount: float,
        memo: str | None = None,
    ) -> CashTxn:
        if amount <= 0:
            raise ValueError(f"deposit amount must be positive, got {amount}")
        self._require_account(account_id)
        return self._append_txn(account_id, TxnType.DEPOSIT, +amount, memo=memo)

    def withdraw(
        self,
        account_id: str,
        amount: float,
        memo: str | None = None,
    ) -> CashTxn:
        if amount <= 0:
            raise ValueError(f"withdraw amount must be positive, got {amount}")
        self._require_account(account_id)
        self._require_balance(account_id, amount)
        return self._append_txn(account_id, TxnType.WITHDRAW, -amount, memo=memo)

    # ========================================================================
    # 定投计划
    # ========================================================================

    def create_plan(
        self,
        plan_id: str,
        account_id: str,
        code: str,
        amount_per_run: float,
        frequency: DcaFrequency,
        start_date: date,
        end_date: date | None = None,
        memo: str | None = None,
    ) -> DcaPlan:
        if amount_per_run <= 0:
            raise ValueError(f"amount_per_run must be positive, got {amount_per_run}")
        self._require_account(account_id)
        start_iso = start_date.isoformat()
        plan = DcaPlan(
            plan_id=plan_id,
            account_id=account_id,
            code=code,
            amount_per_run=amount_per_run,
            frequency=frequency.value,
            start_date=start_iso,
            next_run_date=start_iso,
            status=PlanStatus.ACTIVE.value,
            end_date=end_date.isoformat() if end_date else None,
            memo=memo,
        )
        self._save_one(plan)
        return plan

    def pause_plan(self, plan_id: str) -> DcaPlan:
        return self._transition_plan(plan_id, PlanStatus.PAUSED, from_={PlanStatus.ACTIVE})

    def resume_plan(self, plan_id: str) -> DcaPlan:
        return self._transition_plan(plan_id, PlanStatus.ACTIVE, from_={PlanStatus.PAUSED})

    def stop_plan(self, plan_id: str) -> DcaPlan:
        return self._transition_plan(
            plan_id,
            PlanStatus.STOPPED,
            from_={PlanStatus.ACTIVE, PlanStatus.PAUSED},
        )

    # ========================================================================
    # 执行
    # ========================================================================

    def execute_plan(
        self,
        plan_id: str,
        trade_date: date,
        price: float,
        fee: float = 0.0,
    ) -> DcaExecution:
        """执行一次定投扣款：扣现金 + 记成交 + 推进 next_run_date。"""
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")
        if fee < 0:
            raise ValueError(f"fee must be non-negative, got {fee}")

        plan = self._load_plan(plan_id)
        if plan is None:
            raise PlanNotFound(f"plan not found: {plan_id}")
        if plan.status != PlanStatus.ACTIVE.value:
            raise PlanNotActive(f"plan {plan_id} is {plan.status}, not ACTIVE")

        amount = plan.amount_per_run
        total_cost = amount + fee
        self._require_balance(plan.account_id, total_cost)

        shares = amount / price
        exec_id = f"E{uuid.uuid4().hex[:14]}"
        execution = DcaExecution(
            exec_id=exec_id,
            plan_id=plan_id,
            account_id=plan.account_id,
            code=plan.code,
            trade_date=trade_date.isoformat(),
            price=price,
            shares=shares,
            amount=amount,
            fee=fee,
            total_cost=total_cost,
        )
        self._save_one(execution)

        self._append_txn(
            plan.account_id,
            TxnType.DCA_BUY,
            -total_cost,
            ref_plan_id=plan_id,
            ref_exec_id=exec_id,
            memo=f"定投 {plan.code} {shares:.4f}股 @ {price:.4f}",
        )

        # 推进 next_run_date；若超过 end_date 则自动 STOPPED
        next_run = _advance(trade_date, DcaFrequency(plan.frequency))
        new_status = plan.status
        if plan.end_date and next_run > date.fromisoformat(plan.end_date):
            new_status = PlanStatus.STOPPED.value
        updated = replace(plan, next_run_date=next_run.isoformat(), status=new_status)
        self._save_one(updated)
        return execution

    def list_due_plans(self, as_of: date) -> list[DcaPlan]:
        """返回 status=ACTIVE 且 next_run_date <= as_of 的计划。"""
        from sqlalchemy import text

        query = text(
            f"SELECT * FROM {DcaPlan.TABLE} "
            f"WHERE status = :s AND next_run_date <= :d "
            f"ORDER BY next_run_date"
        )
        params = {"s": PlanStatus.ACTIVE.value, "d": as_of.isoformat()}
        with self._db.engine.connect() as conn:
            rows = conn.execute(query, params).mappings().all()
        return [_plan_from_row(dict(r)) for r in rows]

    # ========================================================================
    # 持仓视图
    # ========================================================================

    def portfolio(self, account_id: str) -> pd.DataFrame:
        """定投累计持仓：按 code 聚合 shares / avg_cost / total_cost。"""
        from sqlalchemy import text

        query = text(
            f"""
            SELECT
                code,
                SUM(shares) AS shares,
                SUM(amount) AS total_amount,
                SUM(fee) AS total_fee,
                SUM(total_cost) AS total_cost
            FROM {DcaExecution.TABLE}
            WHERE account_id = :aid
            GROUP BY code
            ORDER BY code
            """,
        )
        try:
            df = pd.read_sql(query, self._db.engine, params={"aid": account_id})
        except Exception:
            return pd.DataFrame()
        if df.empty:
            return df
        # avg_cost = total_amount / shares (不含手续费的"持仓成本价")
        df["avg_cost"] = df["total_amount"] / df["shares"]
        return df.reindex(
            columns=["code", "shares", "avg_cost", "total_amount", "total_fee", "total_cost"],
        )

    # ========================================================================
    # 内部
    # ========================================================================

    def _append_txn(
        self,
        account_id: str,
        txn_type: TxnType,
        amount: float,
        ref_plan_id: str | None = None,
        ref_exec_id: str | None = None,
        memo: str | None = None,
    ) -> CashTxn:
        current = self.balance(account_id)
        new_bal = current + amount
        txn = CashTxn(
            txn_id=f"X{uuid.uuid4().hex[:14]}",
            account_id=account_id,
            ts=_now_iso(),
            txn_type=txn_type.value,
            amount=amount,
            balance_after=new_bal,
            ref_plan_id=ref_plan_id,
            ref_exec_id=ref_exec_id,
            memo=memo,
        )
        self._save_one(txn)
        return txn

    def _save_one(self, record: Any) -> None:
        """UPSERT 一条 dataclass 记录 — 类别由 type(record) 自动推导。"""
        self._db.save(type(record), pd.DataFrame([asdict(record)]))

    def _require_account(self, account_id: str) -> None:
        if not self._db.has_rows(CashAccount.TABLE, account_id=account_id):
            raise AccountNotFound(f"账户 {account_id} 未开立，请先 open_account")

    def _require_balance(self, account_id: str, needed: float) -> None:
        bal = self.balance(account_id)
        if bal + _BALANCE_EPS < needed:
            raise InsufficientFunds(
                f"账户 {account_id} 余额 {bal:.2f} < 需要 {needed:.2f}",
            )

    def _load_plan(self, plan_id: str) -> DcaPlan | None:
        from sqlalchemy import text

        query = text(f"SELECT * FROM {DcaPlan.TABLE} WHERE plan_id = :pid")
        with self._db.engine.connect() as conn:
            row = conn.execute(query, {"pid": plan_id}).mappings().first()
        return _plan_from_row(dict(row)) if row else None

    def _transition_plan(
        self,
        plan_id: str,
        to: PlanStatus,
        from_: set[PlanStatus],
    ) -> DcaPlan:
        plan = self._load_plan(plan_id)
        if plan is None:
            raise PlanNotFound(f"plan not found: {plan_id}")
        if PlanStatus(plan.status) not in from_:
            raise PlanNotActive(
                f"plan {plan_id} is {plan.status}, cannot transition to {to.value}",
            )
        updated = replace(plan, status=to.value)
        self._save_one(updated)
        return updated


# ============================================================================
# 纯函数 helpers
# ============================================================================


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


_FIXED_DELTA_DAYS: dict[DcaFrequency, int] = {
    DcaFrequency.DAILY: 1,
    DcaFrequency.WEEKLY: 7,
    DcaFrequency.BIWEEKLY: 14,
}


def _advance(current: date, freq: DcaFrequency) -> date:
    """按 frequency 从当前日推进到下次执行日。"""
    delta = _FIXED_DELTA_DAYS.get(freq)
    if delta is not None:
        return current + timedelta(days=delta)
    # MONTHLY 的月历算术特殊 (月末夹紧)
    return _add_month(current)


def _add_month(d: date) -> date:
    """加一个月，日号保留，月末夹紧 (1/31 + 1月 = 2/28)。"""
    y, m = d.year, d.month + 1
    if m > 12:
        y, m = y + 1, 1
    last_day = monthrange(y, m)[1]
    return date(y, m, min(d.day, last_day))


def _plan_from_row(row: dict[str, Any]) -> DcaPlan:
    """从 SQL 行构造 DcaPlan — 走共享 from_row helper (消除重复)."""
    from floatshare.infrastructure.storage.schema_sql import from_row

    return from_row(DcaPlan, row)
