"""资金账簿 — 活期账户、流水、定投计划、定投成交。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class CashAccount:
    """活期资金账户 — 支持多账户 (不同策略/子组合各开一本)。"""

    TABLE: ClassVar[str] = "cash_account"
    PK: ClassVar[tuple[str, ...]] = ("account_id",)

    account_id: str
    name: str
    created_at: str
    memo: str | None = None


@dataclass(frozen=True, slots=True)
class CashTxn:
    """资金流水 — 账户余额的单一真相来源 (事件溯源，balance=SUM(amount))。"""

    TABLE: ClassVar[str] = "cash_txn"
    PK: ClassVar[tuple[str, ...]] = ("txn_id",)

    txn_id: str
    account_id: str
    ts: str  # ISO 时间戳
    txn_type: str  # TxnType.value
    amount: float  # 有符号: +入账 -出账
    balance_after: float  # 执行后余额 (冗余，便于审计)
    ref_plan_id: str | None = None  # 关联定投计划 (DCA_BUY 时)
    ref_exec_id: str | None = None  # 关联定投成交 (DCA_BUY 时)
    memo: str | None = None


@dataclass(frozen=True, slots=True)
class DcaPlan:
    """定投计划 (Dollar Cost Averaging)。"""

    TABLE: ClassVar[str] = "dca_plan"
    PK: ClassVar[tuple[str, ...]] = ("plan_id",)

    plan_id: str
    account_id: str
    code: str  # 标的代码 (如 510300.SH)
    amount_per_run: float  # 每次扣款金额
    frequency: str  # DcaFrequency.value
    start_date: str
    next_run_date: str  # 下次应执行日 (按 frequency 推进)
    status: str  # PlanStatus.value
    end_date: str | None = None  # 终止日 (None=无限期)
    memo: str | None = None


@dataclass(frozen=True, slots=True)
class DcaExecution:
    """定投每次扣款成交记录。"""

    TABLE: ClassVar[str] = "dca_execution"
    PK: ClassVar[tuple[str, ...]] = ("exec_id",)

    exec_id: str
    plan_id: str
    account_id: str
    code: str
    trade_date: str
    price: float  # 成交价
    shares: float  # 实际买入份额 (= amount / price)
    amount: float  # 成交金额 (= price * shares)
    fee: float  # 手续费
    total_cost: float  # = amount + fee (实际扣现金)
