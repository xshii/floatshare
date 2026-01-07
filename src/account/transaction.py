"""交易记录"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime, date
from enum import Enum
import pandas as pd


class TransactionType(Enum):
    """交易类型"""

    BUY = "buy"
    SELL = "sell"
    DIVIDEND = "dividend"  # 分红
    SPLIT = "split"  # 拆股
    TRANSFER_IN = "transfer_in"  # 转入
    TRANSFER_OUT = "transfer_out"  # 转出


@dataclass
class Transaction:
    """交易记录"""

    code: str
    type: TransactionType
    quantity: int
    price: float
    amount: float
    commission: float = 0.0
    tax: float = 0.0
    datetime: datetime = field(default_factory=datetime.now)
    order_id: Optional[str] = None
    remarks: str = ""

    @property
    def net_amount(self) -> float:
        """净金额（扣除费用）"""
        if self.type == TransactionType.BUY:
            return -(self.amount + self.commission + self.tax)
        elif self.type == TransactionType.SELL:
            return self.amount - self.commission - self.tax
        return self.amount

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "code": self.code,
            "type": self.type.value,
            "quantity": self.quantity,
            "price": self.price,
            "amount": self.amount,
            "commission": self.commission,
            "tax": self.tax,
            "net_amount": self.net_amount,
            "datetime": self.datetime.isoformat(),
            "order_id": self.order_id,
            "remarks": self.remarks,
        }


class TransactionLog:
    """交易日志"""

    def __init__(self):
        self._transactions: List[Transaction] = []

    def add(self, transaction: Transaction) -> None:
        """添加交易记录"""
        self._transactions.append(transaction)

    def add_buy(
        self,
        code: str,
        quantity: int,
        price: float,
        commission: float = 0.0,
        order_id: Optional[str] = None,
    ) -> Transaction:
        """记录买入"""
        txn = Transaction(
            code=code,
            type=TransactionType.BUY,
            quantity=quantity,
            price=price,
            amount=price * quantity,
            commission=commission,
            order_id=order_id,
        )
        self.add(txn)
        return txn

    def add_sell(
        self,
        code: str,
        quantity: int,
        price: float,
        commission: float = 0.0,
        tax: float = 0.0,
        order_id: Optional[str] = None,
    ) -> Transaction:
        """记录卖出"""
        txn = Transaction(
            code=code,
            type=TransactionType.SELL,
            quantity=quantity,
            price=price,
            amount=price * quantity,
            commission=commission,
            tax=tax,
            order_id=order_id,
        )
        self.add(txn)
        return txn

    def get_transactions(
        self,
        code: Optional[str] = None,
        type_: Optional[TransactionType] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[Transaction]:
        """获取交易记录"""
        result = self._transactions

        if code:
            result = [t for t in result if t.code == code]

        if type_:
            result = [t for t in result if t.type == type_]

        if start_date:
            result = [t for t in result if t.datetime.date() >= start_date]

        if end_date:
            result = [t for t in result if t.datetime.date() <= end_date]

        return result

    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        if not self._transactions:
            return pd.DataFrame()

        data = [t.to_dict() for t in self._transactions]
        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    def get_summary(self) -> Dict:
        """获取汇总"""
        if not self._transactions:
            return {
                "total_count": 0,
                "buy_count": 0,
                "sell_count": 0,
                "total_commission": 0,
                "total_tax": 0,
            }

        buy_txns = [t for t in self._transactions if t.type == TransactionType.BUY]
        sell_txns = [t for t in self._transactions if t.type == TransactionType.SELL]

        return {
            "total_count": len(self._transactions),
            "buy_count": len(buy_txns),
            "sell_count": len(sell_txns),
            "total_buy_amount": sum(t.amount for t in buy_txns),
            "total_sell_amount": sum(t.amount for t in sell_txns),
            "total_commission": sum(t.commission for t in self._transactions),
            "total_tax": sum(t.tax for t in self._transactions),
        }

    def get_trades_by_stock(self, code: str) -> Dict:
        """获取某只股票的交易统计"""
        txns = self.get_transactions(code=code)

        buy_txns = [t for t in txns if t.type == TransactionType.BUY]
        sell_txns = [t for t in txns if t.type == TransactionType.SELL]

        total_buy_qty = sum(t.quantity for t in buy_txns)
        total_sell_qty = sum(t.quantity for t in sell_txns)
        total_buy_amount = sum(t.amount for t in buy_txns)
        total_sell_amount = sum(t.amount for t in sell_txns)

        avg_buy_price = total_buy_amount / total_buy_qty if total_buy_qty > 0 else 0
        avg_sell_price = total_sell_amount / total_sell_qty if total_sell_qty > 0 else 0

        return {
            "code": code,
            "buy_count": len(buy_txns),
            "sell_count": len(sell_txns),
            "total_buy_qty": total_buy_qty,
            "total_sell_qty": total_sell_qty,
            "avg_buy_price": avg_buy_price,
            "avg_sell_price": avg_sell_price,
            "realized_profit": total_sell_amount - total_buy_amount if total_sell_qty >= total_buy_qty else None,
        }

    def clear(self) -> None:
        """清空记录"""
        self._transactions.clear()
