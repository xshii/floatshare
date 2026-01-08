"""组合持久化存储

提供 Portfolio 和 AssetSnapshot 的数据库持久化能力
"""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.account.portfolio import Portfolio, Position
from src.account.asset import AssetManager, AssetSnapshot


class PortfolioStorage:
    """组合存储管理"""

    def __init__(self, db_path: Optional[str] = None):
        """
        初始化存储

        Args:
            db_path: 数据库路径，默认使用SQLite
        """
        if db_path is None:
            db_path = str(Path(__file__).parent.parent.parent / "data" / "floatshare.db")

        self.db_path = db_path
        self._engine = None

    @property
    def engine(self):
        """延迟初始化数据库引擎"""
        if self._engine is None:
            from sqlalchemy import create_engine

            self._engine = create_engine(f"sqlite:///{self.db_path}")
            self.init_tables()
        return self._engine

    def init_tables(self):
        """初始化数据表"""
        from sqlalchemy import text

        create_statements = [
            # 组合表
            """
            CREATE TABLE IF NOT EXISTS portfolios (
                name TEXT PRIMARY KEY,
                initial_capital REAL,
                cash REAL,
                created_at TEXT,
                updated_at TEXT
            )
            """,
            # 持仓表
            """
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_name TEXT,
                code TEXT,
                quantity INTEGER,
                avg_cost REAL,
                current_price REAL,
                frozen INTEGER DEFAULT 0,
                updated_at TEXT,
                UNIQUE(portfolio_name, code),
                FOREIGN KEY (portfolio_name) REFERENCES portfolios(name)
            )
            """,
            # 资产快照表
            """
            CREATE TABLE IF NOT EXISTS asset_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_name TEXT,
                snapshot_date TEXT,
                cash REAL,
                position_value REAL,
                total_value REAL,
                positions_json TEXT,
                created_at TEXT,
                FOREIGN KEY (portfolio_name) REFERENCES portfolios(name)
            )
            """,
            # 交易记录表
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_name TEXT,
                trade_date TEXT,
                code TEXT,
                direction TEXT,
                quantity INTEGER,
                price REAL,
                amount REAL,
                commission REAL,
                created_at TEXT,
                FOREIGN KEY (portfolio_name) REFERENCES portfolios(name)
            )
            """,
            # 索引
            """
            CREATE INDEX IF NOT EXISTS idx_positions_portfolio
            ON positions(portfolio_name)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_snapshots_portfolio_date
            ON asset_snapshots(portfolio_name, snapshot_date)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_transactions_portfolio_date
            ON transactions(portfolio_name, trade_date)
            """,
        ]

        with self.engine.connect() as conn:
            for stmt in create_statements:
                conn.execute(text(stmt))
            conn.commit()

    # ============================================================
    # Portfolio 操作
    # ============================================================

    def save_portfolio(self, portfolio: Portfolio) -> bool:
        """
        保存组合

        Args:
            portfolio: 组合对象

        Returns:
            是否成功
        """
        from sqlalchemy import text

        now = datetime.now().isoformat()

        with self.engine.connect() as conn:
            # 保存或更新组合基本信息
            conn.execute(
                text("""
                    INSERT OR REPLACE INTO portfolios
                    (name, initial_capital, cash, created_at, updated_at)
                    VALUES (:name, :initial_capital, :cash, :created_at, :updated_at)
                """),
                {
                    "name": portfolio.name,
                    "initial_capital": portfolio.initial_capital,
                    "cash": portfolio.cash,
                    "created_at": portfolio.created_at.isoformat(),
                    "updated_at": now,
                },
            )

            # 删除旧持仓
            conn.execute(
                text("DELETE FROM positions WHERE portfolio_name = :name"),
                {"name": portfolio.name},
            )

            # 保存新持仓
            for code, pos in portfolio.positions.items():
                conn.execute(
                    text("""
                        INSERT INTO positions
                        (portfolio_name, code, quantity, avg_cost, current_price, frozen, updated_at)
                        VALUES (:portfolio_name, :code, :quantity, :avg_cost, :current_price, :frozen, :updated_at)
                    """),
                    {
                        "portfolio_name": portfolio.name,
                        "code": code,
                        "quantity": pos.quantity,
                        "avg_cost": pos.avg_cost,
                        "current_price": pos.current_price,
                        "frozen": pos.frozen,
                        "updated_at": now,
                    },
                )

            conn.commit()

        return True

    def load_portfolio(self, name: str) -> Optional[Portfolio]:
        """
        加载组合

        Args:
            name: 组合名称

        Returns:
            组合对象，不存在返回None
        """
        # 加载组合基本信息
        try:
            df = pd.read_sql(
                f"SELECT * FROM portfolios WHERE name = '{name}'",
                self.engine,
            )
        except Exception:
            return None

        if df.empty:
            return None

        row = df.iloc[0]

        portfolio = Portfolio(
            name=row["name"],
            initial_capital=row["initial_capital"],
            cash=row["cash"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

        # 加载持仓
        try:
            positions_df = pd.read_sql(
                f"SELECT * FROM positions WHERE portfolio_name = '{name}'",
                self.engine,
            )

            for _, pos_row in positions_df.iterrows():
                portfolio.positions[pos_row["code"]] = Position(
                    code=pos_row["code"],
                    quantity=int(pos_row["quantity"]),
                    avg_cost=pos_row["avg_cost"],
                    current_price=pos_row["current_price"],
                    frozen=int(pos_row["frozen"]),
                )
        except Exception:
            pass

        return portfolio

    def delete_portfolio(self, name: str) -> bool:
        """删除组合"""
        from sqlalchemy import text

        with self.engine.connect() as conn:
            conn.execute(
                text("DELETE FROM positions WHERE portfolio_name = :name"),
                {"name": name},
            )
            conn.execute(
                text("DELETE FROM asset_snapshots WHERE portfolio_name = :name"),
                {"name": name},
            )
            conn.execute(
                text("DELETE FROM transactions WHERE portfolio_name = :name"),
                {"name": name},
            )
            conn.execute(
                text("DELETE FROM portfolios WHERE name = :name"),
                {"name": name},
            )
            conn.commit()

        return True

    def list_portfolios(self) -> List[Dict]:
        """列出所有组合"""
        try:
            df = pd.read_sql("SELECT * FROM portfolios ORDER BY updated_at DESC", self.engine)
            return df.to_dict("records")
        except Exception:
            return []

    # ============================================================
    # AssetSnapshot 操作
    # ============================================================

    def save_snapshot(self, portfolio_name: str, snapshot: AssetSnapshot) -> bool:
        """保存资产快照"""
        from sqlalchemy import text

        now = datetime.now().isoformat()

        with self.engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO asset_snapshots
                    (portfolio_name, snapshot_date, cash, position_value, total_value, positions_json, created_at)
                    VALUES (:portfolio_name, :snapshot_date, :cash, :position_value, :total_value, :positions_json, :created_at)
                """),
                {
                    "portfolio_name": portfolio_name,
                    "snapshot_date": snapshot.date.isoformat(),
                    "cash": snapshot.cash,
                    "position_value": snapshot.position_value,
                    "total_value": snapshot.total_value,
                    "positions_json": json.dumps(snapshot.positions),
                    "created_at": now,
                },
            )
            conn.commit()

        return True

    def load_snapshots(
        self,
        portfolio_name: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[AssetSnapshot]:
        """加载资产快照"""
        query = f"SELECT * FROM asset_snapshots WHERE portfolio_name = '{portfolio_name}'"

        if start_date:
            query += f" AND snapshot_date >= '{start_date.isoformat()}'"
        if end_date:
            query += f" AND snapshot_date <= '{end_date.isoformat()}'"

        query += " ORDER BY snapshot_date"

        try:
            df = pd.read_sql(query, self.engine)
        except Exception:
            return []

        snapshots = []
        for _, row in df.iterrows():
            snapshots.append(
                AssetSnapshot(
                    date=date.fromisoformat(row["snapshot_date"]),
                    cash=row["cash"],
                    position_value=row["position_value"],
                    total_value=row["total_value"],
                    positions=json.loads(row["positions_json"]) if row["positions_json"] else {},
                )
            )

        return snapshots

    def get_snapshot_history(self, portfolio_name: str) -> pd.DataFrame:
        """获取快照历史DataFrame"""
        query = f"""
            SELECT snapshot_date as date, cash, position_value, total_value
            FROM asset_snapshots
            WHERE portfolio_name = '{portfolio_name}'
            ORDER BY snapshot_date
        """

        try:
            df = pd.read_sql(query, self.engine)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception:
            return pd.DataFrame()

    # ============================================================
    # 交易记录操作
    # ============================================================

    def save_transaction(
        self,
        portfolio_name: str,
        trade_date: date,
        code: str,
        direction: str,
        quantity: int,
        price: float,
        commission: float = 0.0,
    ) -> bool:
        """保存交易记录"""
        from sqlalchemy import text

        now = datetime.now().isoformat()
        amount = price * quantity

        with self.engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO transactions
                    (portfolio_name, trade_date, code, direction, quantity, price, amount, commission, created_at)
                    VALUES (:portfolio_name, :trade_date, :code, :direction, :quantity, :price, :amount, :commission, :created_at)
                """),
                {
                    "portfolio_name": portfolio_name,
                    "trade_date": trade_date.isoformat(),
                    "code": code,
                    "direction": direction,
                    "quantity": quantity,
                    "price": price,
                    "amount": amount,
                    "commission": commission,
                    "created_at": now,
                },
            )
            conn.commit()

        return True

    def load_transactions(
        self,
        portfolio_name: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """加载交易记录"""
        query = f"SELECT * FROM transactions WHERE portfolio_name = '{portfolio_name}'"

        if start_date:
            query += f" AND trade_date >= '{start_date.isoformat()}'"
        if end_date:
            query += f" AND trade_date <= '{end_date.isoformat()}'"

        query += " ORDER BY trade_date, id"

        try:
            df = pd.read_sql(query, self.engine)
            if not df.empty and "trade_date" in df.columns:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
            return df
        except Exception:
            return pd.DataFrame()

    def get_trade_summary(self, portfolio_name: str) -> Dict:
        """获取交易统计"""
        df = self.load_transactions(portfolio_name)

        if df.empty:
            return {
                "total_trades": 0,
                "buy_count": 0,
                "sell_count": 0,
                "total_amount": 0,
                "total_commission": 0,
            }

        return {
            "total_trades": len(df),
            "buy_count": len(df[df["direction"] == "buy"]),
            "sell_count": len(df[df["direction"] == "sell"]),
            "total_amount": df["amount"].sum(),
            "total_commission": df["commission"].sum(),
        }


# ============================================================
# 全局存储实例
# ============================================================

_portfolio_storage: Optional[PortfolioStorage] = None


def get_portfolio_storage() -> PortfolioStorage:
    """获取全局存储实例"""
    global _portfolio_storage
    if _portfolio_storage is None:
        _portfolio_storage = PortfolioStorage()
    return _portfolio_storage


def set_portfolio_storage(storage: PortfolioStorage) -> None:
    """设置全局存储实例"""
    global _portfolio_storage
    _portfolio_storage = storage
