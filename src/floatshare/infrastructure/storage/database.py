"""SQLite 存储 — 路径通过构造参数注入，避免到处探父级。"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine


class DatabaseStorage:
    """日线/股票列表的本地持久化。"""

    def __init__(self, db_path: str | Path = "data/floatshare.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine: Engine | None = None

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            from sqlalchemy import create_engine

            self._engine = create_engine(f"sqlite:///{self.db_path}")
        return self._engine

    def save_daily(self, df: pd.DataFrame, table: str = "stock_daily") -> int:
        if df.empty:
            return 0
        df.to_sql(table, self.engine, if_exists="append", index=False)
        return len(df)

    def load_daily(
        self,
        code: str,
        start: date | None = None,
        end: date | None = None,
        table: str = "stock_daily",
    ) -> pd.DataFrame:
        from sqlalchemy import text

        clauses = ["code = :code"]
        params: dict[str, object] = {"code": code}
        if start is not None:
            clauses.append("trade_date >= :start")
            params["start"] = start.isoformat()
        if end is not None:
            clauses.append("trade_date <= :end")
            params["end"] = end.isoformat()
        query = text(f"SELECT * FROM {table} WHERE {' AND '.join(clauses)} ORDER BY trade_date")
        try:
            df = pd.read_sql(query, self.engine, params=params)
        except Exception:
            return pd.DataFrame()
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df

    def save_stock_list(self, df: pd.DataFrame, table: str = "stock_info") -> int:
        if df.empty:
            return 0
        df.to_sql(table, self.engine, if_exists="replace", index=False)
        return len(df)

    def load_stock_list(self, table: str = "stock_info") -> pd.DataFrame:
        try:
            return pd.read_sql(f"SELECT * FROM {table}", self.engine)
        except Exception:
            return pd.DataFrame()

    def get_latest_date(self, code: str, table: str = "stock_daily") -> date | None:
        from sqlalchemy import text

        query = text(f"SELECT MAX(trade_date) AS max_date FROM {table} WHERE code = :code")
        try:
            result = pd.read_sql(query, self.engine, params={"code": code})
            value = result["max_date"].iloc[0]
        except Exception:
            return None
        if value is None:
            return None
        return pd.to_datetime(value).date()

    def init_tables(self) -> None:
        from sqlalchemy import text

        statements = [
            """
            CREATE TABLE IF NOT EXISTS stock_info (
                code TEXT PRIMARY KEY,
                ticker TEXT,
                name TEXT,
                market TEXT,
                industry TEXT,
                list_date TEXT,
                delist_date TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS stock_daily (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT,
                trade_date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                amount REAL,
                pre_close REAL,
                change REAL,
                pct_change REAL,
                UNIQUE(code, trade_date)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_daily_code_date ON stock_daily(code, trade_date)",
        ]
        with self.engine.connect() as conn:
            for stmt in statements:
                conn.execute(text(stmt))
            conn.commit()
