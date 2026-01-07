"""数据库存储"""

from typing import Optional, List
from datetime import date
import pandas as pd
from pathlib import Path


class DatabaseStorage:
    """数据库存储管理"""

    def __init__(self, db_path: Optional[str] = None):
        """
        初始化数据库存储

        Args:
            db_path: 数据库路径，默认使用SQLite
        """
        if db_path is None:
            db_path = str(Path(__file__).parent.parent.parent.parent / "data" / "floatshare.db")

        self.db_path = db_path
        self._engine = None

    @property
    def engine(self):
        """延迟初始化数据库引擎"""
        if self._engine is None:
            from sqlalchemy import create_engine

            self._engine = create_engine(f"sqlite:///{self.db_path}")
        return self._engine

    def save_daily(self, df: pd.DataFrame, table: str = "stock_daily") -> int:
        """
        保存日线数据

        Args:
            df: 数据DataFrame
            table: 表名

        Returns:
            保存的行数
        """
        if df.empty:
            return 0

        df.to_sql(table, self.engine, if_exists="append", index=False)
        return len(df)

    def load_daily(
        self,
        code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        table: str = "stock_daily",
    ) -> pd.DataFrame:
        """
        加载日线数据

        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            table: 表名
        """
        query = f"SELECT * FROM {table} WHERE code = '{code}'"

        if start_date:
            query += f" AND trade_date >= '{start_date}'"
        if end_date:
            query += f" AND trade_date <= '{end_date}'"

        query += " ORDER BY trade_date"

        try:
            df = pd.read_sql(query, self.engine)
            if "trade_date" in df.columns:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
            return df
        except Exception:
            return pd.DataFrame()

    def save_stock_list(self, df: pd.DataFrame, table: str = "stock_info") -> int:
        """保存股票列表"""
        if df.empty:
            return 0

        df.to_sql(table, self.engine, if_exists="replace", index=False)
        return len(df)

    def load_stock_list(self, table: str = "stock_info") -> pd.DataFrame:
        """加载股票列表"""
        try:
            return pd.read_sql(f"SELECT * FROM {table}", self.engine)
        except Exception:
            return pd.DataFrame()

    def get_latest_date(self, code: str, table: str = "stock_daily") -> Optional[date]:
        """获取某只股票最新数据日期"""
        query = f"SELECT MAX(trade_date) as max_date FROM {table} WHERE code = '{code}'"

        try:
            result = pd.read_sql(query, self.engine)
            if result["max_date"].iloc[0]:
                return pd.to_datetime(result["max_date"].iloc[0]).date()
        except Exception:
            pass

        return None

    def execute(self, query: str) -> pd.DataFrame:
        """执行自定义查询"""
        return pd.read_sql(query, self.engine)

    def init_tables(self):
        """初始化数据表"""
        from sqlalchemy import text

        create_statements = [
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
            """
            CREATE INDEX IF NOT EXISTS idx_daily_code_date
            ON stock_daily(code, trade_date)
            """,
        ]

        with self.engine.connect() as conn:
            for stmt in create_statements:
                conn.execute(text(stmt))
            conn.commit()
