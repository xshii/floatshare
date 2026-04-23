"""SQLite 存储 — 路径通过构造参数注入。

表设计遵循 DDIA "真相来源 + 派生数据" 原则：
- Layer 0 (Source of Truth): raw_daily, adj_factor, income/balancesheet/cashflow, ...
- sync_watermark: 增量同步水位线

Schema 单一真相来源在 `domain/records.py` (dataclass)，本文件仅负责 CRUD
和动态查询; 列名/DDL/UPSERT SQL 全部从 dataclass 自动生成
(`infrastructure/storage/schema_sql.py`)。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from floatshare.domain.records import (
    ALL_RECORDS,
    StockLifecycle,
    SyncWatermark,
    TradeCalendar,
)
from floatshare.infrastructure.storage.schema_sql import (
    ddl,
    row_to_params,
    upsert_sql,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.engine import Engine


class DatabaseStorage:
    """金融数据本地持久化 — SQLite。

    通用 API:
        db.save(RecordCls, df)          # UPSERT，自动从 dataclass 推 SQL
        db.load(table, code, start, end)  # 通用查询
        db.latest_date(table, code)     # MAX(trade_date)
        db.date_range(table, code)      # (MIN, MAX)
        db.has_rows(table, **filters)   # 是否有匹配行
    """

    def __init__(self, db_path: str | Path = "data/floatshare.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine: Engine | None = None

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            from sqlalchemy import create_engine, event

            self._engine = create_engine(
                f"sqlite:///{self.db_path}",
                connect_args={"timeout": 30.0},  # 锁等待 30s 而非默认 5s
            )

            # 开 WAL 模式: 读写并发 (sync 写 + web 读/写 同时工作)
            @event.listens_for(self._engine, "connect")
            def _set_sqlite_pragma(dbapi_conn, _conn_record) -> None:
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA busy_timeout=30000")
                cursor.close()

        return self._engine

    # ========================================================================
    # 通用 CRUD (dataclass-driven)
    # ========================================================================

    def save(self, record_cls: type, df: pd.DataFrame) -> int:
        """通用 UPSERT — 从 dataclass schema 自动生成 SQL。"""
        sql = upsert_sql(record_cls)
        return self._upsert_rows(
            sql,
            df,
            row_mapper=lambda row: row_to_params(record_cls, row),
        )

    def _upsert_rows(
        self,
        sql: str,
        df: pd.DataFrame,
        row_mapper: Callable[[dict], dict] | None = None,
    ) -> int:
        if df.empty:
            return 0
        from sqlalchemy import text

        stmt = text(sql)
        rows = df.to_dict("records")
        with self.engine.connect() as conn:
            for row in rows:
                params = row_mapper(row) if row_mapper is not None else row
                conn.execute(stmt, params)
            conn.commit()
        return len(rows)

    def load(
        self,
        table: str,
        code: str,
        start: date | None = None,
        end: date | None = None,
        columns: str = "*",
        date_col: str = "trade_date",
    ) -> pd.DataFrame:
        """通用查询：按 code + 日期范围过滤，自动解析日期列。"""
        from sqlalchemy import text

        clauses = ["code = :code"]
        params: dict[str, object] = {"code": code}
        if start is not None:
            clauses.append(f"{date_col} >= :start")
            params["start"] = start.isoformat()
        if end is not None:
            clauses.append(f"{date_col} <= :end")
            params["end"] = end.isoformat()
        query = text(
            f"SELECT {columns} FROM {table} WHERE {' AND '.join(clauses)} ORDER BY {date_col}"
        )
        try:
            df = pd.read_sql(query, self.engine, params=params)
        except Exception:
            return pd.DataFrame()
        if date_col in df.columns:
            # SQLite 存的日期是 "YYYY-MM-DDTHH:MM:SS" (ISO 8601 带 T),
            # pandas 默认 strptime 模板不识别 T 分隔符, 显式声明 ISO8601
            df[date_col] = pd.to_datetime(df[date_col], format="ISO8601")
        return df

    def latest_date(
        self,
        table: str,
        code: str | None = None,
        date_col: str = "trade_date",
    ) -> date | None:
        """返回某表（按 code 过滤）的最新日期。"""
        filters = {"code": code} if code else {}
        return self._get_max_date(table, date_col=date_col, **filters)

    def date_range(
        self,
        table: str,
        code: str | None = None,
        date_col: str = "trade_date",
    ) -> tuple[date | None, date | None]:
        """返回某表（按 code 过滤）的 (MIN, MAX) 日期范围。"""
        from sqlalchemy import text

        clauses = ["code = :code"] if code else []
        params: dict[str, object] = {"code": code} if code else {}
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = text(f"SELECT MIN({date_col}) AS lo, MAX({date_col}) AS hi FROM {table}{where}")
        try:
            result = pd.read_sql(query, self.engine, params=params)
            lo, hi = result["lo"].iloc[0], result["hi"].iloc[0]
        except Exception:
            return (None, None)
        return (
            pd.to_datetime(lo).date() if lo is not None else None,
            pd.to_datetime(hi).date() if hi is not None else None,
        )

    def has_rows(self, table: str, **filters: str) -> bool:
        """判断某表（按 filters 过滤）是否已有数据。"""
        from sqlalchemy import text

        clauses = [f"{k} = :{k}" for k in filters]
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = text(f"SELECT 1 FROM {table}{where} LIMIT 1")
        try:
            with self.engine.connect() as conn:
                return conn.execute(query, filters).first() is not None
        except Exception:
            return False

    def _get_max_date(
        self, table: str, date_col: str = "trade_date", **filters: str
    ) -> date | None:
        from sqlalchemy import text

        clauses = [f"{k} = :{k}" for k in filters]
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = text(f"SELECT MAX({date_col}) AS d FROM {table}{where}")
        try:
            result = pd.read_sql(query, self.engine, params=filters)
            value = result["d"].iloc[0]
        except Exception:
            return None
        if value is None:
            return None
        return pd.to_datetime(value).date()

    # ========================================================================
    # 特殊持久化 (非 dataclass-upsert 形态)
    # ========================================================================

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

    # ========================================================================
    # 特殊查询 (非通用 CRUD)
    # ========================================================================

    def save_trade_calendar(self, dates: list[date]) -> int:
        if not dates:
            return 0
        df = pd.DataFrame({"trade_date": [d.isoformat() for d in dates]})
        return self.save(TradeCalendar, df)

    def load_trade_calendar(self, start: date | None = None, end: date | None = None) -> list[date]:
        from sqlalchemy import text

        clauses: list[str] = []
        params: dict[str, object] = {}
        if start is not None:
            clauses.append("trade_date >= :start")
            params["start"] = start.isoformat()
        if end is not None:
            clauses.append("trade_date <= :end")
            params["end"] = end.isoformat()
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = text(f"SELECT trade_date FROM trade_calendar{where} ORDER BY trade_date")
        try:
            df = pd.read_sql(query, self.engine, params=params)
        except Exception:
            return []
        return sorted(pd.to_datetime(df["trade_date"]).dt.date.tolist())

    def get_watermark(self, code: str) -> date | None:
        from sqlalchemy import text

        query = text("SELECT last_date FROM sync_watermark WHERE code = :code")
        try:
            result = pd.read_sql(query, self.engine, params={"code": code})
            if result.empty:
                return None
            return pd.to_datetime(result["last_date"].iloc[0]).date()
        except Exception:
            return None

    def update_watermark(self, code: str, last_date: date, source: str) -> None:
        df = pd.DataFrame(
            [
                {
                    "code": code,
                    "last_date": last_date.isoformat(),
                    "source": source,
                    "updated_at": pd.Timestamp.now().isoformat(),
                }
            ]
        )
        self.save(SyncWatermark, df)

    def get_lifecycle(self, code: str) -> StockLifecycle | None:
        """返回单个 code 的生命周期 dataclass (schema 即 StockLifecycle 字段).

        用 dataclass 替代裸 dict — 消费方属性访问 (lc.list_date), 不再 lc.get('list_date').
        """
        from sqlalchemy import text

        from floatshare.infrastructure.storage.schema_sql import from_row

        query = text("SELECT * FROM stock_lifecycle WHERE code = :code")
        try:
            with self.engine.connect() as conn:
                row = conn.execute(query, {"code": code}).mappings().first()
        except Exception:
            return None
        return from_row(StockLifecycle, row) if row else None

    def load_lifecycle(self, list_status: str | None = None) -> pd.DataFrame:
        from sqlalchemy import text

        if list_status:
            query = text("SELECT * FROM stock_lifecycle WHERE list_status = :s ORDER BY code")
            params: dict[str, object] = {"s": list_status}
        else:
            query = text("SELECT * FROM stock_lifecycle ORDER BY code")
            params = {}
        try:
            return pd.read_sql(query, self.engine, params=params)
        except Exception:
            return pd.DataFrame()

    # save_healthcheck 已移除 — 走 observability.metrics.record_counter
    # (domain='healthcheck/<source>/<method>'). 详见 cli/healthcheck.py.

    # ========================================================================
    # DDL — 全部从 dataclass 自动生成 (单一真相来源)
    # ========================================================================

    def init_tables(self) -> None:
        from sqlalchemy import text

        # 所有表 DDL 由 dataclass 自动生成
        auto_ddl = [ddl(rec) for rec in ALL_RECORDS]

        indexes = [
            # 注意: PK autoindex 已经是 (code, trade_date), 下面 idx_*_code 是冗余的,
            # 历史遗留, 后续可清; 但不影响正确性。
            "CREATE INDEX IF NOT EXISTS idx_raw_daily_code ON raw_daily(code, trade_date)",
            "CREATE INDEX IF NOT EXISTS idx_adj_code ON adj_factor(code, trade_date)",
            "CREATE INDEX IF NOT EXISTS idx_chip_perf_code ON chip_perf(code, trade_date)",
            "CREATE INDEX IF NOT EXISTS idx_chip_dist_code ON chip_dist(code, trade_date)",
            "CREATE INDEX IF NOT EXISTS idx_daily_basic_code ON daily_basic(code, trade_date)",
            "CREATE INDEX IF NOT EXISTS idx_index_weight_idx ON index_weight(index_code, trade_date)",
            "CREATE INDEX IF NOT EXISTS idx_moneyflow_code ON moneyflow(code, trade_date)",
            "CREATE INDEX IF NOT EXISTS idx_income_code ON income(code, end_date)",
            "CREATE INDEX IF NOT EXISTS idx_balancesheet_code ON balancesheet(code, end_date)",
            "CREATE INDEX IF NOT EXISTS idx_cashflow_code ON cashflow(code, end_date)",
            "CREATE INDEX IF NOT EXISTS idx_counter_scope_name_ts ON counter_event(scope, name, ts)",
            "CREATE INDEX IF NOT EXISTS idx_counter_run_id ON counter_event(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_kpi_domain_name_ts ON kpi_snapshot(domain, kpi_name, ts)",
            "CREATE INDEX IF NOT EXISTS idx_kpi_run_id ON kpi_snapshot(run_id)",
            # (trade_date, code) 覆盖索引 — daily_status_cells 按日期范围 GROUP BY,
            # 索引-only 扫描, 6 表并行 <150ms (替代全表扫的 30s+)
            "CREATE INDEX IF NOT EXISTS idx_raw_daily_date_code ON raw_daily(trade_date, code)",
            "CREATE INDEX IF NOT EXISTS idx_adj_factor_date_code ON adj_factor(trade_date, code)",
            "CREATE INDEX IF NOT EXISTS idx_daily_basic_date_code ON daily_basic(trade_date, code)",
            "CREATE INDEX IF NOT EXISTS idx_moneyflow_date_code ON moneyflow(trade_date, code)",
            "CREATE INDEX IF NOT EXISTS idx_margin_detail_date_code ON margin_detail(trade_date, code)",
            "CREATE INDEX IF NOT EXISTS idx_chip_perf_date_code ON chip_perf(trade_date, code)",
        ]

        with self.engine.connect() as conn:
            for stmt in [*auto_ddl, *indexes]:
                conn.execute(text(stmt))
            conn.commit()
