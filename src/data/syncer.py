"""数据同步管理器

用于批量爬取历史数据，支持：
- 股票池管理（按优先级：沪深300、中证500、全市场）
- 增量同步（只获取缺失数据）
- 进度追踪（支持多日中断恢复）
- 速率控制（防止被封禁）
"""

import json
import time
import logging
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field, asdict

import pandas as pd

from .loader import DataLoader
from .storage.database import DatabaseStorage

logger = logging.getLogger(__name__)


class SyncPriority(Enum):
    """同步优先级"""

    HS300 = "hs300"  # 沪深300成分股
    ZZ500 = "zz500"  # 中证500成分股
    ZZ1000 = "zz1000"  # 中证1000成分股
    ALL = "all"  # 全市场


class SyncStatus(Enum):
    """同步状态"""

    PENDING = "pending"  # 待同步
    IN_PROGRESS = "in_progress"  # 同步中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败
    SKIPPED = "skipped"  # 跳过


@dataclass
class SyncTask:
    """单个同步任务"""

    code: str  # 股票代码
    name: str  # 股票名称
    priority: SyncPriority  # 优先级
    status: SyncStatus = SyncStatus.PENDING
    last_sync_date: Optional[str] = None  # 上次同步到的日期
    target_start_date: Optional[str] = None  # 目标起始日期
    target_end_date: Optional[str] = None  # 目标结束日期
    error_msg: Optional[str] = None  # 错误信息
    retry_count: int = 0  # 重试次数
    synced_rows: int = 0  # 已同步行数


@dataclass
class SyncState:
    """同步状态（用于持久化）"""

    session_id: str  # 会话ID
    created_at: str  # 创建时间
    updated_at: str  # 更新时间
    source: str  # 数据源
    data_type: str  # 数据类型（daily/minute/dividend）
    total_tasks: int = 0  # 总任务数
    completed_tasks: int = 0  # 已完成任务数
    failed_tasks: int = 0  # 失败任务数
    current_priority: str = "hs300"  # 当前处理的优先级
    tasks: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # code -> task


class RateLimiter:
    """速率限制器"""

    def __init__(
        self,
        requests_per_minute: int = 60,
        batch_size: int = 50,
        batch_pause: float = 10.0,
    ):
        """
        Args:
            requests_per_minute: 每分钟最大请求数
            batch_size: 批次大小（每处理多少个后暂停）
            batch_pause: 批次间暂停时间（秒）
        """
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.batch_size = batch_size
        self.batch_pause = batch_pause
        self.request_count = 0
        self.last_request_time = 0.0

    def wait(self) -> None:
        """等待以满足速率限制"""
        now = time.time()
        elapsed = now - self.last_request_time

        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

        self.last_request_time = time.time()
        self.request_count += 1

        # 批次暂停
        if self.request_count % self.batch_size == 0:
            logger.info(f"已处理 {self.request_count} 个请求，暂停 {self.batch_pause} 秒...")
            time.sleep(self.batch_pause)


class DataSyncer:
    """数据同步管理器"""

    def __init__(
        self,
        source: str = "akshare",
        db_path: Optional[str] = None,
        state_path: Optional[str] = None,
    ):
        """
        Args:
            source: 数据源（akshare/tushare）
            db_path: 数据库路径
            state_path: 状态文件路径
        """
        self.source = source
        self.loader = DataLoader(source=source)
        self.storage = DatabaseStorage(db_path=db_path)

        # 状态文件路径
        if state_path is None:
            state_path = str(
                Path(__file__).parent.parent.parent / "data" / "sync_state.json"
            )
        self.state_path = Path(state_path)

        # 速率限制器
        self.rate_limiter = RateLimiter(
            requests_per_minute=30,  # 保守设置，避免被封
            batch_size=50,
            batch_pause=30.0,
        )

        # 回调函数
        self._on_progress: Optional[Callable[[int, int, str], None]] = None
        self._on_error: Optional[Callable[[str, str], None]] = None

    def set_rate_limit(
        self,
        requests_per_minute: int = 30,
        batch_size: int = 50,
        batch_pause: float = 30.0,
    ) -> "DataSyncer":
        """设置速率限制"""
        self.rate_limiter = RateLimiter(
            requests_per_minute=requests_per_minute,
            batch_size=batch_size,
            batch_pause=batch_pause,
        )
        return self

    def on_progress(
        self, callback: Callable[[int, int, str], None]
    ) -> "DataSyncer":
        """设置进度回调 (completed, total, current_code)"""
        self._on_progress = callback
        return self

    def on_error(self, callback: Callable[[str, str], None]) -> "DataSyncer":
        """设置错误回调 (code, error_msg)"""
        self._on_error = callback
        return self

    # ============================================================
    # 股票池管理
    # ============================================================

    def get_universe(self, priority: SyncPriority) -> pd.DataFrame:
        """
        获取股票池

        Args:
            priority: 优先级

        Returns:
            DataFrame with columns: code, name, priority
        """
        if priority == SyncPriority.HS300:
            return self._get_index_constituents("000300.SH", priority)
        elif priority == SyncPriority.ZZ500:
            return self._get_index_constituents("000905.SH", priority)
        elif priority == SyncPriority.ZZ1000:
            return self._get_index_constituents("000852.SH", priority)
        else:
            return self._get_all_stocks(priority)

    def _get_index_constituents(
        self, index_code: str, priority: SyncPriority
    ) -> pd.DataFrame:
        """获取指数成分股"""
        try:
            # 尝试通过 akshare 获取成分股
            import akshare as ak

            index_ticker = index_code.split(".")[0]

            if index_ticker == "000300":
                df = ak.index_stock_cons_csindex(symbol="000300")
            elif index_ticker == "000905":
                df = ak.index_stock_cons_csindex(symbol="000905")
            elif index_ticker == "000852":
                df = ak.index_stock_cons_csindex(symbol="000852")
            else:
                return pd.DataFrame()

            if df.empty:
                return df

            # 统一列名
            if "成分券代码" in df.columns:
                df = df.rename(columns={"成分券代码": "ticker", "成分券名称": "name"})
            elif "证券代码" in df.columns:
                df = df.rename(columns={"证券代码": "ticker", "证券名称": "name"})

            df["code"] = df["ticker"].apply(self._add_market_suffix)
            df["priority"] = priority.value

            return df[["code", "name", "priority"]].drop_duplicates()

        except Exception as e:
            logger.warning(f"获取指数成分股失败: {e}")
            return pd.DataFrame()

    def _get_all_stocks(self, priority: SyncPriority) -> pd.DataFrame:
        """获取全市场股票"""
        df = self.loader.get_stock_list()

        if df.empty:
            return df

        df["priority"] = priority.value

        # 确保有必要的列
        if "code" not in df.columns and "ticker" in df.columns:
            df["code"] = df["ticker"].apply(self._add_market_suffix)

        return df[["code", "name", "priority"]].drop_duplicates()

    def _add_market_suffix(self, ticker: str) -> str:
        """添加市场后缀"""
        ticker = str(ticker).zfill(6)
        if ticker.startswith("6"):
            return f"{ticker}.SH"
        elif ticker.startswith(("0", "3")):
            return f"{ticker}.SZ"
        elif ticker.startswith(("4", "8")):
            return f"{ticker}.BJ"
        return ticker

    # ============================================================
    # 增量同步
    # ============================================================

    def sync_daily(
        self,
        priorities: Optional[List[SyncPriority]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        resume: bool = True,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        同步日线数据

        Args:
            priorities: 优先级列表，按顺序处理
            start_date: 起始日期（默认从1990年开始）
            end_date: 结束日期（默认到今天）
            resume: 是否从上次中断处继续
            max_retries: 最大重试次数

        Returns:
            同步结果统计
        """
        if priorities is None:
            priorities = [SyncPriority.HS300, SyncPriority.ZZ500, SyncPriority.ALL]

        if start_date is None:
            start_date = date(1990, 1, 1)

        if end_date is None:
            end_date = date.today()

        # 初始化数据库表
        self._init_daily_table()

        # 加载或创建状态
        state = self._load_or_create_state(resume, "daily")

        # 构建任务列表
        if not state.tasks:
            self._build_tasks(state, priorities, start_date, end_date)
            self._save_state(state)

        # 执行同步
        result = self._execute_sync(state, start_date, end_date, max_retries)

        return result

    def _init_daily_table(self) -> None:
        """初始化日线数据表（包含adj_factor列）"""
        from sqlalchemy import text

        create_stmt = """
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
                adj_factor REAL DEFAULT 1.0,
                UNIQUE(code, trade_date)
            )
        """

        create_index = """
            CREATE INDEX IF NOT EXISTS idx_daily_code_date
            ON stock_daily(code, trade_date)
        """

        with self.storage.engine.connect() as conn:
            conn.execute(text(create_stmt))
            conn.execute(text(create_index))
            conn.commit()

    def _build_tasks(
        self,
        state: SyncState,
        priorities: List[SyncPriority],
        start_date: date,
        end_date: date,
    ) -> None:
        """构建任务列表"""
        seen_codes = set()

        for priority in priorities:
            logger.info(f"获取 {priority.value} 股票池...")
            universe = self.get_universe(priority)

            for _, row in universe.iterrows():
                code = row["code"]
                if code in seen_codes:
                    continue

                seen_codes.add(code)

                task = SyncTask(
                    code=code,
                    name=row.get("name", ""),
                    priority=priority,
                    target_start_date=start_date.isoformat(),
                    target_end_date=end_date.isoformat(),
                )
                state.tasks[code] = asdict(task)

        state.total_tasks = len(state.tasks)
        logger.info(f"共 {state.total_tasks} 个股票待同步")

    def _execute_sync(
        self,
        state: SyncState,
        start_date: date,
        end_date: date,
        max_retries: int,
    ) -> Dict[str, Any]:
        """执行同步"""
        completed = 0
        failed = 0
        skipped = 0
        total_rows = 0

        # 按优先级排序任务
        priority_order = {
            SyncPriority.HS300.value: 0,
            SyncPriority.ZZ500.value: 1,
            SyncPriority.ZZ1000.value: 2,
            SyncPriority.ALL.value: 3,
        }

        tasks = sorted(
            state.tasks.items(),
            key=lambda x: (
                priority_order.get(x[1].get("priority", "all"), 99),
                x[0],
            ),
        )

        for idx, (code, task_dict) in enumerate(tasks):
            status = task_dict.get("status", "pending")

            # 跳过已完成或跳过的任务
            if status in ("completed", "skipped"):
                if status == "completed":
                    completed += 1
                else:
                    skipped += 1
                continue

            # 速率限制
            self.rate_limiter.wait()

            # 同步单个股票
            task_dict["status"] = "in_progress"
            state.updated_at = datetime.now().isoformat()

            try:
                rows = self._sync_single_stock(
                    code=code,
                    start_date=start_date,
                    end_date=end_date,
                    last_sync_date=task_dict.get("last_sync_date"),
                )

                task_dict["status"] = "completed"
                task_dict["synced_rows"] = rows
                task_dict["last_sync_date"] = end_date.isoformat()
                completed += 1
                total_rows += rows

                logger.info(
                    f"[{idx + 1}/{state.total_tasks}] {code} 同步完成，新增 {rows} 条"
                )

            except Exception as e:
                task_dict["retry_count"] = task_dict.get("retry_count", 0) + 1
                task_dict["error_msg"] = str(e)

                if task_dict["retry_count"] >= max_retries:
                    task_dict["status"] = "failed"
                    failed += 1
                    logger.error(f"[{idx + 1}/{state.total_tasks}] {code} 同步失败: {e}")

                    if self._on_error:
                        self._on_error(code, str(e))
                else:
                    task_dict["status"] = "pending"  # 留待重试
                    logger.warning(
                        f"[{idx + 1}/{state.total_tasks}] {code} 同步出错，将重试: {e}"
                    )

            # 更新状态
            state.tasks[code] = task_dict
            state.completed_tasks = completed
            state.failed_tasks = failed

            # 定期保存状态
            if (idx + 1) % 10 == 0:
                self._save_state(state)

            # 进度回调
            if self._on_progress:
                self._on_progress(completed + failed + skipped, state.total_tasks, code)

        # 最终保存状态
        self._save_state(state)

        return {
            "total": state.total_tasks,
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "total_rows": total_rows,
            "state_file": str(self.state_path),
        }

    def _sync_single_stock(
        self,
        code: str,
        start_date: date,
        end_date: date,
        last_sync_date: Optional[str] = None,
    ) -> int:
        """同步单个股票的日线数据"""
        # 确定实际起始日期（增量同步）
        actual_start = start_date

        if last_sync_date:
            # 从上次同步位置继续
            last_date = date.fromisoformat(last_sync_date)
            actual_start = last_date + timedelta(days=1)
        else:
            # 检查数据库中最新日期
            db_latest = self.storage.get_latest_date(code)
            if db_latest:
                actual_start = db_latest + timedelta(days=1)

        # 如果已是最新，跳过
        if actual_start > end_date:
            return 0

        # 获取数据
        df = self.loader.get_daily(
            code=code, start_date=actual_start, end_date=end_date, adj=None
        )

        if df.empty:
            return 0

        # 确保有必要的列
        required_cols = ["code", "trade_date", "open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                return 0

        # 处理列名
        if "adj_factor" not in df.columns:
            df["adj_factor"] = 1.0

        # 转换日期格式
        if not pd.api.types.is_datetime64_any_dtype(df["trade_date"]):
            df["trade_date"] = pd.to_datetime(df["trade_date"])

        df["trade_date"] = df["trade_date"].dt.strftime("%Y-%m-%d")

        # 选择需要的列
        save_cols = [
            "code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "adj_factor",
        ]
        save_cols = [c for c in save_cols if c in df.columns]

        # 保存到数据库
        df_save = df[save_cols].copy()
        rows = self._save_with_upsert(df_save, "stock_daily")

        return rows

    def _save_with_upsert(self, df: pd.DataFrame, table: str) -> int:
        """保存数据（冲突时更新）"""
        if df.empty:
            return 0

        from sqlalchemy import text

        # 使用 INSERT OR REPLACE
        cols = list(df.columns)
        placeholders = ", ".join([f":{c}" for c in cols])
        col_names = ", ".join(cols)

        sql = f"INSERT OR REPLACE INTO {table} ({col_names}) VALUES ({placeholders})"

        records = df.to_dict("records")

        with self.storage.engine.connect() as conn:
            for record in records:
                conn.execute(text(sql), record)
            conn.commit()

        return len(records)

    # ============================================================
    # 状态持久化
    # ============================================================

    def _load_or_create_state(self, resume: bool, data_type: str) -> SyncState:
        """加载或创建状态"""
        if resume and self.state_path.exists():
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # 验证是否是同类型任务
                if data.get("data_type") == data_type and data.get("source") == self.source:
                    logger.info(f"从 {self.state_path} 恢复同步状态")
                    return SyncState(**data)
            except Exception as e:
                logger.warning(f"加载状态失败: {e}，将创建新状态")

        # 创建新状态
        now = datetime.now().isoformat()
        return SyncState(
            session_id=f"{data_type}_{int(time.time())}",
            created_at=now,
            updated_at=now,
            source=self.source,
            data_type=data_type,
        )

    def _save_state(self, state: SyncState) -> None:
        """保存状态"""
        state.updated_at = datetime.now().isoformat()

        # 确保目录存在
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(asdict(state), f, ensure_ascii=False, indent=2)

    def clear_state(self) -> None:
        """清除状态文件（重新开始同步）"""
        if self.state_path.exists():
            self.state_path.unlink()
            logger.info("状态文件已清除")

    def get_sync_progress(self) -> Optional[Dict[str, Any]]:
        """获取当前同步进度"""
        if not self.state_path.exists():
            return None

        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            total = data.get("total_tasks", 0)
            completed = data.get("completed_tasks", 0)
            failed = data.get("failed_tasks", 0)

            return {
                "session_id": data.get("session_id"),
                "source": data.get("source"),
                "data_type": data.get("data_type"),
                "total": total,
                "completed": completed,
                "failed": failed,
                "pending": total - completed - failed,
                "progress": f"{completed}/{total} ({100 * completed / total:.1f}%)"
                if total > 0
                else "0/0",
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
            }
        except Exception:
            return None
