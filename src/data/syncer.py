"""数据同步管理器

用于批量爬取历史数据，支持：
- 股票池管理（按优先级：沪深300、中证500、全市场）
- 增量同步（只获取缺失数据）
- 进度追踪（支持多日中断恢复）
- 速率控制（防止被封禁）
- 多数据源自动降级/并行获取
"""

import json
import time
import random
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field, asdict

import pandas as pd

from src.data.loader import DataLoader
from src.data.storage.database import DatabaseStorage

logger = logging.getLogger(__name__)


# ============================================================
# 代理池管理
# ============================================================


class ProxyPool:
    """代理IP池"""

    def __init__(self, proxies: Optional[List[str]] = None):
        """
        Args:
            proxies: 代理列表，如 ["http://ip1:port", "http://ip2:port"]
        """
        self.proxies = proxies or []
        self.current_index = 0
        self.failed_proxies: Dict[str, int] = {}  # proxy -> fail_count
        self.max_fails = 3

    def get_proxy(self) -> Optional[str]:
        """获取下一个可用代理（轮换）"""
        if not self.proxies:
            return None

        available = [p for p in self.proxies if self.failed_proxies.get(p, 0) < self.max_fails]

        if not available:
            # 所有代理都失败了，重置
            self.failed_proxies.clear()
            available = self.proxies

        proxy = available[self.current_index % len(available)]
        self.current_index += 1
        return proxy

    def report_failure(self, proxy: str) -> None:
        """报告代理失败"""
        self.failed_proxies[proxy] = self.failed_proxies.get(proxy, 0) + 1
        logger.warning(f"代理 {proxy} 失败 {self.failed_proxies[proxy]} 次")

    def report_success(self, proxy: str) -> None:
        """报告代理成功，重置失败计数"""
        if proxy in self.failed_proxies:
            self.failed_proxies[proxy] = 0

    def add_proxies(self, proxies: List[str]) -> None:
        """添加代理"""
        for p in proxies:
            if p not in self.proxies:
                self.proxies.append(p)

    @property
    def available_count(self) -> int:
        """可用代理数量"""
        return len([p for p in self.proxies if self.failed_proxies.get(p, 0) < self.max_fails])


# ============================================================
# 多数据源管理
# ============================================================


@dataclass
class SourceHealth:
    """数据源健康状态"""

    name: str
    success_count: int = 0
    fail_count: int = 0
    consecutive_fails: int = 0
    consecutive_empty: int = 0  # 连续返回空数据次数（可能被限流）
    last_fail_time: Optional[float] = None
    disabled_until: Optional[float] = None  # 禁用到什么时候
    avg_response_time: float = 0.0  # 平均响应时间

    @property
    def success_rate(self) -> float:
        """成功率"""
        total = self.success_count + self.fail_count
        return self.success_count / total if total > 0 else 1.0

    @property
    def is_available(self) -> bool:
        """是否可用"""
        if self.disabled_until is None:
            return True
        return time.time() >= self.disabled_until

    @property
    def may_be_rate_limited(self) -> bool:
        """可能被限流（连续空数据）"""
        return self.consecutive_empty >= 5

    def record_success(self, response_time: float) -> None:
        """记录成功"""
        self.success_count += 1
        self.consecutive_fails = 0
        self.consecutive_empty = 0  # 有数据，重置空计数
        # 更新平均响应时间（滑动平均）
        if self.avg_response_time == 0:
            self.avg_response_time = response_time
        else:
            self.avg_response_time = 0.9 * self.avg_response_time + 0.1 * response_time

    def record_empty(self) -> None:
        """记录返回空数据（可能被限流）"""
        self.consecutive_empty += 1
        if self.consecutive_empty >= 5:
            logger.warning(
                f"数据源 {self.name} 连续 {self.consecutive_empty} 次返回空数据，"
                "可能被限流，建议切换数据源或暂停"
            )

    def record_failure(self, backoff_base: float = 60.0) -> None:
        """记录失败，指数退避"""
        self.fail_count += 1
        self.consecutive_fails += 1
        self.last_fail_time = time.time()

        # 连续失败达到阈值，临时禁用
        if self.consecutive_fails >= 3:
            # 指数退避：60s, 120s, 240s, ...
            backoff = backoff_base * (2 ** (self.consecutive_fails - 3))
            backoff = min(backoff, 3600)  # 最多禁用1小时
            self.disabled_until = time.time() + backoff
            logger.warning(
                f"数据源 {self.name} 连续失败 {self.consecutive_fails} 次，"
                f"禁用 {backoff:.0f} 秒"
            )


class SourcePool:
    """数据源池 - 管理多个数据源，支持自动降级和并行获取"""

    def __init__(
        self,
        sources: Optional[List[str]] = None,
        parallel: bool = False,
        max_workers: int = 3,
    ):
        """
        Args:
            sources: 数据源列表，按优先级排序，如 ["akshare", "tushare", "eastmoney"]
            parallel: 是否并行获取
            max_workers: 并行时的最大工作线程数
        """
        if sources is None:
            sources = ["akshare", "eastmoney"]  # 默认数据源

        self.sources = sources
        self.parallel = parallel
        self.max_workers = max_workers

        # 初始化数据源加载器和健康状态
        self.loaders: Dict[str, DataLoader] = {}
        self.health: Dict[str, SourceHealth] = {}

        for source in sources:
            try:
                self.loaders[source] = DataLoader(source=source)
                self.health[source] = SourceHealth(name=source)
            except Exception as e:
                logger.warning(f"初始化数据源 {source} 失败: {e}")

    def get_available_sources(self) -> List[str]:
        """获取可用的数据源列表（按健康度排序）"""
        available = [
            name for name, h in self.health.items()
            if h.is_available and name in self.loaders
        ]

        # 按成功率和响应时间排序
        def score(name: str) -> Tuple[float, float]:
            h = self.health[name]
            return (-h.success_rate, h.avg_response_time)

        return sorted(available, key=score)

    def fetch_daily(
        self,
        code: str,
        start_date: date,
        end_date: date,
        adj: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, str]:
        """
        获取日线数据

        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            adj: 复权方式

        Returns:
            (DataFrame, 数据源名称)
        """
        if self.parallel:
            return self._fetch_parallel(code, start_date, end_date, adj)
        else:
            return self._fetch_sequential(code, start_date, end_date, adj)

    def _fetch_sequential(
        self,
        code: str,
        start_date: date,
        end_date: date,
        adj: Optional[str],
    ) -> Tuple[pd.DataFrame, str]:
        """顺序获取（自动降级）"""
        sources = self.get_available_sources()

        if not sources:
            raise RuntimeError("所有数据源均不可用")

        last_error = None
        for source in sources:
            loader = self.loaders[source]
            health = self.health[source]

            # 如果该数据源可能被限流，跳过尝试下一个
            if health.may_be_rate_limited and len(sources) > 1:
                logger.info(f"数据源 {source} 可能被限流，尝试下一个")
                continue

            try:
                start_time = time.time()
                df = loader.get_daily(
                    code=code,
                    start_date=start_date,
                    end_date=end_date,
                    adj=adj,
                )
                elapsed = time.time() - start_time

                if not df.empty:
                    health.record_success(elapsed)
                    return df, source

                # 空数据记录（用于检测限流）
                health.record_empty()
                return df, source

            except Exception as e:
                health.record_failure()
                last_error = e
                logger.warning(f"数据源 {source} 获取 {code} 失败: {e}")
                continue

        # 所有数据源都失败
        raise RuntimeError(f"所有数据源获取失败: {last_error}")

    def _fetch_parallel(
        self,
        code: str,
        start_date: date,
        end_date: date,
        adj: Optional[str],
    ) -> Tuple[pd.DataFrame, str]:
        """并行获取（取最快返回的有效结果）"""
        sources = self.get_available_sources()

        if not sources:
            raise RuntimeError("所有数据源均不可用")

        def fetch_from_source(source: str) -> Tuple[pd.DataFrame, str, float]:
            loader = self.loaders[source]
            start_time = time.time()
            df = loader.get_daily(
                code=code,
                start_date=start_date,
                end_date=end_date,
                adj=adj,
            )
            elapsed = time.time() - start_time
            return df, source, elapsed

        with ThreadPoolExecutor(max_workers=min(len(sources), self.max_workers)) as executor:
            futures = {
                executor.submit(fetch_from_source, source): source
                for source in sources
            }

            for future in as_completed(futures):
                source = futures[future]
                try:
                    df, src, elapsed = future.result()
                    if not df.empty:
                        self.health[src].record_success(elapsed)
                        # 取消其他任务
                        for f in futures:
                            f.cancel()
                        return df, src
                except Exception as e:
                    self.health[source].record_failure()
                    logger.warning(f"数据源 {source} 并行获取 {code} 失败: {e}")
                    continue

        # 所有都失败了，尝试返回空 DataFrame
        return pd.DataFrame(), sources[0] if sources else "unknown"

    def get_health_report(self) -> Dict[str, Dict[str, Any]]:
        """获取健康报告"""
        return {
            name: {
                "success_rate": f"{h.success_rate:.1%}",
                "success_count": h.success_count,
                "fail_count": h.fail_count,
                "consecutive_fails": h.consecutive_fails,
                "consecutive_empty": h.consecutive_empty,
                "avg_response_time": f"{h.avg_response_time:.2f}s",
                "is_available": h.is_available,
                "may_be_rate_limited": h.may_be_rate_limited,
            }
            for name, h in self.health.items()
        }


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
    """速率限制器（带随机抖动）"""

    def __init__(
        self,
        requests_per_minute: int = 60,
        batch_size: int = 50,
        batch_pause: float = 10.0,
        jitter: float = 0.5,
    ):
        """
        Args:
            requests_per_minute: 每分钟最大请求数
            batch_size: 批次大小（每处理多少个后暂停）
            batch_pause: 批次间暂停时间（秒）
            jitter: 随机抖动比例 (0-1)，避免固定频率特征
        """
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.batch_size = batch_size
        self.batch_pause = batch_pause
        self.jitter = jitter
        self.request_count = 0
        self.last_request_time = 0.0

    def _add_jitter(self, base_time: float) -> float:
        """添加随机抖动"""
        if self.jitter <= 0:
            return base_time
        jitter_range = base_time * self.jitter
        return base_time + random.uniform(-jitter_range, jitter_range)

    def wait(self) -> None:
        """等待以满足速率限制（带随机抖动）"""
        now = time.time()
        elapsed = now - self.last_request_time

        # 计算带抖动的等待时间
        wait_time = self._add_jitter(self.min_interval)

        if elapsed < wait_time:
            time.sleep(wait_time - elapsed)

        self.last_request_time = time.time()
        self.request_count += 1

        # 批次暂停（也带抖动）
        if self.request_count % self.batch_size == 0:
            pause = self._add_jitter(self.batch_pause)
            logger.info(f"已处理 {self.request_count} 个请求，暂停 {pause:.1f} 秒...")
            time.sleep(pause)


class DataSyncer:
    """数据同步管理器"""

    def __init__(
        self,
        source: Optional[str] = None,
        sources: Optional[List[str]] = None,
        parallel: bool = False,
        db_path: Optional[str] = None,
        state_path: Optional[str] = None,
    ):
        """
        Args:
            source: 单数据源（向后兼容）
            sources: 多数据源列表，按优先级排序，如 ["akshare", "tushare", "eastmoney"]
            parallel: 是否并行获取（多数据源模式）
            db_path: 数据库路径
            state_path: 状态文件路径
        """
        # 处理数据源配置
        if sources is not None:
            # 多数据源模式
            self.source = sources[0]
            self.source_pool = SourcePool(sources=sources, parallel=parallel)
            self.multi_source = True
        elif source is not None:
            # 单数据源模式（向后兼容）
            self.source = source
            self.loader = DataLoader(source=source)
            self.source_pool = None
            self.multi_source = False
        else:
            # 默认：多数据源模式
            self.source = "akshare"
            self.source_pool = SourcePool(sources=["akshare", "eastmoney"], parallel=parallel)
            self.multi_source = True

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
        import time as _time
        sync_start_time = _time.time()

        logger.info("=" * 50)
        logger.info(f"开始同步任务: {state.session_id}")
        logger.info(f"数据源: {self.source}, 日期范围: {start_date} ~ {end_date}")
        logger.info(f"待同步: {state.total_tasks} 只股票")
        logger.info("=" * 50)

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

        # 计算耗时
        sync_elapsed = _time.time() - sync_start_time
        sync_minutes = sync_elapsed / 60

        result = {
            "total": state.total_tasks,
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "total_rows": total_rows,
            "elapsed_seconds": sync_elapsed,
            "state_file": str(self.state_path),
        }

        # 多数据源模式下，附带健康报告
        if self.multi_source and self.source_pool:
            result["source_health"] = self.source_pool.get_health_report()

        # 输出同步结束汇总日志
        logger.info("=" * 50)
        logger.info("同步任务完成")
        logger.info(f"会话ID: {state.session_id}")
        logger.info(f"耗时: {sync_minutes:.1f} 分钟 ({sync_elapsed:.0f} 秒)")
        logger.info(f"成功: {completed}, 失败: {failed}, 跳过: {skipped}")
        logger.info(f"新增数据: {total_rows} 条")
        if completed > 0:
            avg_time = sync_elapsed / completed
            logger.info(f"平均每只: {avg_time:.2f} 秒")
        if failed > 0:
            logger.warning(f"有 {failed} 只股票同步失败，请检查日志")
        logger.info("=" * 50)

        return result

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
                logger.debug(f"{code} 数据库最新日期: {db_latest}, 从 {actual_start} 开始增量同步")

        # 如果已是最新，跳过
        if actual_start > end_date:
            logger.debug(f"{code} 已是最新，跳过")
            return 0

        # 获取数据（支持多数据源）
        if self.multi_source and self.source_pool:
            df, used_source = self.source_pool.fetch_daily(
                code=code, start_date=actual_start, end_date=end_date, adj=None
            )
            logger.debug(f"{code} 数据来自 {used_source}, 获取 {len(df)} 条")
        else:
            df = self.loader.get_daily(
                code=code, start_date=actual_start, end_date=end_date, adj=None
            )
            logger.debug(f"{code} 获取 {len(df)} 条数据")

        if df.empty:
            logger.debug(f"{code} 无新数据")
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

        # 数据校验
        from src.data.validator import DataValidator
        validator = DataValidator(max_pct_change=22.0)
        df, validation_result = validator.filter_valid(df, code)

        if validation_result.rows_invalid > 0:
            logger.warning(
                f"{code} 过滤异常数据: {validation_result.rows_invalid} 条 "
                f"(剩余 {len(df)} 条)"
            )

        if df.empty:
            logger.debug(f"{code} 校验后无有效数据")
            return 0

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

    def get_source_health(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """获取数据源健康状态（仅多数据源模式）"""
        if self.multi_source and self.source_pool:
            return self.source_pool.get_health_report()
        return None

    def set_parallel(self, parallel: bool = True) -> "DataSyncer":
        """设置是否并行获取"""
        if self.source_pool:
            self.source_pool.parallel = parallel
        return self
