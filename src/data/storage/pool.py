"""数据库连接池模块

提供连接池管理：
- SQLAlchemy 连接池配置
- PostgreSQL/SQLite 支持
- 连接健康检查
- 连接统计
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Union

from sqlalchemy import create_engine, event, pool, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)


class PoolType(Enum):
    """连接池类型"""
    STATIC = "static"  # 固定大小池 (QueuePool)
    OVERFLOW = "overflow"  # 可溢出池 (QueuePool with overflow)
    NULL = "null"  # 无池 (NullPool)
    SINGLE = "single"  # 单连接 (SingletonThreadPool)


@dataclass
class PoolConfig:
    """连接池配置"""
    pool_type: PoolType = PoolType.OVERFLOW
    pool_size: int = 5  # 核心连接数
    max_overflow: int = 10  # 最大溢出连接数
    pool_timeout: int = 30  # 获取连接超时（秒）
    pool_recycle: int = 3600  # 连接回收时间（秒）
    pool_pre_ping: bool = True  # 使用前检查连接
    echo: bool = False  # 是否打印 SQL
    echo_pool: bool = False  # 是否打印连接池日志


@dataclass
class PoolStats:
    """连接池统计"""
    size: int  # 当前池大小
    checked_in: int  # 空闲连接数
    checked_out: int  # 使用中连接数
    overflow: int  # 溢出连接数
    invalid: int  # 无效连接数

    @property
    def total_connections(self) -> int:
        return self.checked_in + self.checked_out

    @property
    def usage_pct(self) -> float:
        if self.size == 0:
            return 0
        return self.checked_out / self.size * 100


class ConnectionPool:
    """数据库连接池"""

    def __init__(
        self,
        url: str,
        config: Optional[PoolConfig] = None,
    ):
        """
        Args:
            url: 数据库连接 URL
                - SQLite: sqlite:///path/to/db.sqlite
                - PostgreSQL: postgresql://user:pass@host:port/db
            config: 连接池配置
        """
        self.url = url
        self.config = config or PoolConfig()
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None

    @property
    def engine(self) -> Engine:
        """获取数据库引擎（延迟初始化）"""
        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine

    def _create_engine(self) -> Engine:
        """创建数据库引擎"""
        cfg = self.config

        # 根据池类型选择池类
        pool_class = None
        pool_kwargs = {}

        if "sqlite" in self.url:
            # SQLite 使用 SingletonThreadPool（同一线程共享连接）
            pool_class = pool.SingletonThreadPool
            pool_kwargs = {}
        else:
            # 其他数据库使用配置的池类型
            if cfg.pool_type == PoolType.NULL:
                pool_class = pool.NullPool
            elif cfg.pool_type == PoolType.SINGLE:
                pool_class = pool.SingletonThreadPool
            else:
                pool_class = pool.QueuePool
                pool_kwargs = {
                    "pool_size": cfg.pool_size,
                    "max_overflow": cfg.max_overflow,
                    "pool_timeout": cfg.pool_timeout,
                    "pool_recycle": cfg.pool_recycle,
                }

        engine = create_engine(
            self.url,
            poolclass=pool_class,
            pool_pre_ping=cfg.pool_pre_ping,
            echo=cfg.echo,
            echo_pool=cfg.echo_pool,
            **pool_kwargs,
        )

        # 添加连接事件监听
        self._setup_events(engine)

        logger.info(
            f"数据库连接池已创建: {self._safe_url(self.url)}, "
            f"池类型: {pool_class.__name__}"
        )

        return engine

    def _setup_events(self, engine: Engine) -> None:
        """设置连接事件"""

        @event.listens_for(engine, "connect")
        def on_connect(dbapi_conn, connection_record):
            logger.debug("数据库连接已建立")

        @event.listens_for(engine, "checkout")
        def on_checkout(dbapi_conn, connection_record, connection_proxy):
            logger.debug("连接已从池中取出")

        @event.listens_for(engine, "checkin")
        def on_checkin(dbapi_conn, connection_record):
            logger.debug("连接已归还到池中")

    def _safe_url(self, url: str) -> str:
        """隐藏密码的 URL"""
        if "@" in url:
            # postgresql://user:pass@host/db -> postgresql://user:***@host/db
            parts = url.split("@")
            auth = parts[0]
            if ":" in auth.split("//")[-1]:
                return auth.rsplit(":", 1)[0] + ":***@" + parts[1]
        return url

    @property
    def session_factory(self) -> sessionmaker:
        """获取 Session 工厂"""
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self.engine)
        return self._session_factory

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """
        获取数据库会话（上下文管理器）

        Example:
            with pool.session() as session:
                session.execute(text("SELECT 1"))
        """
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @contextmanager
    def connection(self) -> Generator:
        """
        获取原始连接（上下文管理器）

        Example:
            with pool.connection() as conn:
                conn.execute(text("SELECT 1"))
        """
        conn = self.engine.connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """执行 SQL 查询"""
        with self.connection() as conn:
            result = conn.execute(text(query), params or {})
            return result

    def get_stats(self) -> Optional[PoolStats]:
        """获取连接池统计"""
        if self._engine is None:
            return None

        pool_obj = self._engine.pool

        # QueuePool 有统计信息
        if hasattr(pool_obj, "size"):
            return PoolStats(
                size=pool_obj.size(),
                checked_in=pool_obj.checkedin(),
                checked_out=pool_obj.checkedout(),
                overflow=pool_obj.overflow() if hasattr(pool_obj, "overflow") else 0,
                invalid=pool_obj.invalidatedcount() if hasattr(pool_obj, "invalidatedcount") else 0,
            )

        return None

    def health_check(self) -> bool:
        """检查连接池健康状态"""
        try:
            with self.connection() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"连接池健康检查失败: {e}")
            return False

    def dispose(self) -> None:
        """关闭所有连接"""
        if self._engine:
            self._engine.dispose()
            logger.info("数据库连接池已关闭")

    def __enter__(self) -> "ConnectionPool":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.dispose()


# ============================================================
# 全局连接池
# ============================================================

_pools: Dict[str, ConnectionPool] = {}


def get_pool(
    name: str = "default",
    url: Optional[str] = None,
    config: Optional[PoolConfig] = None,
) -> ConnectionPool:
    """
    获取命名连接池

    Args:
        name: 连接池名称
        url: 数据库 URL（首次创建时需要）
        config: 连接池配置

    Returns:
        ConnectionPool 实例
    """
    if name not in _pools:
        if url is None:
            raise ValueError(f"连接池 '{name}' 不存在，首次创建需要提供 url")
        _pools[name] = ConnectionPool(url, config)

    return _pools[name]


def close_pool(name: str = "default") -> None:
    """关闭命名连接池"""
    if name in _pools:
        _pools[name].dispose()
        del _pools[name]


def close_all_pools() -> None:
    """关闭所有连接池"""
    for name in list(_pools.keys()):
        close_pool(name)


# ============================================================
# 便捷函数
# ============================================================


def create_sqlite_pool(
    db_path: Union[str, Path],
    echo: bool = False,
) -> ConnectionPool:
    """创建 SQLite 连接池"""
    url = f"sqlite:///{db_path}"
    config = PoolConfig(echo=echo)
    return ConnectionPool(url, config)


def create_postgres_pool(
    host: str = "localhost",
    port: int = 5432,
    database: str = "floatshare",
    username: str = "postgres",
    password: str = "",
    pool_size: int = 5,
    max_overflow: int = 10,
    echo: bool = False,
) -> ConnectionPool:
    """创建 PostgreSQL 连接池"""
    url = f"postgresql://{username}:{password}@{host}:{port}/{database}"
    config = PoolConfig(
        pool_type=PoolType.OVERFLOW,
        pool_size=pool_size,
        max_overflow=max_overflow,
        echo=echo,
    )
    return ConnectionPool(url, config)
