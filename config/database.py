"""数据库配置"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DatabaseConfig:
    """数据库配置类"""

    # 主数据库
    host: str = "localhost"
    port: int = 5432
    database: str = "floatshare"
    username: str = "postgres"
    password: str = ""

    # Redis缓存
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    @property
    def postgres_url(self) -> str:
        """PostgreSQL连接URL"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def redis_url(self) -> str:
        """Redis连接URL"""
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """从环境变量创建配置"""
        import os

        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "floatshare"),
            username=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_db=int(os.getenv("REDIS_DB", "0")),
        )
