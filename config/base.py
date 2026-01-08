"""Pydantic 配置基类

使用 Pydantic 进行配置校验，支持：
- 类型自动转换
- 值范围校验
- 环境变量加载
- 配置文件加载 (YAML/JSON)
"""

import os
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================
# 枚举类型
# ============================================================


class Market(str, Enum):
    """市场类型"""
    SH = "SH"  # 上海
    SZ = "SZ"  # 深圳
    BJ = "BJ"  # 北京


class OrderType(str, Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"


class Direction(str, Enum):
    """交易方向"""
    BUY = "buy"
    SELL = "sell"


class LogLevel(str, Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# ============================================================
# 配置模型
# ============================================================


class DatabaseConfig(BaseModel):
    """数据库配置"""

    # SQLite (默认)
    sqlite_path: Path = Field(default=Path("data/floatshare.db"))

    # PostgreSQL (可选)
    pg_host: str = Field(default="localhost")
    pg_port: int = Field(default=5432, ge=1, le=65535)
    pg_database: str = Field(default="floatshare")
    pg_username: str = Field(default="postgres")
    pg_password: str = Field(default="")

    # Redis (可选)
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379, ge=1, le=65535)
    redis_db: int = Field(default=0, ge=0, le=15)
    redis_password: Optional[str] = Field(default=None)

    # 连接池
    pool_size: int = Field(default=5, ge=1, le=100)
    pool_timeout: int = Field(default=30, ge=1)

    @property
    def sqlite_url(self) -> str:
        return f"sqlite:///{self.sqlite_path}"

    @property
    def postgres_url(self) -> str:
        auth = f"{self.pg_username}:{self.pg_password}" if self.pg_password else self.pg_username
        return f"postgresql://{auth}@{self.pg_host}:{self.pg_port}/{self.pg_database}"

    @property
    def redis_url(self) -> str:
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"


class TradingConfig(BaseModel):
    """交易配置"""

    # 佣金设置
    commission_rate: float = Field(default=0.0003, ge=0, le=0.01)
    min_commission: float = Field(default=5.0, ge=0)

    # 印花税（卖出时收取）
    stamp_duty: float = Field(default=0.001, ge=0, le=0.01)

    # 过户费
    transfer_fee: float = Field(default=0.00001, ge=0)

    # 滑点设置
    slippage: float = Field(default=0.001, ge=0, le=0.1)

    # 交易限制
    max_position_pct: float = Field(default=0.25, gt=0, le=1.0)
    min_trade_quantity: int = Field(default=100, ge=1)

    # 涨跌停限制
    price_limit_normal: float = Field(default=0.10, ge=0, le=0.5)
    price_limit_st: float = Field(default=0.05, ge=0, le=0.2)
    price_limit_kcb: float = Field(default=0.20, ge=0, le=0.5)  # 科创板

    @field_validator("max_position_pct")
    @classmethod
    def validate_position_pct(cls, v: float) -> float:
        if v > 1.0:
            raise ValueError("单只股票最大持仓比例不能超过 100%")
        return v

    def calculate_commission(self, amount: float, direction: Direction) -> float:
        """计算交易费用"""
        commission = max(amount * self.commission_rate, self.min_commission)
        stamp = amount * self.stamp_duty if direction == Direction.SELL else 0
        transfer = amount * self.transfer_fee
        return commission + stamp + transfer


class BacktestConfig(BaseModel):
    """回测配置"""

    initial_capital: float = Field(default=1_000_000, gt=0)
    start_date: date = Field(default=date(2020, 1, 1))
    end_date: date = Field(default=date(2024, 12, 31))
    benchmark: str = Field(default="000300.SH")

    # 风控限制
    max_drawdown: float = Field(default=0.2, gt=0, le=1.0)
    daily_loss_limit: float = Field(default=0.05, gt=0, le=1.0)

    @model_validator(mode="after")
    def validate_dates(self) -> "BacktestConfig":
        if self.start_date >= self.end_date:
            raise ValueError("开始日期必须早于结束日期")
        return self


class DataSourceConfig(BaseModel):
    """数据源配置"""

    # 主数据源
    primary_source: str = Field(default="akshare")
    fallback_sources: List[str] = Field(default=["baostock"])

    # Tushare
    tushare_token: Optional[str] = Field(default=None)

    # 速率限制
    requests_per_minute: int = Field(default=30, ge=1, le=1000)
    batch_size: int = Field(default=50, ge=1)
    batch_pause: float = Field(default=30.0, ge=0)

    # 重试
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay: float = Field(default=1.0, ge=0)

    @field_validator("primary_source", "fallback_sources")
    @classmethod
    def validate_source(cls, v: Union[str, List[str]]) -> Union[str, List[str]]:
        valid_sources = {"akshare", "tushare", "baostock", "eastmoney"}
        if isinstance(v, str):
            if v not in valid_sources:
                raise ValueError(f"不支持的数据源: {v}, 可选: {valid_sources}")
        elif isinstance(v, list):
            for source in v:
                if source not in valid_sources:
                    raise ValueError(f"不支持的数据源: {source}, 可选: {valid_sources}")
        return v


class CacheConfig(BaseModel):
    """缓存配置"""

    enabled: bool = Field(default=True)
    backend: str = Field(default="memory")  # memory, redis
    max_size: int = Field(default=1000, ge=100)
    default_ttl: int = Field(default=3600, ge=0)  # 秒，0 表示不过期

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        if v not in {"memory", "redis"}:
            raise ValueError(f"不支持的缓存后端: {v}, 可选: memory, redis")
        return v


class LoggingConfig(BaseModel):
    """日志配置"""

    level: LogLevel = Field(default=LogLevel.INFO)
    log_dir: Path = Field(default=Path("logs"))
    log_file: str = Field(default="floatshare.log")
    max_bytes: int = Field(default=50 * 1024 * 1024, ge=1024)  # 50MB
    backup_count: int = Field(default=10, ge=0)
    rotation: str = Field(default="size")  # size, time
    console: bool = Field(default=True)

    @field_validator("rotation")
    @classmethod
    def validate_rotation(cls, v: str) -> str:
        if v not in {"size", "time"}:
            raise ValueError(f"不支持的轮转方式: {v}, 可选: size, time")
        return v


# ============================================================
# 主配置类
# ============================================================


class AppSettings(BaseSettings):
    """应用主配置

    支持从环境变量和 .env 文件加载
    """

    model_config = SettingsConfigDict(
        env_prefix="FLOATSHARE_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # 项目路径
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")

    # 子配置
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    data_source: DataSourceConfig = Field(default_factory=DataSourceConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def ensure_dirs(self) -> "AppSettings":
        """确保必要目录存在"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logging.log_dir.mkdir(parents=True, exist_ok=True)
        return self

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典"""
        return self.model_dump()

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "AppSettings":
        """从 YAML 文件加载"""
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "AppSettings":
        """从 JSON 文件加载"""
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def save_yaml(self, path: Union[str, Path]) -> None:
        """保存为 YAML 文件"""
        import yaml

        data = self.model_dump(mode="json")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

    def save_json(self, path: Union[str, Path]) -> None:
        """保存为 JSON 文件"""
        import json

        data = self.model_dump(mode="json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================
# 全局配置实例
# ============================================================

_settings: Optional[AppSettings] = None


def get_settings() -> AppSettings:
    """获取全局配置实例"""
    global _settings
    if _settings is None:
        _settings = AppSettings()
    return _settings


def reload_settings() -> AppSettings:
    """重新加载配置"""
    global _settings
    _settings = AppSettings()
    return _settings
