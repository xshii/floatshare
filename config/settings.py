"""全局配置"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Settings:
    """全局配置类"""

    # 项目路径
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    DATA_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    LOG_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent / "logs")

    # 数据源配置
    TUSHARE_TOKEN: Optional[str] = field(
        default_factory=lambda: os.getenv("TUSHARE_TOKEN")
    )

    # 数据库配置
    DATABASE_URL: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "sqlite:///data/floatshare.db"
        )
    )

    # 交易配置
    DEFAULT_COMMISSION: float = 0.0003  # 默认佣金率
    DEFAULT_SLIPPAGE: float = 0.001  # 默认滑点
    DEFAULT_INITIAL_CAPITAL: float = 1_000_000  # 默认初始资金

    # 日志配置
    LOG_LEVEL: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # 回测配置
    BACKTEST_START_DATE: str = "2020-01-01"
    BACKTEST_END_DATE: str = "2024-12-31"

    def __post_init__(self):
        """确保目录存在"""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
