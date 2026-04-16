#!/usr/bin/env python
"""初始化本地 SQLite 数据库表。"""

from __future__ import annotations

from floatshare.infrastructure.storage import DatabaseStorage
from floatshare.observability import logger


def main() -> None:
    logger.info("初始化数据库...")
    storage = DatabaseStorage()
    storage.init_tables()
    logger.info(f"数据库初始化完成: {storage.db_path}")


if __name__ == "__main__":
    main()
